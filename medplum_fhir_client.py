#!/usr/bin/env python3
"""Medplum backend for the benchmark's FHIRClient — so the STANDARD reference agents
(single/multi_turn_resource and *_code_resource) can run against a self-hosted Medplum
instead of GCP Cloud Healthcare API, with zero change to the agent logic or tools.

The benchmark's FHIRClient query methods (search_with_pagination, _fetch_resources_with_pagination,
get_resources_by_resource_ids) only use `self.session.get(url)` + `self.fhir_store_url`, and issue
plain FHIR R4 searches (e.g. "Encounter?patient=X") with `link[rel=next]` pagination. Medplum serves
exactly that. So we just swap the base URL (-> /fhir/R4) and the session (-> bearer token, 401-refresh)
and inherit everything else. FHIR R4 is FHIR R4: the agents are unchanged, only the backend differs.

Activated transparently: get_fhir_client() returns this when MEDPLUM_BASE_URL is set.
"""
import os
import json
import secrets
import hashlib
import base64
import urllib.request
import urllib.parse
import urllib.error

from fhir_client import FHIRClient


def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def medplum_login(base_url: str) -> str:
    """Bare-PKCE login (no client_id) against a self-hosted Medplum -> access token. Mirrors the bundle's
    get_token.py and treatment_mcp_server.py exactly."""
    base = base_url.rstrip("/")
    verifier = _b64u(secrets.token_bytes(32))
    challenge = _b64u(hashlib.sha256(verifier.encode()).digest())

    def _post(path, data, form=False):
        body = urllib.parse.urlencode(data).encode() if form else json.dumps(data).encode()
        ct = "application/x-www-form-urlencoded" if form else "application/json"
        req = urllib.request.Request(base + path, data=body, headers={"Content-Type": ct})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)

    login = _post("/auth/login", {
        "email": os.environ.get("MEDPLUM_EMAIL", "admin@example.com"),
        "password": os.environ.get("MEDPLUM_PASSWORD", "medplum_admin"),
        "scope": "openid", "codeChallenge": challenge, "codeChallengeMethod": "S256",
    })
    if "code" not in login:
        raise RuntimeError(f"Medplum login failed: {login}")
    tok = _post("/oauth2/token", {
        "grant_type": "authorization_code", "code": login["code"], "code_verifier": verifier,
    }, form=True)
    if "access_token" not in tok:
        raise RuntimeError(f"Medplum token exchange failed: {tok}")
    return tok["access_token"]


import time as _time
_TOKEN_CACHE = {"value": None, "exp": 0.0}


def cached_token(base_url: str, force: bool = False) -> str:
    """Module-global token cache. The benchmark calls get_fhir_client() per tool invocation (dozens of
    times), and a login-per-client trips Medplum's AUTH rate limit. Reuse one token across all clients;
    only re-login on expiry or a forced refresh (401)."""
    now = _time.time()
    if not force and _TOKEN_CACHE["value"] and now < _TOKEN_CACHE["exp"]:
        return _TOKEN_CACHE["value"]
    tok = medplum_login(base_url)
    _TOKEN_CACHE["value"] = tok
    _TOKEN_CACHE["exp"] = now + 50 * 60  # Medplum tokens last ~1h; refresh at 50 min
    return tok


class _Resp:
    def __init__(self, code, body):
        self.status_code = code
        self._body = body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}: {self._body[:300]}")

    def json(self):
        return json.loads(self._body or b"{}")


class _MedplumSession:
    """A drop-in for requests.Session.get(url) -> response, with bearer auth + one 401-refresh."""
    def __init__(self, base_url: str):
        self._base = base_url
        self._token = cached_token(base_url)

    def get(self, url):
        import time
        backoff = 1.0
        for attempt in range(15):
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self._token}"})
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    return _Resp(r.status, r.read())
            except urllib.error.HTTPError as e:
                if e.code == 401:
                    self._token = cached_token(self._base, force=True)  # token expired -> refresh (shared)
                    continue
                if e.code == 429:  # Medplum rate limit -> exponential backoff (don't poison the run with errors)
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30)
                    continue
                return _Resp(e.code, e.read())
            except Exception:  # transient network blip -> backoff + retry
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue
        return _Resp(429, b'{"error":"rate limited after retries"}')


class MedplumFHIRClient(FHIRClient):
    """FHIRClient backed by a self-hosted Medplum. Inherits all query/pagination logic unchanged."""
    def __init__(self, base_url: str = None):
        base = (base_url or os.environ.get("MEDPLUM_BASE_URL", "http://localhost:8103")).rstrip("/")
        self.fhir_store_url = base + "/fhir/R4"
        self.session = _MedplumSession(base)
        self.metadata = {"backend": "medplum", "fhir_store_url": self.fhir_store_url}

    def search_with_pagination(self, query_string):
        # Inject a large page size. Medplum paginates at 20/page by default, so a patient with 11k+
        # observations would take ~560 requests -> saturates Medplum's rate limit. _count=1000 (Medplum's
        # max page) drops that to ~12 requests. Same resources returned; far fewer calls = no 429 storm.
        if "_count=" not in query_string:
            sep = "&" if "?" in query_string else "?"
            query_string = f"{query_string}{sep}_count=1000"
        return super().search_with_pagination(query_string)

    def _fetch_resources_with_pagination(self, initial_resource_path):
        # Medplum caps OFFSET-based pagination at offset 10000. For very high-volume patients (>10k
        # resources of a type) the next-link eventually requests offset>10000 and 400s. Cap retrieval at
        # the first 10 pages (=10000 resources at _count=1000) and stop gracefully — ample for the QA
        # tasks, and the resource agent would overflow its context on that many anyway.
        all_resources, path = [], initial_resource_path
        for _ in range(10):
            resp = self.session.get(path)
            if resp.status_code >= 400:
                break  # offset cap or other error -> return what we have
            data = resp.json()
            for e in data.get("entry", []):
                all_resources.append(self.remove_fields(e["resource"], ["text", "meta"]))
            nxt = next((l.get("url") for l in data.get("link", []) if l.get("relation") == "next"), None)
            if not nxt:
                break
            path = nxt
        return all_resources
