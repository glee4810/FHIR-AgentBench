#!/usr/bin/env python3
"""Bare-PKCE login (no client_id) against a self-hosted Medplum -> prints an access token."""
import secrets, hashlib, base64, json, urllib.request, urllib.parse, urllib.error, sys, os
B = os.environ.get("MEDPLUM_BASE_URL", "http://localhost:8103")


def b64u(b): return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def post(path, data, form=False):
    body = urllib.parse.urlencode(data).encode() if form else json.dumps(data).encode()
    ct = "application/x-www-form-urlencoded" if form else "application/json"
    req = urllib.request.Request(B + path, data=body, headers={"Content-Type": ct})
    try:
        return json.load(urllib.request.urlopen(req, timeout=30))
    except urllib.error.HTTPError as e:
        return {"_http": e.code, "_body": e.read().decode()[:200]}


v = b64u(secrets.token_bytes(32))
ch = b64u(hashlib.sha256(v.encode()).digest())
login = post("/auth/login", {"email": os.environ.get("MEDPLUM_EMAIL", "admin@example.com"),
                             "password": os.environ.get("MEDPLUM_PASSWORD", "medplum_admin"),
                             "scope": "openid", "codeChallenge": ch, "codeChallengeMethod": "S256"})
if "code" not in login:
    sys.stderr.write("LOGIN_FAIL " + json.dumps(login)); sys.exit(1)
tok = post("/oauth2/token", {"grant_type": "authorization_code", "code": login["code"],
                             "code_verifier": v}, form=True)
if "access_token" in tok:
    sys.stdout.write(tok["access_token"])
else:
    sys.stderr.write("TOKEN_FAIL " + json.dumps(tok)); sys.exit(1)
