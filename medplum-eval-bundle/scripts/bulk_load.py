#!/usr/bin/env python3
"""Parallel FHIR batch-bundle loader. PUT entries preserve resource ids (so true_fhir_ids match).
Token cached per process + refresh-on-401, retry on 429/5xx. Usage: python3 bulk_load.py file1.ndjson.gz ..."""
import gzip, json, sys, os, threading, time, secrets, hashlib, base64, urllib.request, urllib.error, urllib.parse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

B = os.environ.get("MEDPLUM_BASE_URL", "http://localhost:8103")
W = int(os.environ.get("W", "16"))
BATCH = int(os.environ.get("BATCH", "150"))
_tok = [None]
_lk = threading.Lock()


def b64u(b): return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def fetch_token():
    v = b64u(secrets.token_bytes(32))
    ch = b64u(hashlib.sha256(v.encode()).digest())

    def p(path, d, form=False):
        body = urllib.parse.urlencode(d).encode() if form else json.dumps(d).encode()
        ct = "application/x-www-form-urlencoded" if form else "application/json"
        return json.load(urllib.request.urlopen(urllib.request.Request(B + path, data=body, headers={"Content-Type": ct})))

    lg = p("/auth/login", {"email": "admin@example.com", "password": "medplum_admin",
                           "scope": "openid", "codeChallenge": ch, "codeChallengeMethod": "S256"})
    return p("/oauth2/token", {"grant_type": "authorization_code", "code": lg["code"],
                               "code_verifier": v}, form=True)["access_token"]


def refresh():
    with _lk:
        _tok[0] = fetch_token()


refresh()
ok = [0]; fail = [0]; cl = threading.Lock()


def post_bundle(entries):
    data = json.dumps({"resourceType": "Bundle", "type": "batch", "entry": entries}).encode()
    for _ in range(6):
        try:
            req = urllib.request.Request(B + "/fhir/R4", data=data,
                                         headers={"Authorization": "Bearer " + _tok[0], "Content-Type": "application/fhir+json"})
            resp = json.load(urllib.request.urlopen(req, timeout=120))
            f = sum(1 for e in resp.get("entry", []) if not str(e.get("response", {}).get("status", "")).startswith(("200", "201")))
            with cl:
                ok[0] += len(entries) - f; fail[0] += f
            return
        except urllib.error.HTTPError as e:
            if e.code == 401:
                refresh()
            time.sleep(1 + secrets.randbelow(1500) / 1000.0)
        except Exception:
            time.sleep(1)
    with cl:
        fail[0] += len(entries)


def entries(rs):
    return [{"resource": r, "request": {"method": "PUT", "url": r["resourceType"] + "/" + r["id"]}} for r in rs]


def run(files):
    ex = ThreadPoolExecutor(max_workers=W); futs = set(); buf = []; t0 = time.time()

    def submit(e):
        nonlocal futs
        while len(futs) >= 2 * W:
            _d, futs = wait(futs, return_when=FIRST_COMPLETED)
        futs.add(ex.submit(post_bundle, e))

    for path in files:
        opn = gzip.open if path.endswith(".gz") else open
        with opn(path, "rt") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                buf.append(json.loads(line))
                if len(buf) >= BATCH:
                    submit(entries(buf)); buf = []
        if buf:
            submit(entries(buf)); buf = []
        print(f"  queued {os.path.basename(path)} ok={ok[0]} fail={fail[0]} t={int(time.time()-t0)}s", flush=True)
    ex.shutdown(wait=True)
    print(f"DONE ok={ok[0]} fail={fail[0]} in {time.time()-t0:.0f}s", flush=True)
    if fail[0] > 0:
        # an eval substrate must NOT silently finish with a partial load (it would corrupt every
        # downstream true_fhir_ids match). Fail loudly so the caller knows the store is incomplete.
        print(f"ERROR: {fail[0]} FHIR writes failed — the store is INCOMPLETE. Aborting nonzero.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    run(sys.argv[1:])
