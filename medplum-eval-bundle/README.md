# Reproducible FHIR-agent eval environment (Docker)

A one-command-ish substrate for the typed-FHIR-tool ablation eval: a self-hosted **Medplum**
(server + Postgres + Redis, MCP enabled) loaded with the **MIMIC-IV-on-FHIR demo** (100 real
de-identified ICU patients, open-access, ODbL — the substrate FHIR-AgentBench uses). No PHI, no
PhysioNet credentialing.

> **Boot path verified on macOS + Docker Desktop 28.4.0 / Compose v2.39.2 (2026-06-21)** — see
> [`SMOKE_TEST.md`](SMOKE_TEST.md). The full data-load + ablation **results** were produced on EC2,
> not this laptop path (see "Where the results came from" below).

## Run

```bash
# 1. Start Medplum (server + postgres + redis), MCP enabled.
#    First boot runs DB migrations — verified healthy in ~75s on a laptop.
docker compose up -d
until curl -s http://localhost:8103/healthcheck | grep -q '"ok":true'; do sleep 5; done
#    -> {"ok":true,"version":"5.1.21-...","postgres":true,"redis":true,"redisInstances":{"default":true}}

# 2. Get an admin token (bare-PKCE, no client_id; default super admin admin@example.com / medplum_admin)
python3 scripts/get_token.py    # -> a ~695-char JWT on stdout

# 3. Load the MIMIC demo (8 gold resource types, ~1h on 4 vCPU; idempotent — PUTs UUID ids)
bash scripts/load_mimic.sh
```

Then point the eval harness at `http://localhost:8103` (the repo `run_matrix.py` spawns
`treatment_mcp_server.py` against it per cell). **Prereq:** the loader (step 3) needs `wget`
(`brew install wget` on macOS).

> **Two distinct MCP URLs — don't conflate them.** `http://localhost:8103/mcp/stream` is *Medplum's own*
> shipped MCP surface (the smoke-test target; advertises the generic `fhir-request` tool). The eval
> agents instead talk to the **harness treatment server** at `http://127.0.0.1:8765/mcp`, which
> `run_matrix.py` spawns per cell with the selected `TOOL_SUBSET` and wires up via `MEDPLUM_MCP_URL`.
> When reproducing, do **not** set `MEDPLUM_MCP_URL` by hand — let `run_matrix.py` manage it, or you'll
> measure Medplum's generic surface instead of the arm under test.

## Verified on first boot (smoke test)

| Check | Result |
|---|---|
| `docker compose up -d` | postgres:16 + redis:7 + medplum/medplum-server:latest all start; pg+redis healthy |
| `/healthcheck` | `{"ok":true,...}` after ~75s (migrations on first boot) |
| `scripts/get_token.py` | 695-char JWT for the seeded super admin |
| `POST /fhir/R4/Patient` | `201` (server-assigned id) |
| `PUT /fhir/R4/Patient/<uuid>` | `200` — the loader's path (preserves `true_fhir_ids`) |
| `/mcp/stream` (no auth) | `401` — MCP is gated |
| `/mcp/stream` `tools/list` (auth) | `200`, advertises **`fhir-request`** (+ `search`/`fetch` stubs) = the byte-for-byte generic baseline arm |

## Notes / gotchas

- **Default super admin** is seeded on first boot: `admin@example.com` / `medplum_admin`.
- **MCP endpoint:** `http://localhost:8103/mcp/stream` (returns `401` unauthenticated; `200` +
  `tools/list` with a bearer token). Enabled via `MEDPLUM_MCP_ENABLED=true` in the compose file.
- **Client-assigned ids must be UUIDs.** Medplum rejects non-UUID ids on `PUT` with `400 "Invalid id"`.
  The MIMIC demo ids *are* UUIDs, so `scripts/bulk_load.py` (PUT, to preserve ids) is unaffected — but
  hand-crafted test resources need a UUID id, or use `POST` for a server-assigned one.
- **`$ai` operation:** enable per-project via super-admin (add the `ai` feature + an OpenAI key as a
  project secret) — see the harness `$ai` backend (`agent/ai_agent.py`). Not required for the MCP arms.
- Loader uses `PUT` so resource ids are preserved (the benchmark's `true_fhir_ids` must match).
- Reset everything: `docker compose down -v` (removes the postgres volume too).

## Where the results came from (honesty caveat)

The eval **results** in the repo [`REPORT.md`](../REPORT.md) were produced on ephemeral **AWS EC2**
boxes (`t3.xlarge`, us-east-2) — the ~1h MIMIC load and the multi-hour Opus / GPT-5.5 ablation runs are
memory- and time-heavy, and those boxes are torn down. This bundle is a faithful, **boot-smoke-verified
recipe** of that environment, but the **full load + end-to-end ablation has not been re-run on this
laptop Docker path**. Reproducing the numbers from scratch here (steps 1–3 + the harness) is the open
reproducibility item, not a completed claim.
