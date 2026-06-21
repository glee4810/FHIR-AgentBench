# Reproducible FHIR-agent eval environment

A one-command-ish environment for the typed-FHIR-tool ablation eval: a self-hosted Medplum loaded
with the **MIMIC-IV-on-FHIR demo** (100 real de-identified ICU patients, open-access, ODbL — the
substrate FHIR-AgentBench uses). No PHI, no credentialing.

## Run

```bash
# 1. Start Medplum (server + postgres + redis), MCP enabled
docker compose up -d
# wait for health (~60s):
until curl -s http://localhost:8103/healthcheck | grep -q '"ok":true'; do sleep 5; done

# 2. Load the MIMIC demo (8 gold resource types, ~1h on 4 vCPU; idempotent)
bash scripts/load_mimic.sh

# 3. Get an admin token (bare-PKCE; default super admin admin@example.com / medplum_admin)
python3 scripts/get_token.py
```

Then point the eval harness (FHIR-AgentBench `agent/mcp_agent.py`) at `http://localhost:8103`.

## Notes
- Default super admin is seeded on first boot: `admin@example.com` / `medplum_admin`.
- MCP endpoint: `http://localhost:8103/mcp/stream` (returns 401 when enabled+gated).
- `$ai` operation: enable per-project via super-admin (add the `ai` feature + an OpenAI key as a
  project secret) — see the harness `$ai` backend setup.
- Loader uses `PUT` so resource ids are preserved (the benchmark's `true_fhir_ids` must match).
