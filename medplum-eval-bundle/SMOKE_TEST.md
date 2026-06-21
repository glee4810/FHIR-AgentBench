# Docker smoke test — verified boot path

**Verified 2026-06-21 on macOS (Apple Silicon), Docker Desktop 28.4.0 / Compose v2.39.2.**
This validates that the bundle *boots and is wired correctly* on a laptop — NOT that the full
eval was re-run locally (see the honesty caveat in the repo README §"Where this was actually run").

| Check | Result |
|---|---|
| `docker compose up -d` (postgres:16 + redis:7 + medplum/medplum-server:latest) | all 3 containers created, postgres+redis healthy |
| Medplum `/healthcheck` | `{"ok":true,"version":"5.1.21-...","postgres":true,"redis":true}` after ~75s (first boot runs migrations) |
| Bare-PKCE token (`scripts/get_token.py`, default super admin `admin@example.com`/`medplum_admin`) | 695-char JWT issued |
| FHIR create — `POST /fhir/R4/Patient` | `201` |
| FHIR upsert — `PUT /fhir/R4/Patient/<uuid>` (the loader's path; ids must be UUIDs) | `200`, read-back `200` |
| MCP `/mcp/stream` unauthenticated | `401` (gated, as designed) |
| MCP `/mcp/stream` authenticated `tools/list` | `200`, advertises **`fhir-request`** (+ `search`/`fetch` stubs) — i.e. the byte-for-byte generic baseline arm |

**Gotcha surfaced by the smoke test:** Medplum rejects non-UUID client-assigned ids with
`400 "Invalid id"` on `PUT`. The MIMIC-IV-on-FHIR demo ids *are* UUIDs, so `scripts/bulk_load.py`
(which PUTs to preserve `true_fhir_ids`) is unaffected — but if you hand-craft test resources, use a
UUID id or `POST` for a server-assigned one.
