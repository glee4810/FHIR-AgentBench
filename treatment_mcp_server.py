"""Catalog-driven FHIR tool server for the ablation eval.

ONE server, many arms. The ablation's only variable is the TOOL SUBSET this server advertises,
selected by the TOOL_SUBSET env var (a preset name or a comma-list of tool names). That keeps the
MCP plumbing identical across arms so the measured lift is attributable to the tools, not the
transport.

Arms (Arch-3):
  - control          : `fhir_request` — BYTE-FOR-BYTE Medplum's shipped generic tool
                       (examples/.../spaces-bots/fhir-translator-bot.ts). The honest baseline.
  - c0               : `fhir_request_frugal` — the SAME generic mechanism but a "best generic"
                       prompt (coaches _elements / _count / patient-filter / pagination). Separates
                       "typed tools help" from "you could've just prompted the generic better".
  - validated5       : the 5 purpose-built tools the POC validated.
  - + resolve_references (locked: fixes the MedicationRequest->Medication reference failure).
  - <research tools> : typed-per-resource searches / terminology / introspection drop into CATALOG.

Run:  MEDPLUM_BASE_URL=http://localhost:8103 TOOL_SUBSET=validated5 python treatment_mcp_server.py
Serves streamable-HTTP MCP at http://127.0.0.1:8765/mcp
"""
import os
import json
import time
import base64
import hashlib
import secrets
import urllib.request
import urllib.parse
import urllib.error

from mcp.server.fastmcp import FastMCP

MEDPLUM_BASE = os.environ.get("MEDPLUM_BASE_URL", "http://localhost:8103").rstrip("/")
FHIR = MEDPLUM_BASE + "/fhir/R4"

mcp = FastMCP("medplum-plus", host="127.0.0.1", port=int(os.environ.get("TREATMENT_PORT", "8765")))

_token = {"v": None}


def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _login():
    v = _b64u(secrets.token_bytes(32))
    ch = _b64u(hashlib.sha256(v.encode()).digest())

    def post(path, data, form=False):
        body = urllib.parse.urlencode(data).encode() if form else json.dumps(data).encode()
        ct = "application/x-www-form-urlencoded" if form else "application/json"
        last = None
        for attempt in range(7):  # auth endpoint 429s when arms restart back-to-back across the matrix
            try:
                req = urllib.request.Request(MEDPLUM_BASE + path, data=body, headers={"Content-Type": ct})
                return json.load(urllib.request.urlopen(req, timeout=30))
            except urllib.error.HTTPError as e:
                last = e
                if e.code in (429, 500, 502, 503):
                    time.sleep(1.0 + attempt * 1.5)
                    continue
                raise
            except Exception as e:
                last = e
                time.sleep(1.0 + attempt)
        raise last

    lg = post("/auth/login", {"email": os.environ.get("MEDPLUM_EMAIL", "admin@example.com"),
                              "password": os.environ.get("MEDPLUM_PASSWORD", "medplum_admin"),
                              "scope": "openid", "codeChallenge": ch, "codeChallengeMethod": "S256"})
    _token["v"] = post("/oauth2/token", {"grant_type": "authorization_code", "code": lg["code"],
                                         "code_verifier": v}, form=True)["access_token"]


def _fhir_get(path: str) -> dict:
    if _token["v"] is None:
        _login()
    url = FHIR + "/" + path.lstrip("/")
    for attempt in range(2):
        try:
            req = urllib.request.Request(url, headers={"Authorization": "Bearer " + _token["v"],
                                                       "Accept": "application/fhir+json"})
            return json.load(urllib.request.urlopen(req, timeout=60))
        except urllib.error.HTTPError as e:
            if e.code == 401 and attempt == 0:
                _login()
                continue
            return {"error": f"HTTP {e.code}", "detail": e.read().decode()[:300]}
        except Exception as e:
            return {"error": str(e)}
    return {"error": "unreachable"}


def _strip_prefix(p: str) -> str:
    p = (p or "").strip().lstrip("/")
    if p.lower().startswith("fhir/r4/"):
        p = p[len("fhir/r4/"):]
    return p


# ----------------------------------------------------------------------------------------------
# Tool functions (plain; registered selectively below). Each is READ-ONLY and FHIR-native.
# ----------------------------------------------------------------------------------------------

# Medplum's EXACT shipped description (fhir-translator-bot.ts:48-49) — the control must match it.
_MEDPLUM_FHIR_REQUEST_DESC = (
    "REQUIRED for all FHIR operations. Make a FHIR request to the Medplum server. You MUST use "
    "this tool - you cannot execute FHIR requests yourself. For updates: first GET the resource, "
    "then PUT with the modified full resource."
)

# control_include = the SAME generic tool + _include COACHING (so control->control_include isolates
# pure description coaching; control_include->resolve_references then isolates tool STRUCTURE).
_MEDPLUM_FHIR_REQUEST_INCLUDE_DESC = _MEDPLUM_FHIR_REQUEST_DESC + (
    " To resolve a reference in ONE request, add the FHIR `_include` parameter to a search: e.g. "
    "'MedicationRequest?patient=123&_include=MedicationRequest:medication' returns each "
    "MedicationRequest together with its referenced Medication in the same Bundle. Use `_include` "
    "whenever a question needs a field from a referenced resource (medication, encounter, requester)."
)


def fhir_request(method: str = "GET", path: str = "", body: dict = None, visualize: bool = False) -> str:
    p = _strip_prefix(path)
    if (method or "GET").upper() != "GET":
        return json.dumps({"error": "read-only eval harness: only GET is supported"})
    return json.dumps(_fhir_get(p))


def fhir_request_frugal(method: str = "GET", path: str = "", body: dict = None) -> str:
    """Make a FHIR GET request. BE FRUGAL with context: ALWAYS scope to the patient
    (e.g. '?patient=<id>'); use '&_elements=' to return only the fields you need; set '&_count=' to
    bound page size; follow the Bundle 'next' link for more results instead of widening the query.
    Avoid pulling whole charts. path e.g. 'Observation?patient=123&category=laboratory&_elements=code,valueQuantity,effectiveDateTime&_count=50'."""
    p = _strip_prefix(path)
    if (method or "GET").upper() != "GET":
        return json.dumps({"error": "read-only eval harness: only GET is supported"})
    return json.dumps(_fhir_get(p))


def fhir_request_include(method: str = "GET", path: str = "", body: dict = None) -> str:
    """Make a FHIR GET request. To resolve REFERENCES in one call, add the FHIR `_include` parameter:
    e.g. 'MedicationRequest?patient=123&_include=MedicationRequest:medication' returns each
    MedicationRequest AND its referenced Medication (the drug name/code) in the same Bundle. Use
    `_include` whenever a question needs data from a referenced resource instead of making extra calls
    — common: 'MedicationRequest:medication', 'Observation:encounter', 'Condition:encounter',
    'Procedure:encounter'. path e.g. 'MedicationRequest?patient=123&_include=MedicationRequest:medication&_count=100'."""
    p = _strip_prefix(path)
    if (method or "GET").upper() != "GET":
        return json.dumps({"error": "read-only eval harness: only GET is supported"})
    return json.dumps(_fhir_get(p))


def get_patient_chart(patient_id: str, resource_types: str = "", max_items: int = 200) -> str:
    """Retrieve a patient's chart in ONE call via FHIR $everything. Returns a FHIR Bundle of the
    patient's resources so you can reason over the chart directly instead of guessing search params.
    Optionally narrow with resource_types (comma-separated, e.g.
    'Observation,Condition,MedicationRequest') to control size. max_items caps the page size."""
    q = f"_count={max_items}"
    if resource_types.strip():
        types = ",".join(t.strip() for t in resource_types.split(",") if t.strip())
        q += f"&_type={urllib.parse.quote(types)}"
    return json.dumps(_fhir_get(f"Patient/{patient_id}/$everything?{q}"))


def search_observations(patient_id: str, category: str = "", code_text: str = "",
                        date_from: str = "", date_to: str = "", max_items: int = 200) -> str:
    """Purpose-built Observation search for one patient (labs AND vitals/chartevents).
    category: optional FHIR category code, e.g. 'laboratory' or 'vital-signs'.
    code_text: optional free-text of the measurement name (e.g. 'Arterial Blood Pressure mean',
    'Hemoglobin') matched against the observation's code text.
    date_from / date_to: optional inclusive ISO dates (YYYY-MM-DD).
    Returns matching Observations as a FHIR Bundle. Prefer this over raw search for clinical values."""
    parts = [f"patient={patient_id}", f"_count={max_items}", "_sort=-date"]
    if category.strip():
        parts.append(f"category={urllib.parse.quote(category.strip())}")
    if code_text.strip():
        parts.append(f"code:text={urllib.parse.quote(code_text.strip())}")
    if date_from.strip():
        parts.append(f"date=ge{date_from.strip()}")
    if date_to.strip():
        parts.append(f"date=le{date_to.strip()}")
    return json.dumps(_fhir_get("Observation?" + "&".join(parts)))


def search_fhir(resource_type: str, patient_id: str = "", query: str = "", max_items: int = 200) -> str:
    """Search any FHIR resource type for a patient. resource_type e.g. 'MedicationRequest',
    'Condition', 'Procedure', 'Encounter'. patient_id auto-adds the patient filter. query is extra
    FHIR search params (e.g. 'status=active&date=ge2136-01-01'). Returns a FHIR Bundle."""
    parts = [f"_count={max_items}"]
    if patient_id.strip():
        parts.append(f"patient={patient_id.strip()}")
    if query.strip():
        parts.append(query.strip().lstrip("?&"))
    return json.dumps(_fhir_get(f"{resource_type}?" + "&".join(parts)))


def read_resource(resource_type: str, id: str) -> str:
    """Read a single FHIR resource by type and id. Returns the resource JSON."""
    return json.dumps(_fhir_get(f"{resource_type}/{id}"))


def list_search_params(resource_type: str) -> str:
    """Introspection: list the valid FHIR search parameters for a resource type (from the server's
    CapabilityStatement). Use this if unsure how to filter a resource."""
    cap = _fhir_get("metadata")
    if "error" in cap:
        return json.dumps(cap)
    out = []
    for rest in cap.get("rest", []):
        for res in rest.get("resource", []):
            if res.get("type") == resource_type:
                out = [sp.get("name") for sp in res.get("searchParam", [])]
    return json.dumps({"resourceType": resource_type, "searchParams": out})


def resolve_references(resource_type: str, patient_id: str = "", include: str = "",
                       query: str = "", max_items: int = 100) -> str:
    """Search a resource type for a patient with referenced resources included in the same result.
    resource_type e.g. 'MedicationRequest'; include is the reference to follow in FHIR _include
    syntax (e.g. 'MedicationRequest:medication'); query adds extra FHIR search params. Returns a
    FHIR Bundle of the matched resources together with their included referenced resources."""
    # NOTE (de-coached on purpose): the description states the tool's FUNCTION, not a persuasive
    # worked example. control_include carries the equivalent _include *coaching* as prose, so the
    # arm_ref-vs-control_include delta isolates tool STRUCTURE from description coaching.
    parts = [f"_count={max_items}"]
    if patient_id.strip():
        parts.append(f"patient={patient_id.strip()}")
    if include.strip():
        parts.append(f"_include={urllib.parse.quote(include.strip())}")
    if query.strip():
        parts.append(query.strip().lstrip("?&"))
    return json.dumps(_fhir_get(f"{resource_type}?" + "&".join(parts)))


def search_encounters(patient_id: str, status: str = "", date_from: str = "",
                      date_to: str = "", max_items: int = 100) -> str:
    """Purpose-built Encounter search for one patient (admissions, ED visits, ICU stays).
    status: optional FHIR status (e.g. 'finished', 'in-progress'). date_from / date_to: optional
    inclusive ISO dates (YYYY-MM-DD) filtering the encounter period. Returns Encounters as a FHIR
    Bundle sorted most-recent-first. Use for 'when was the patient admitted/discharged', visit counts,
    length-of-stay, and to get the encounter an observation/procedure belongs to."""
    parts = [f"patient={patient_id}", f"_count={max_items}", "_sort=-date"]
    if status.strip():
        parts.append(f"status={urllib.parse.quote(status.strip())}")
    if date_from.strip():
        parts.append(f"date=ge{date_from.strip()}")
    if date_to.strip():
        parts.append(f"date=le{date_to.strip()}")
    return json.dumps(_fhir_get("Encounter?" + "&".join(parts)))


def search_procedures(patient_id: str, date_from: str = "", date_to: str = "",
                      encounter: str = "", max_items: int = 100) -> str:
    """Purpose-built Procedure search for one patient. date_from / date_to: optional inclusive ISO
    dates (YYYY-MM-DD). encounter: optional Encounter id to scope procedures to one visit. Returns
    Procedures as a FHIR Bundle sorted most-recent-first."""
    parts = [f"patient={patient_id}", f"_count={max_items}", "_sort=-date"]
    if date_from.strip():
        parts.append(f"date=ge{date_from.strip()}")
    if date_to.strip():
        parts.append(f"date=le{date_to.strip()}")
    if encounter.strip():
        parts.append(f"encounter={urllib.parse.quote(encounter.strip())}")
    return json.dumps(_fhir_get("Procedure?" + "&".join(parts)))


# ----------------------------------------------------------------------------------------------
# Catalog + subset registration
# ----------------------------------------------------------------------------------------------
CATALOG = {
    "fhir_request": (fhir_request, _MEDPLUM_FHIR_REQUEST_DESC),
    "fhir_request_frugal": (fhir_request_frugal, None),
    "fhir_request_include": (fhir_request_include, _MEDPLUM_FHIR_REQUEST_INCLUDE_DESC),
    "get_patient_chart": (get_patient_chart, None),
    "search_observations": (search_observations, None),
    "search_fhir": (search_fhir, None),
    "read_resource": (read_resource, None),
    "list_search_params": (list_search_params, None),
    "resolve_references": (resolve_references, None),
    # research-locked typed additions (the only slices with real benchmark demand)
    "search_encounters": (search_encounters, None),
    "search_procedures": (search_procedures, None),
}

# Ablation arms (research-locked, 2026-06-21): demand is CONCENTRATED, so 4 cumulative arms + 1
# orthogonal efficiency arm — NOT a padded 12-point curve. resolve_references is the one CORE add
# (closes the 17% MedicationRequest->Medication slice; novel — no surveyed FHIR MCP server ships it).
_V5 = ["get_patient_chart", "search_observations", "search_fhir", "read_resource", "list_search_params"]
PRESETS = {
    "control": ["fhir_request"],                       # Medplum's exact shipped generic (floor)
    "control_include": ["fhir_request_include"],       # generic + _include coaching (attribution arm)
    "c0": ["fhir_request_frugal"],                     # orthogonal efficiency arm (vs control, head-to-head)
    "cat2": _V5[:2],                                   # 2 tools (curve point) — nested subset of validated5
    "cat4": _V5[:4],                                   # 4 tools (curve point) — nested subset of validated5
    "validated5": _V5,                                 # current baseline
    "arm_ref": _V5 + ["resolve_references"],           # +CORE add (the steep segment)
    "arm_enc": _V5 + ["resolve_references", "search_encounters"],            # +TAIL probe (13.4%)
    "arm_full8": _V5 + ["resolve_references", "search_encounters", "search_procedures"],  # designed-flat (3.7%)
    "validated6": _V5 + ["resolve_references"],        # alias for arm_ref
    "full": list(CATALOG.keys()),
}


def _selected_tool_names():
    raw = os.environ.get("TOOL_SUBSET", "full").strip()
    if raw in PRESETS:
        return PRESETS[raw]
    names = [n.strip() for n in raw.split(",") if n.strip()]
    unknown = [n for n in names if n not in CATALOG]
    if unknown:
        raise SystemExit(f"unknown tools in TOOL_SUBSET: {unknown}; known: {list(CATALOG)}")
    return names


def _register(names):
    for n in names:
        fn, desc = CATALOG[n]
        if desc:
            mcp.add_tool(fn, name=n, description=desc)
        else:
            mcp.add_tool(fn, name=n)


if __name__ == "__main__":
    sel = _selected_tool_names()
    _register(sel)
    print(f"[treatment-server] TOOL_SUBSET -> {sel}", flush=True)
    _login()
    mcp.run(transport="streamable-http")
