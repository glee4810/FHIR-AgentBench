"""
MCP agent backend for FHIR-AgentBench.

Benchmarks an LLM agent that accesses FHIR through a Model Context Protocol (MCP)
server (Medplum) instead of direct GCP FHIR REST calls. The SAME agent benchmarks
the BASELINE (whatever tools the MCP currently advertises, e.g. the generic
`fhir-request`) and the TREATMENT (purpose-built tools) -- the only variable is the
tool surface the MCP server exposes. Output shape matches the other agents so the
existing evaluation_metrics.py scores it unchanged.
"""
import os
import json
import time
import base64
import hashlib
import secrets
import asyncio
import threading
import urllib.request
import urllib.parse
import urllib.error

from .base_agent import BaseAgent
from utils import safe_llm_call

from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

MEDPLUM_BASE = os.environ.get("MEDPLUM_BASE_URL", "http://localhost:8103").rstrip("/")
MCP_URL = os.environ.get("MEDPLUM_MCP_URL", MEDPLUM_BASE + "/mcp/stream")
# The two dummy stubs `search`/`fetch` exist only for ChatGPT's handshake and return a
# placeholder doc; exclude them so the baseline reflects the real functional surface.
EXCLUDE_TOOLS = {t for t in os.environ.get("MCP_EXCLUDE_TOOLS", "search,fetch").split(",") if t}
MAX_STEPS = int(os.environ.get("MCP_MAX_STEPS", "12"))
TOOL_CHAR_CAP = int(os.environ.get("MCP_TOOL_CHAR_CAP", "100000"))
# Input context gate (cap-factorial variable): safe_llm_call returns overflow if the prompt
# exceeds this. Default 32k = the benchmark's stock cap (the artifact under test). The runner
# raises it (e.g. 100k) for the raised-cap arm so we can separate reasoning-gain from cap-dodging.
MCP_INPUT_CAP = int(os.environ.get("MCP_INPUT_CAP", "32000"))


def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def get_medplum_token() -> str:
    """Bare-PKCE login (no client_id) against a self-hosted Medplum -> access token."""
    v = _b64u(secrets.token_bytes(32))
    ch = _b64u(hashlib.sha256(v.encode()).digest())

    def post(path, data, form=False):
        body = urllib.parse.urlencode(data).encode() if form else json.dumps(data).encode()
        ct = "application/x-www-form-urlencoded" if form else "application/json"
        last = None
        for attempt in range(7):  # auth endpoint rate-limits when many shards log in at once
            try:
                req = urllib.request.Request(MEDPLUM_BASE + path, data=body, headers={"Content-Type": ct})
                return json.load(urllib.request.urlopen(req, timeout=30))
            except urllib.error.HTTPError as e:
                last = e
                if e.code in (429, 500, 502, 503):
                    time.sleep(1.0 + attempt * 1.5 + secrets.randbelow(1500) / 1000.0)
                    continue
                raise
            except Exception as e:
                last = e
                time.sleep(1.0 + attempt)
        raise last

    email = os.environ.get("MEDPLUM_EMAIL", "admin@example.com")
    pw = os.environ.get("MEDPLUM_PASSWORD", "medplum_admin")
    login = post("/auth/login", {"email": email, "password": pw, "scope": "openid",
                                 "codeChallenge": ch, "codeChallengeMethod": "S256"})
    tok = post("/oauth2/token", {"grant_type": "authorization_code", "code": login["code"],
                                 "code_verifier": v}, form=True)
    return tok["access_token"]


# One Medplum token per process, reused across all agents/questions (the harness builds a fresh
# agent per question; without caching that's one login per question -> auth rate-limit 429s).
_TOKEN = {"v": None}
_TLOCK = threading.Lock()


def _cached_token(force=False):
    with _TLOCK:
        if force or not _TOKEN["v"]:
            _TOKEN["v"] = get_medplum_token()
    return _TOKEN["v"]


def _tc_fields(tc):
    """litellm returns tool_calls as objects on the first turn but as dicts on multi-turn
    continuations (esp. the Anthropic path); normalize to (id, name, arguments_str)."""
    if isinstance(tc, dict):
        fn = tc.get("function") or {}
        return tc.get("id"), fn.get("name"), fn.get("arguments")
    fn = tc.function
    return tc.id, fn.name, fn.arguments


def _assistant_msg(resp):
    """Normalize an assistant turn to a plain dict so litellm can re-serialize it on the next
    call. Re-feeding the raw litellm Message (whose tool_calls may be dicts) makes litellm's own
    conversion do tc.function on a dict -> 'dict has no attribute function'."""
    tcs = getattr(resp, "tool_calls", None) or []
    norm = []
    for tc in tcs:
        tid, name, arguments = _tc_fields(tc)
        norm.append({"id": tid, "type": "function",
                     "function": {"name": name, "arguments": arguments or "{}"}})
    msg = {"role": "assistant", "content": getattr(resp, "content", None) or ""}
    if norm:
        msg["tool_calls"] = norm
    return msg


def _mcp(url, token, *, list_tools=False, name=None, args=None):
    """Synchronous one-shot MCP call (Medplum runs stateless streamable-HTTP)."""
    async def go():
        async with streamablehttp_client(url, headers={"Authorization": "Bearer " + token}) as (r, w, _):
            async with ClientSession(r, w) as s:
                await s.initialize()
                if list_tools:
                    return (await s.list_tools()).tools
                res = await s.call_tool(name, args or {})
                return "".join(getattr(c, "text", "") for c in res.content)
    return asyncio.run(asyncio.wait_for(go(), timeout=float(os.environ.get("MCP_CALL_TIMEOUT", "90"))))


class MCPAgent(BaseAgent):
    """Multi-turn agent that retrieves FHIR via an MCP server and reasons in natural language."""

    def __init__(self, model: str, verbose: bool = False, base_url=None):
        super().__init__(model, verbose, base_url)
        self.token = _cached_token()
        tool_meta = _mcp(MCP_URL, self.token, list_tools=True)
        self.tools = []
        for t in tool_meta:
            if t.name in EXCLUDE_TOOLS:
                continue
            self.tools.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": (t.description or "")[:1024],
                    "parameters": t.inputSchema or {"type": "object", "properties": {}},
                },
            })
        names = ", ".join(tt["function"]["name"] for tt in self.tools)
        self.system_msg = [{"role": "system", "content": f"""You are a clinical data assistant. You answer questions about a single patient stored on a FHIR R4 server, which you access ONLY through these tools: {names}.

Workflow: use the tools to retrieve the relevant FHIR resources for the patient, then reason over the retrieved data to answer. Construct precise FHIR searches (filter by patient, code, date, category) to avoid pulling unnecessary data. Follow pagination if a result is truncated.

Answer rules: report values in the same format/units as the underlying data; if there are multiple answers return them all as a list; if the data is not present, clearly state you cannot find it. Never guess or fabricate values — only state what the tools return. Be efficient: do not repeat identical calls."""}]

    def _accumulate(self, retrieved, text):
        try:
            data = json.loads(text)
        except Exception:
            return
        items = []
        if isinstance(data, dict):
            if data.get("resourceType") == "Bundle":
                for e in data.get("entry", []):
                    r = e.get("resource")
                    if isinstance(r, dict) and r.get("resourceType"):
                        items.append(r)
            elif data.get("resourceType"):
                items.append(data)
        for r in items:
            retrieved.setdefault(r["resourceType"], []).append(r)

    @staticmethod
    def _safe_trace(msgs):
        # Multiprocessing must pickle the return value; litellm Message objects (esp. the
        # Anthropic path) hold non-picklable stream refs, so flatten to plain dicts.
        out = []
        for m in msgs:
            if isinstance(m, dict):
                out.append({"role": m.get("role"), "name": m.get("name"),
                            "content": str(m.get("content", ""))[:4000]})
            else:
                entry = {"role": getattr(m, "role", "assistant"),
                         "content": str(getattr(m, "content", "") or "")[:4000]}
                tcs = getattr(m, "tool_calls", None)
                if tcs:
                    entry["tool_calls"] = [
                        {"name": tc.function.name, "arguments": (tc.function.arguments or "")[:800]}
                        for tc in tcs
                    ]
                out.append(entry)
        return out

    def _out(self, retrieved, answer, msgs):
        return {
            "retrieved_fhir_resources": retrieved,
            "final_answer": answer,
            "trace": self._safe_trace(msgs),
            "usage": self.total_usage,
        }

    def _mcp_call(self, name, args):
        """Execute one tool against the (treatment) MCP server. Shared by MCPAgent and AIAgent."""
        return _mcp(MCP_URL, self.token, name=name, args=args)

    def run(self, question: str) -> dict:
        msgs = self.system_msg + [{"role": "user", "content": question}]
        retrieved = {}
        for _ in range(MAX_STEPS):
            cap = int(os.environ.get("MCP_INPUT_CAP", str(MCP_INPUT_CAP)))  # per-cell (cap-factorial)
            resp, err, usage = safe_llm_call(self.model, msgs, tools=self.tools, max_tokens=cap)
            self._update_usage(usage)
            if err:
                return self._out(retrieved, f"Error: {err}", msgs)
            tool_calls = getattr(resp, "tool_calls", None)
            msgs.append(_assistant_msg(resp))
            if not tool_calls:
                return self._out(retrieved, getattr(resp, "content", "") or "", msgs)
            for tc in tool_calls:
                tc_id, tc_name, tc_args_raw = _tc_fields(tc)
                try:
                    args = json.loads(tc_args_raw or "{}")
                except Exception:
                    args = {}
                if self.verbose:
                    print(f"[mcp] {tc_name} {args}")
                try:
                    out = _mcp(MCP_URL, self.token, name=tc_name, args=args)
                except Exception as e:
                    out = json.dumps({"error": str(e)})
                self._accumulate(retrieved, out)
                msgs.append({
                    "tool_call_id": tc_id,
                    "role": "tool",
                    "name": tc_name,
                    "content": out[:TOOL_CHAR_CAP],
                })
        return self._out(retrieved, "Max steps reached without a final answer.", msgs)
