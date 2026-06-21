"""$ai-surface agent backend for FHIR-AgentBench.

Same agent loop and SAME tool surface as MCPAgent — the ONLY difference is the LLM transport:
instead of calling the model directly (litellm), the completion is routed through Medplum's in-FHIR
`$ai` operation (`POST /fhir/R4/$ai`), which proxies to the OpenAI-compatible API using the
project's server-side key. This measures whether the tool-surface lift holds on Medplum's OWN
agentic surface ($ai/Spaces), not just an external MCP client.

Contract (verified against medplum/.../operations/ai.ts):
  - Request: FHIR Parameters {messages: JSON, model: str, tools: JSON, temperature: decimal}.
  - MUST be non-streaming (no `Accept: text/event-stream`) — tools are DROPPED in streaming mode.
  - Response: FHIR Parameters {content: str, tool_calls: JSON} where each tool_call's
    function.arguments is already a PARSED OBJECT (ai.ts buildParametersResponse).
  - `$ai` returns NO token usage -> we ESTIMATE tokens (input via count_tokens_in_messages,
    output via the tokenizer) so the budget ledger still binds on this arm.

Requires the Medplum project to have the 'ai' feature + an OPENAI_API_KEY project secret.
"""
import os
import json
import urllib.request
import urllib.error

from .mcp_agent import MCPAgent, MCP_URL, TOOL_CHAR_CAP, MEDPLUM_BASE, _cached_token
from utils.core_utils import count_tokens_in_messages, is_reasoning_llm

try:
    from utils.core_utils import count_token_encoding as _ENC
except Exception:
    _ENC = None

AI_ENDPOINT = MEDPLUM_BASE + "/fhir/R4/$ai"
MAX_STEPS = int(os.environ.get("MCP_MAX_STEPS", "12"))
AI_TIMEOUT = float(os.environ.get("AI_CALL_TIMEOUT", "150"))
# $ai passes `model` straight to the OpenAI-compatible backend; default to the frozen GPT id.
AI_MODEL = os.environ.get("AI_MODEL", os.environ.get("EVAL_GPT_MODEL", "gpt-5"))
AI_INPUT_CAP = int(os.environ.get("MCP_INPUT_CAP", os.environ.get("AI_INPUT_CAP", "100000")))


def _count_out(text):
    if not text:
        return 0
    if _ENC is not None:
        try:
            return len(_ENC.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


class AIAgent(MCPAgent):
    """Multi-turn agent whose completions go through Medplum's `$ai` operation (GPT, server-side)."""

    def __init__(self, model: str, verbose: bool = False, base_url=None):
        super().__init__(model, verbose, base_url)
        # the id Medplum forwards to OpenAI; read per-cell from env (NOT import-time).
        self.ai_model = os.environ.get("AI_MODEL") or model

    def _ai_call(self, messages, _auth_retry=0):
        """One non-streaming `$ai` completion. Returns (content, tool_calls, err)."""
        param = [
            {"name": "messages", "valueString": json.dumps(messages)},
            {"name": "model", "valueString": self.ai_model},
        ]
        # gpt-5 / o-series 400 on a non-default temperature -> omit it for reasoning models.
        if not is_reasoning_llm(self.ai_model):
            param.append({"name": "temperature", "valueDecimal": 0.0})
        if self.tools:
            param.append({"name": "tools", "valueString": json.dumps(self.tools)})
        body = json.dumps({"resourceType": "Parameters", "parameter": param}).encode()
        req = urllib.request.Request(
            AI_ENDPOINT, data=body,
            headers={"Authorization": "Bearer " + self.token,
                     "Content-Type": "application/fhir+json",
                     "Accept": "application/fhir+json"})  # NOT event-stream -> tools enabled
        try:
            resp = json.load(urllib.request.urlopen(req, timeout=AI_TIMEOUT))
        except urllib.error.HTTPError as e:
            if e.code == 401 and _auth_retry == 0:  # bounded: one token refresh, no recursion storm
                self.token = _cached_token(force=True)
                return self._ai_call(messages, _auth_retry + 1)
            return None, [], f"$ai HTTP {e.code}: {e.read().decode()[:300]}"
        except Exception as e:
            return None, [], f"$ai error: {e}"
        content, tool_calls = None, []
        for p in resp.get("parameter", []):
            if p.get("name") == "content":
                content = p.get("valueString")
            elif p.get("name") == "tool_calls":
                try:
                    tool_calls = json.loads(p.get("valueString", "[]"))
                except Exception:
                    tool_calls = []
        return content, tool_calls, None

    @staticmethod
    def _asst_msg(content, tool_calls):
        """OpenAI-shaped assistant turn for re-feeding. $ai returns parsed-object arguments;
        OpenAI's messages array expects function.arguments as a STRING -> re-stringify."""
        msg = {"role": "assistant", "content": content or ""}
        if tool_calls:
            norm = []
            for tc in tool_calls:
                fn = tc.get("function", {})
                args = fn.get("arguments")
                if not isinstance(args, str):
                    args = json.dumps(args or {})
                norm.append({"id": tc.get("id"), "type": "function",
                             "function": {"name": fn.get("name"), "arguments": args}})
            msg["tool_calls"] = norm
        return msg

    def run(self, question: str) -> dict:
        msgs = self.system_msg + [{"role": "user", "content": question}]
        retrieved = {}
        # tools schema is sent to OpenAI EVERY turn but lives outside message `content`; count it so
        # the $ai estimate matches what OpenAI actually bills (and the cap gate fires correctly).
        tool_toks = len(_ENC.encode(json.dumps(self.tools))) if (self.tools and _ENC is not None) else 0
        for _ in range(MAX_STEPS):
            cap = int(os.environ.get("MCP_INPUT_CAP", os.environ.get("AI_INPUT_CAP", str(AI_INPUT_CAP))))
            # estimate over the FULL serialized messages (content + tool_calls + tool outputs) + tools
            if _ENC is not None:
                in_toks = len(_ENC.encode(json.dumps(msgs))) + tool_toks
            else:
                in_toks = count_tokens_in_messages(msgs) + tool_toks
            if in_toks > cap:
                # no API call happens on overflow -> charge nothing (symmetric with the MCP arm)
                return self._out(retrieved, f"Error: Input tokens exceeded: {in_toks} > {cap}", msgs)
            content, tool_calls, err = self._ai_call(msgs)
            self._update_usage({"prompt_tokens": in_toks, "completion_tokens": _count_out(content)})
            if err:
                return self._out(retrieved, f"Error: {err}", msgs)
            msgs.append(self._asst_msg(content, tool_calls))
            if not tool_calls:
                return self._out(retrieved, content or "", msgs)
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name")
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args or "{}")
                    except Exception:
                        args = {}
                if self.verbose:
                    print(f"[$ai] {name} {args}")
                try:
                    out = self._mcp_call(name, args)
                except Exception as e:
                    out = json.dumps({"error": str(e)})
                self._accumulate(retrieved, out)
                msgs.append({"tool_call_id": tc.get("id"), "role": "tool",
                             "name": name, "content": out[:TOOL_CHAR_CAP]})
        return self._out(retrieved, "Max steps reached without a final answer.", msgs)
