import json
from .multi_turn_resource_agent import MultiTurnResourceAgent
from tools import get_tool
from utils import safe_llm_call
from .projection import project


class MultiTurnProjectedResourceAgent(MultiTurnResourceAgent):
    """A0': identical to MultiTurnResourceAgent (same tools, same system prompt + few-shot), EXCEPT the
    retrieved FHIR resources are projected (stripped of narrative/meta/extensions + recency-capped) before
    they enter the model's context. No typed tools, no SQL, no multi-turn change.

    This isolates "keep the blob out of the 32k window" from the three Bonfire levers. The only behavioral
    difference vs A0 is the single line that stringifies tool output (was `str(tool_output)`)."""

    def __init__(self, model: str, max_iterations: int = 30, verbose: bool = False, base_url=None,
                 max_per_type: int = 200):
        super().__init__(model, max_iterations, verbose, base_url)
        self.max_per_type = max_per_type

    def run(self, question: str) -> dict:
        self.messages = self.system_msg.copy()
        self.messages.append({"role": "user", "content": question})
        retrieved_resources = {}
        final_answer = None
        iteration = 0
        while final_answer is None and iteration < self.max_iterations:
            response_message, error, usage_info = safe_llm_call(
                model=self.model,
                messages=self.messages,
                tools=self.tools,
                base_url=self.base_url,
            )
            self.messages.append(response_message)
            self._update_usage(usage_info)

            if error:
                return {
                    "retrieved_fhir_resources": retrieved_resources,
                    "final_answer": f"Error: {error}",
                    "trace": self.messages,
                    "usage": self.total_usage,
                }

            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    if tool_name in ("get_resources_by_patient_fhir_id", "get_resources_by_resource_id"):
                        tool_function = get_tool(tool_name)
                        tool_output = tool_function(**tool_args)
                        retrieved_resources.update(tool_output)  # keep the FULL output in the record (for grading parity)
                        projected = project(tool_output, self.max_per_type)  # <-- the ONLY behavioral change vs A0
                        if self.verbose:
                            print(f"[A0'] {tool_name}{tool_args}: {len(str(tool_output))}->{len(str(projected))} chars")
                        self.messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_name,
                            "content": str(projected),
                        })
                    else:
                        return {
                            "retrieved_fhir_resources": retrieved_resources,
                            "final_answer": f"Expected 'get_resources_by_patient_fhir_id' or 'get_resources_by_resource_id' tool call, but got '{tool_name}'",
                            "trace": self.messages,
                            "usage": self.total_usage,
                        }
            else:
                if self._is_final_answer(response_message.content):
                    final_answer = response_message.content
                    break
            iteration += 1

        if final_answer is None:
            final_answer = "No final answer reached within iteration limit."

        return {
            "retrieved_fhir_resources": retrieved_resources,
            "final_answer": final_answer,
            "trace": self.messages,
            "usage": self.total_usage,
        }
