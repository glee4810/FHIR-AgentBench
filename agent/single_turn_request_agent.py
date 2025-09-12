import json
from .base_agent import BaseAgent
from tools import get_tool_definitions, get_tool
from utils import safe_llm_call
from tools.resource_tools import supported_types


class SingleTurnRequestAgent(BaseAgent):
    """Single-step agent that uses a FHIR request tool to retrieve patient data and perform reasoning using natural language."""

    def __init__(self, model: str, verbose: bool = False, base_url=None):
        super().__init__(model, verbose, base_url)

        all_tools = get_tool_definitions()
        self.tools = [tool for tool in all_tools if tool["function"]["name"] in 
                     ["fhir_request_get"]]
        
        self.system_msg = [{"role": "system", "content": f"""You are a helpful assistant that answers questions about patient data.

Use fhir_request_get to retrieve patient data and answer questions based on the retrieved data.
Available resource types: {', '.join(supported_types)}. You can only call on these FHIR resources types for retrieval.

First, you should retrieve the patient data using fhir_request_get. Then, you should reason about the retrieved data and provide the answer.
When you provide answers, make sure to provide them in the same format as they are in the retrieved data. If multiple answers are provided, provide them all in a list.
If you cannot find the answer or relevant patient data, clearly state that you cannot find the information.
Do not guess attributes; instead, use the provided tool to retrieve the data.
Do not get stuck or repeat the same action.
Do not plan ahead. Instead, directly use the tool to retrieve and reason about the data.

Few-shot examples:

EXAMPLE 1:
Question: Calculate the number of ICU visits for patient 10009628 last year.
Context: The patient's FHIR ID is 51d2190c-cc46-56c5-b2ea-363895cbea75. Assume current time is 2153-12-31 23:59:00.
Tool Call: fhir_request_get with endpoint='Encounter?patient=51d2190c-cc46-56c5-b2ea-363895cbea75'
Analysis: Filter for ICU encounters (identifier.system containing 'encounter-icu') and count those from 2152.
Result: 0 ICU visits during last calendar year (01/01/2152–12/31/2152).

EXAMPLE 2:
Question: Count procedure occurrences during first hospital encounter.
Tool Calls: 1) Get Encounters, 2) Get Procedures
Analysis: Identify first hospital encounter, find linked procedures, count matches.
Result: Found 1 occurrence of the target procedure.

Follow these examples to structure your approach: make appropriate FHIR requests, then analyze systematically."""}]


    def run(self, question: str) -> dict:
        
        self.messages = self.system_msg.copy()
        self.messages.append({"role": "user", "content": question})
        retrieved_resources = {}
        
        # 1. Retrieve patient data
        response_message, error, usage_info = safe_llm_call(
            model=self.model,
            messages=self.messages,
            tools=self.tools,
            base_url=self.base_url
        )
        
        self.messages.append(response_message)
        self._update_usage(usage_info)
        
        if error:
            return {
                "retrieved_fhir_resources": retrieved_resources,
                "final_answer": f"Error: {error}",
                "trace": self.messages,
                "usage": self.total_usage
            }
        

        if response_message.tool_calls:

            for tool_call in response_message.tool_calls:

                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                if tool_name == "fhir_request_get":
                
                    if self.verbose:
                        print(f"Agent detected tool call: '{tool_name}' with arguments: {tool_args}")
                    
                    tool_function = get_tool(tool_name)
                    tool_output = tool_function(**tool_args)
                    retrieved_resources.update(tool_output)

                    if self.verbose:
                        print(f"Tool execution successful. Output snippet: {str(tool_output)[:500]}...")
                
                    self.messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_name,
                            "content": str(tool_output)
                        }
                    )

                else:
                    return {
                        "retrieved_fhir_resources": retrieved_resources,
                        "final_answer": f"Expected 'fhir_request_get' tool call, but got '{tool_name}'",
                        "trace": self.messages,
                        "usage": self.total_usage
                    }                    

        else:
            return {
                "retrieved_fhir_resources": retrieved_resources,
                "final_answer": "No tool calls found in the response",
                "trace": self.messages,
                "usage": self.total_usage
            }

        # 2. Reason about the retrieved data
        final_message, final_error, final_usage_info = safe_llm_call(
            model=self.model,
            messages=self.messages,
            base_url=self.base_url
        )
            
        self.messages.append(final_message)
        self._update_usage(final_usage_info)
        
        if final_error:
            return {
                "retrieved_fhir_resources": retrieved_resources,
                "final_answer": final_error,
                "trace": self.messages,
                "usage": self.total_usage
            }

        final_answer = final_message.content
        agent_output = {
            "retrieved_fhir_resources": retrieved_resources,
            "final_answer": final_answer,
            "trace": self.messages,
            "usage": self.total_usage
        }
        return agent_output