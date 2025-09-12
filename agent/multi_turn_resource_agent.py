import json
import re
from .base_agent import BaseAgent
from tools import get_tool_definitions, get_tool
from utils import safe_llm_call
from tools.resource_tools import supported_types


class MultiTurnResourceAgent(BaseAgent):
    """Multi-step ReAct agent that iteratively uses FHIR resource tools to retrieve patient data and perform reasoning using natural language."""

    def __init__(self, model: str, max_iterations: int = 30, verbose: bool = False, base_url=None):
        super().__init__(model, verbose, base_url)

        all_tools = get_tool_definitions()
        self.tools = [tool for tool in all_tools if tool["function"]["name"] in 
                     ["get_resources_by_patient_fhir_id", "get_resources_by_resource_id"]]

        self.system_msg = [{"role": "system", "content": f"""You are a helpful assistant that can answer questions about patient data.

You have access to the following tools via function calling:
- {', '.join([tool['function']['name'] for tool in self.tools])}

Available FHIR resource types: {', '.join(supported_types)}. You can only call on these FHIR resources types for retrieval.

To answer questions about patient data:
1. Use get_resources_by_patient_fhir_id or get_resources_by_resource_id to retrieve relevant FHIR resources
2. Analyze the retrieved data to answer the question
3. When you have completed your analysis and are ready to provide the final answer, you MUST format your response as follows:

   The final answer is: [your answer here]

If there are multiple answers, provide all of them.
IMPORTANT: Always end your response with 'The final answer is:' followed by your conclusion. This is required for proper processing.
When you provide answers, make sure to provide them in the same format as they are in the retrieved data. If multiple answers are provided, provide them all in a list.
If you cannot find the answer or relevant patient data, clearly state that you cannot find the information.
Do not guess attributes; instead, use the provided tool to retrieve the data.
Do not get stuck or repeat the same action.

Few-shot examples:

EXAMPLE 1:
Question: Calculate the number of ICU visits for patient 10009628 last year.
Context: The patient's FHIR ID is 51d2190c-cc46-56c5-b2ea-363895cbea75. Assume current time is 2153-12-31 23:59:00.
Tool Call Execution:
1) get_resources_by_patient_fhir_id with resource_type='Encounter' → Retrieved 2 Encounter resources
Analysis: I need to filter for ICU encounters and count those from the previous year (2152).
Looking at the encounters, I can identify ICU visits by checking the identifier.system field for 'encounter-icu'.
For the time period, I need to parse the period.start field and check if it falls within 2152.
After examining both encounters, neither has an identifier containing 'encounter-icu', so there are no ICU visits.
The final answer is: 0 ICU visits during last calendar year (01/01/2152–12/31/2152).

EXAMPLE 2:
Question: Count how many times during their first hospital encounter patient 10018423 experienced the bypass coronary artery, one artery from left internal mammary with autologous arterial tissue, open approach procedure.
Context: The patient's FHIR ID is bbad4581-d089-54a7-b7a0-8d986c5fb5ec. Account for case-insensitive/whitespace variations.
Tool Call Execution:
1) get_resources_by_patient_fhir_id with resource_type='Encounter' → Retrieved 4 Encounter resources
2) get_resources_by_patient_fhir_id with resource_type='Procedure' → Retrieved 35 Procedure resources
Analysis: First, I need to identify the first hospital encounter by looking for encounters with 'encounter-hosp' in identifier.system, then sort by period.start date.
Once I have the first hospital encounter ID, I need to find procedures that reference this encounter and match the target procedure description.
I'll normalize the text for comparison to handle case-insensitive and whitespace variations.
After examining the procedures linked to the first hospital encounter, I found 1 procedure that matches the target description.
The final answer is: During the patient's first hospital encounter, that procedure occurred 1 time.

EXAMPLE 3:
Question: Did patient 10022041 have any diagnosis in their last hospital visit?
Context: The patient's FHIR ID is 52462b6a-9b39-5460-9ee6-1a2d7a20394e.
Tool Call Execution:
1) get_resources_by_patient_fhir_id with resource_type='Encounter' → Retrieved 3 Encounter resources
2) get_resources_by_patient_fhir_id with resource_type='Condition' → Retrieved 13 Condition resources
Analysis: I need to find the last hospital encounter, which can be identified by 'encounter-hosp' in identifier.system or class.code='IMP'.
Once I identify the last hospital encounter, I need to include any child encounters (those with partOf reference).
Then I'll look for Condition resources that reference any of these encounter IDs.
After examining the data, I found 13 Condition records linked to the last hospital visit, representing 12 unique diagnoses.
The final answer is: Yes. There were 13 Condition records (12 unique diagnoses) recorded during the patient's last hospital visit. The diagnoses included conditions like 'Other convulsions', 'Altered mental status', and 'Unspecified essential hypertension'.

Follow these examples to structure your approach: retrieve relevant FHIR resources first, then analyze the data systematically using natural language reasoning."""}]

        self.max_iterations = max_iterations

    def _is_final_answer(self, content: str) -> bool:
        if not content:
            return False
        return "the final answer is:" in content.lower().strip()

    def run(self, question: str) -> dict:

        self.messages = self.system_msg.copy()
        self.messages.append({"role": "user", "content": question})
        retrieved_resources = {}
        final_answer = None
        iteration = 0
        while final_answer is None and iteration < self.max_iterations:

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

                    if tool_name == "get_resources_by_patient_fhir_id" or tool_name == "get_resources_by_resource_id":

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
                            "final_answer": f"Expected 'get_resources_by_patient_fhir_id' or 'get_resources_by_resource_id' tool call, but got '{tool_name}'",
                            "trace": self.messages,
                            "usage": self.total_usage
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
            "usage": self.total_usage
        }