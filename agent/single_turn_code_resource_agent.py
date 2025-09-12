import json
import re
from .base_agent import BaseAgent
from tools import get_tool_definitions, get_tool
from utils import safe_llm_call
from tools.resource_tools import supported_types


class SingleTurnCodeResourceAgent(BaseAgent):
    """Single-step agent that uses a FHIR resource tool to retrieve patient data and perform reasoning via Python code execution."""

    def __init__(self, model: str, verbose: bool = False, base_url=None):
        super().__init__(model, verbose, base_url)

        all_tools = get_tool_definitions()
        self.retrieval_tool = [tool for tool in all_tools if tool["function"]["name"] in 
                     ["get_resources_by_patient_fhir_id"]]
        self.reasoning_tool = [tool for tool in all_tools if tool["function"]["name"] in 
                     ["execute_python_code"]]        

        self.system_msg = [{"role": "system", "content": f"""You are a helpful assistant that answers questions about patient data.

You are given the following tools to retrieve and analyze patient data: [{', '.join([tool['function']['name'] for tool in self.retrieval_tool + self.reasoning_tool])}].
The available resource types are: [{', '.join(supported_types)}]. You can only call on these FHIR resources types for retrieval.
First, you should retrieve the patient data using get_resources_by_patient_fhir_id. Then, you should reason about the retrieved data using execute_python_code and provide the answer based on the available information.

When you retrieve FHIR resources, they will be automatically available in your code execution environment as:
- retrieved_resources: dictionary with resource types as keys containing the retrieved FHIR data
    
When you provide answers, make sure to provide them in the same format as they are in the retrieved data. If multiple answers are provided, provide them all in a list.
If you cannot find the answer or relevant patient data, clearly state that you cannot find the information.
Do not guess attributes; instead, use the provided tool to retrieve the data.
Do not get stuck or repeat the same action.
Do not plan ahead. Instead, directly use the tool to retrieve and reason about the data.

Ensure code is syntactically correct and runnable without errors.
When generating code, make sure to first remove timezones so you can compare dates in a timezone-naive way without error.

Few-shot examples:

EXAMPLE 1:
Question: Calculate the number of ICU visits for patient 10009628 last year.
Context: The patient's FHIR ID is 51d2190c-cc46-56c5-b2ea-363895cbea75. Assume current time is 2153-12-31 23:59:00.
Tool Call Execution:
1) get_resources_by_patient_fhir_id with resource_type='Encounter' → retrieved_fhir_resources['Encounter'] (2 items)
2) execute_python_code → filter ICU encounters and count those in 2152 calendar year
Code Execution:
```python
from datetime import datetime
encs = retrieved_resources['Encounter']
start = datetime(2152,1,1)
end   = datetime(2152,12,31,23,59,59)
count = 0
for e in encs:
    ids = [i.get('system','').lower() for i in e.get('identifier',[])]
    if not any('encounter-icu' in s for s in ids): continue
    s = e.get('period',{{}}).get('start')
    if not s: continue
    dt = datetime.fromisoformat(s).replace(tzinfo=None)
    if start <= dt <= end:
        count += 1
print(count)
```
Result: 0 ICU visits during last calendar year (01/01/2152–12/31/2152).

EXAMPLE 2:
Question: Count how many times during their first hospital encounter patient 10018423 experienced the bypass coronary artery, one artery from left internal mammary with autologous arterial tissue, open approach procedure.
Context: The patient's FHIR ID is bbad4581-d089-54a7-b7a0-8d986c5fb5ec. Account for case-insensitive/whitespace variations.
Tool Call Execution:
1) get_resources_by_patient_fhir_id with resource_type='Encounter' → retrieved_fhir_resources['Encounter'] (4 items)
2) get_resources_by_patient_fhir_id with resource_type='Procedure' → retrieved_fhir_resources['Procedure'] (35 items)
3) execute_python_code → identify first hospital encounter, normalize target, count matching procedures
Code Execution:
```python
from datetime import datetime
import re
encs = retrieved_resources['Encounter']
procs = retrieved_resources['Procedure']
# first hospital encounter
hosp = [e for e in encs
        if any('encounter-hosp' in i.get('system','').lower()
               for i in e.get('identifier',[]))]
first = sorted(hosp, key=lambda e: datetime.fromisoformat(e['period']['start'].replace(tzinfo=None))[0]
ref = f"Encounter/{{first['id']}}"
# normalize function
norm = lambda s: re.sub(r'\\s+',' ',(s or '').strip().lower())
target = norm("bypass coronary artery, one artery from left internal mammary with autologous arterial tissue, open approach")
# count matches
count = 0
for p in procs:
    if p.get('encounter',{{}}).get('reference') != ref: continue
    for c in p.get('code',{{}}).get('coding',[]):
        if norm(c.get('display')) == target:
            count += 1
            break
print(count)
```
Result: During the patient's first hospital encounter, that procedure occurred 1 time.

Follow these examples to structure your approach: retrieve relevant FHIR resources first, then use Python code to analyze and process the data systematically."""}]


    def run(self, question: str) -> dict:
        
        self.messages = self.system_msg.copy()
        self.messages.append({"role": "user", "content": question})
        retrieved_resources = {}
        
        # 1. Retrieve patient data
        response_message, error, usage_info = safe_llm_call(
            model=self.model,
            messages=self.messages,
            tools=self.retrieval_tool,
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

                if tool_name == "get_resources_by_patient_fhir_id":

                    if self.verbose:
                        print(f"Agent detected tool call: '{tool_name}' with arguments: {tool_args}")
                    
                    tool_function = get_tool(tool_name)
                    tool_output = tool_function(**tool_args)
                    retrieved_resources.update(tool_output)
                    resource_type = tool_args["resource_type"]
                    if len(tool_output) == 0:
                        resource_len = 0
                    else:
                        resource_len = len(tool_output[resource_type])
                    saved_as_variable = f"""retrieved_fhir_resources['{resource_type}'] ({resource_len} items)"""                    
                    
                    if self.verbose:
                        print(f"Tool execution successful. Output snippet: {str(tool_output)[:500]}...")

                    self.messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_name,
                            "content": saved_as_variable
                        }
                    )
                
                else:
                    return {
                        "retrieved_fhir_resources": {},
                        "final_answer": f"Expected 'get_resources_by_patient_fhir_id' tool call, but got '{tool_name}'",
                        "trace": self.messages,
                        "usage": self.total_usage
                    }

        else:
            return {
                "retrieved_fhir_resources": {},
                "final_answer": "No tool calls found in the response",
                "trace": self.messages,
                "usage": self.total_usage
            }

        # 2. Reason about the retrieved data using execute_python_code
        response_message, error, usage_info = safe_llm_call(
            model=self.model,
            messages=self.messages,
            tools=self.reasoning_tool,
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

            tool_call = response_message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            if self.verbose:
                print(f"Agent detected tool call: '{tool_name}' with arguments: {tool_args}")

            if tool_name == "execute_python_code":

                if self.verbose:
                    print(f"Agent detected tool call: '{tool_name}' with arguments: {tool_args}")

                tool_function = get_tool(tool_name)
                tool_args['global_vars'] = {'retrieved_resources': retrieved_resources}
                tool_output = tool_function(**tool_args)
                
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
                    "final_answer": f"Expected 'execute_python_code' tool call, but got '{tool_name}'",
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

        # 3. Combine the retrieved data and the reasoning about the retrieved data
        final_message, final_error, final_usage_info = safe_llm_call(
            model=self.model,
            messages=self.messages,
            base_url=self.base_url
        )
        
        self._update_usage(final_usage_info)
        self.messages.append(final_message)
        
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