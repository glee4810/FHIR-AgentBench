import json
import re
from .base_agent import BaseAgent
from tools import get_tool_definitions, get_tool
from utils import safe_llm_call
from tools.resource_tools import supported_types


class MultiTurnCodeResourceAgent(BaseAgent):
    """Multi-step ReAct agent that iteratively uses FHIR resource tools to retrieve patient data and perform reasoning via Python code execution."""

    def __init__(self, model: str, max_iterations: int = 30, verbose: bool = False, base_url=None):
        super().__init__(model, verbose, base_url)

        all_tools = get_tool_definitions()
        self.tools = [tool for tool in all_tools if tool["function"]["name"] in 
                     ["get_resources_by_patient_fhir_id", "get_resources_by_resource_id", "execute_python_code"]]

        self.system_msg = [{"role": "system", "content": f"""You are a helpful assistant that can answer questions about patient data.

You have access to the following tools via function calling:
- {', '.join([tool['function']['name'] for tool in self.tools])}

Available FHIR resource types: {', '.join(supported_types)}. You can only call on these FHIR resources types for retrieval.

To answer questions about patient data:
1. Use get_resources_by_patient_fhir_id or get_resources_by_resource_id to retrieve relevant FHIR resources
2. Use execute_python_code to analyze the retrieved data
3. When you have completed your analysis and are ready to provide the final answer, you MUST format your response as follows:

   The final answer is: [your answer here]

When you retrieve FHIR resources, they will be automatically available in your code execution environment as:
- retrieved_resources: dictionary with resource types as keys containing the retrieved FHIR data
When calling execute_python_code and generating code, refer to the following pointers:
    - Ensure code is syntactically correct and runnable without errors.
    - Compare dates in a timezone-naive way.

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
The final answer is: 0 ICU visits during last calendar year (01/01/2152–12/31/2152).
Reasoning steps:
- Retrieved all Encounter resources.
- Filtered those with identifier.system containing 'encounter-icu.'
- Parsed period.start, counted those with year 2152.
- Result: 0.

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
first = sorted(hosp, key=lambda e: datetime.fromisoformat(e['period']['start']).replace(tzinfo=None))[0]
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
The final answer is: During the patient's first hospital encounter, that procedure occurred 1 time.
Reasoning steps:
- Retrieved Encounters and Procedures.
- Selected first 'encounter-hosp' by start date.
- Normalized display strings for target and each procedure.
- Counted exact matches → 1.

EXAMPLE 3:
Question: Did patient 10022041 have any diagnosis in their last hospital visit?
Context: The patient's FHIR ID is 52462b6a-9b39-5460-9ee6-1a2d7a20394e.
Tool Call Execution:
1) get_resources_by_patient_fhir_id with resource_type='Encounter' → retrieved_fhir_resources['Encounter'] (3 items)
2) get_resources_by_patient_fhir_id with resource_type='Condition' → retrieved_fhir_resources['Condition'] (13 items)
3) execute_python_code → find last hospital encounter (by identifier.system or class.code), include child encounters, count & list conditions
Code Execution:
```python
from datetime import datetime
import dateutil.parser as dp
encs  = retrieved_resources['Encounter']
conds = retrieved_resources['Condition']
# find hospital encounters
hosp = [e for e in encs
        if any('encounter-hosp' in i.get('system','').lower()
               for i in e.get('identifier',[]))]
# if none, use class.code == 'IMP'
if not hosp:
    hosp = [e for e in encs if e.get('class',{{}}).get('code','').upper()=='IMP']
# last by period.start
hosp_sorted = sorted(hosp, key=lambda e: dp.isoparse(e['period']['start']).replace(tzinfo=None))
last = hosp_sorted[-1]
pid_set = {{last['id']}}
# include children
for e in encs:
    if e.get('partOf',{{}}).get('reference','').endswith(last['id']):
        pid_set.add(e['id'])
# match conditions
matches = []
for c in conds:
    ref = c.get('encounter',{{}}).get('reference','').split('/')[-1]
    if ref in pid_set:
        code = c.get('code',{{}}).get('coding',[])
        matches.append(code[0].get('display'))
unique = sorted(set(matches))
print(len(matches), unique)
```
The final answer is: Yes. There were 13 Condition records (12 unique diagnoses) recorded during the patient's last hospital visit. The diagnoses were:
- 78039 — Other convulsions
- 78097 — Altered mental status
- 4019  — Unspecified essential hypertension
... (total 12 unique)
Reasoning steps:
- Retrieved Encounters and Conditions.
- Found last 'encounter-hosp' or IMP.
- Collected encounter + child encounter IDs.
- Filtered Condition.encounter references.
- Counted 13 records, extracted 12 unique displays.

Follow these examples to structure your approach: retrieve relevant FHIR resources first, then use Python code to analyze and process the data systematically."""}]

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
                        resource_type = tool_args["resource_type"]
                        if len(tool_output) == 0:
                            resource_len = 0
                        else:
                            resource_len = len(tool_output[resource_type])
                        saved_as_variable = f"""retrieved_fhir_resources['{resource_type}'] ({resource_len} items)"""

                        if self.verbose:
                            print(f"Tool execution successful. Output snippet: {str(tool_output)[:500]}...")

                        self.messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_name,
                            "content": saved_as_variable
                        })

                    elif tool_name == "execute_python_code":

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
                            "final_answer": f"Expected 'execute_python_code' or 'get_resources_by_patient_fhir_id' or 'get_resources_by_resource_id' tool call, but got '{tool_name}'",
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