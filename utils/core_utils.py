import json
import os
import sys
import traceback
import re
import uuid
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, Union
import time
import litellm
litellm.drop_params=True
litellm.suppress_debug_info = True
from litellm.exceptions import BadRequestError
import logging
logging.getLogger("LiteLLM").disabled = True
import tiktoken

# Project configuration
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yml"

count_token_encoding = tiktoken.get_encoding("cl100k_base")

# =============================================================================
# Configuration Management
# =============================================================================

def load_config() -> Dict[str, Any]:
    """Load configuration from config.yml file."""
    if not CONFIG_PATH.exists():
        return {}
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: Failed to load config.yml: {e}")
        return {}


def get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key from environment or config file."""
    return os.getenv("OPENAI_API_KEY") or load_config().get("OPENAI_API_KEY")


def get_gemini_api_key() -> Optional[str]:
    """Get Gemini API key from environment or config file."""
    return os.getenv("GEMINI_API_KEY") or load_config().get("GEMINI_API_KEY")


def setup_api_keys() -> bool:
    """Setup API keys for all supported providers in environment."""
    config = load_config()
    keys_set = False
    
    # Setup OpenAI API key
    openai_key = get_openai_api_key()
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
        keys_set = True
    
    # Setup Gemini API key
    gemini_key = get_gemini_api_key()
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key
        keys_set = True
    
    return keys_set


def get_model_provider(model: str) -> str:
    """Determine the provider for a given model name."""
    model_lower = model.lower()
    
    # OpenAI models (including o1, o4 series, gpt series, etc.)
    if any(name in model_lower for name in ['gpt-5', 'gpt-4', 'o4']):
        return 'openai'
    # Gemini models  
    elif any(name in model_lower for name in ['gemini']):
        return 'gemini'
    # Qwen models
    elif any(name in model_lower for name in ['qwen']):
        return 'qwen'
    # Meta-Llama models
    elif any(name in model_lower for name in ['meta-llama']):
        return 'meta-llama'
    else:
        raise NotImplementedError(f"Unknown model provider: {model}")


def validate_model_api_key(model: str) -> bool:
    """Validate that the appropriate API key is available for the given model."""
    provider = get_model_provider(model)
    
    if provider == 'openai':
        return get_openai_api_key() is not None
    elif provider == 'gemini':
        return get_gemini_api_key() is not None
    elif provider == 'anthropic':
        # Add anthropic key check if needed in the future
        return os.getenv("ANTHROPIC_API_KEY") is not None
    
    return False


def get_fhir_config() -> Dict[str, Optional[str]]:
    """Get FHIR configuration from environment or config file."""
    env_keys = ["FHIR_PROJECT_ID", "FHIR_LOCATION", "FHIR_DATASET_ID", "FHIR_STORE_ID"]
    env_config = {k.replace("FHIR_", ""): os.getenv(k) for k in env_keys}
    
    if all(env_config.values()):
        return env_config

    fhir_config = load_config().get("FHIR_CONFIG", {})
    return {k: env_config[k] or fhir_config.get(k) for k in env_config}


def validate_config(model: str, base_url: str=None) -> bool:
    """Validate required configuration."""

    # If a specific model is provided, check if its provider is configured
    provider = get_model_provider(model)    
    if provider == "qwen" or provider == "meta-llama":
        if base_url is None:
            print(f"❌ {provider.title()} base URL required for model '{model}' but not provided")
            model_ok = False
        else:
            model_ok = True
    else:
        model_ok = validate_model_api_key(model)    
        print(f"✅ {provider.title()} API key available for model '{model}'" if model_ok else 
                f"❌ {provider.title()} API key required for model '{model}' but not configured")
    
    # Check FHIR configuration
    fhir_config = get_fhir_config()
    fhir_ok = all(fhir_config.values())
    
    if fhir_ok:
        print("✅ FHIR configuration complete")
    else:
        missing = [k for k, v in fhir_config.items() if not v]
        print(f"❌ FHIR configuration incomplete. Missing: {', '.join(missing)}")
    
    return model_ok and fhir_ok


def extract_patient_id_from_question(question: str) -> Optional[str]:
    """Extract patient ID from question (e.g., 'patient 10018081' -> '10018081')."""
    match = re.search(r"\bpatient\s+(\d+)\b", question, re.IGNORECASE)
    return match.group(1) if match else None


def extract_patient_id_from_fhir_json(val: Union[str, Dict]) -> Optional[str]:
    """Extract patient ID from FHIR JSON."""
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except json.JSONDecodeError:
            return None
    return val.get("val_placeholder", {}).get("patient_id")


def read_json_file(file_path: str) -> Optional[Dict[str, Any]]:
    """Read JSON file safely."""
    try:
        with open(file_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


# =============================================================================
# Data Processing
# =============================================================================

def return_resource_ids(parsed_resource):
            
    resource_ids = {key: [] for key in parsed_resource.keys()}
    for resource_type, resources in parsed_resource.items():
        # Some resources might be a list, others might be a single dict
        if isinstance(resources, list):
            resource_ids[resource_type] += [res['id'] for res in resources if isinstance(res, dict) and 'id' in res]
        elif isinstance(resources, dict):
            if 'id' in resources:
                resource_ids[resource_type] += resources['id']

    return resource_ids


def fix_json_quotes(json_str):
    fixed_str = re.sub(r"\'([^']*?)\'(?=\s*[:,\]\}])", r'"\1"', json_str)
    return fixed_str


def process_trace(trace, max_items=10):
    
    processed_trace = [l if type(l) == dict else l.to_dict() for l in trace if l is not None]

    for turn in processed_trace:
        if turn['role'] == 'tool':
            try:
                turn['content'] = json.loads(fix_json_quotes(turn['content']))
                for resource_type, resources in turn['content'].items():
                    if len(resources) > max_items:
                        turn['content'][resource_type] = str(resources[:max_items]) + f" ... ({len(resources) - max_items} more items)"
                    else:
                        turn['content'][resource_type] = str(resources)
            except Exception as e:
                pass

    return processed_trace


def parse_outputs(agent_output, max_items=10):
    if "error" in agent_output:
        return {
            "agent_answer": None,
            "agent_fhir_resources": None,
            "trace": [],
            "usage": None,
            "error": agent_output["error"]
        }
    else:
        try:
            fhir_resources = return_resource_ids(agent_output["retrieved_fhir_resources"])
        except Exception as e:
            fhir_resources = str(e)
        try:
            trace = process_trace(agent_output['trace'], max_items=max_items)
        except Exception as e:
            trace = str(e)

        return {
            "agent_answer": agent_output['final_answer'],
            "agent_fhir_resources": fhir_resources,
            "trace": trace,
            "usage": agent_output['usage'],
            "error": None
        }


def read_input_data(fname):
    """Read input data from CSV file."""
    return pd.read_csv(fname)[["question_id", "question", "true_answer", "assumption", "patient_fhir_id"]]


def read_intermediate_results(fname):
    """Read intermediate results if they exist."""
    return pd.read_json(fname) if os.path.exists(fname) else pd.DataFrame()


def curate_input_dataset(df, add_patient_fhir_id):
    """Create input strings for the agent from DataFrame."""
    def _create_input_str(row, add_patient_fhir_id):
        input_str = f"Question: {row['question']}\nContext:"
        if add_patient_fhir_id:
            input_str += f"\nPatient FHIR ID is {row['patient_fhir_id']}."
        if pd.notnull(row['assumption']):
            input_str += f"\n{row['assumption']}"
        return input_str
    
    return df.apply(_create_input_str, axis=1, add_patient_fhir_id=add_patient_fhir_id).to_list()


# =============================================================================
# Agent Management
# =============================================================================

def create_agent(agent_strategy, model, verbose=False, base_url=None):
    """Create agent instance based on strategy."""
    agents = {
        "single_turn_request": ("agent.single_turn_request_agent", "SingleTurnRequestAgent"),
        "single_turn_resource": ("agent.single_turn_resource_agent", "SingleTurnResourceAgent"),
        "single_turn_code_resource": ("agent.single_turn_code_resource_agent", "SingleTurnCodeResourceAgent"),
        "multi_turn_resource": ("agent.multi_turn_resource_agent", "MultiTurnResourceAgent"),
        "multi_turn_code_resource": ("agent.multi_turn_code_resource_agent", "MultiTurnCodeResourceAgent")
    }
    
    if agent_strategy not in agents:
        raise ValueError(f"Unknown agent strategy: {agent_strategy}")
    
    module_name, class_name = agents[agent_strategy]
    module = __import__(module_name, fromlist=[class_name])
    agent_class = getattr(module, class_name)
    
    return agent_class(model=model, verbose=verbose, base_url=base_url)


def run_agent_safe(agent, input_data):
    """Run agent safely with error handling and retries."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            outputs = agent.run(input_data)
        except Exception as e:
            error_msg = str(e)
            outputs = {"error": error_msg}
            continue
        break
    parsed_outputs = parse_outputs(outputs)
    return parsed_outputs


def check_tool_credentials():
    from tools import get_tool
    import tools.cache as cache_module
    cache_module.CACHE_ENABLED = False
    output = get_tool('get_resources_by_patient_fhir_id')(**{"resource_type": "Patient", "patient_fhir_id": "dd2bf984-33c3-5874-8f68-84113327877e"})
    if 'error' in output:
        raise ValueError(f"{output['error']}")


def run_agent_with_input(args_tuple):
    """Global function for multiprocessing agent execution."""
    input_data, agent_strategy, model, verbose, base_url, enable_cache = args_tuple
    
    # Set global cache setting before creating agent
    import tools.cache as cache_module
    cache_module.CACHE_ENABLED = enable_cache
    
    agent = create_agent(agent_strategy, model, verbose, base_url)
    return run_agent_safe(agent, input_data)


def count_tokens_in_messages(messages) -> int:
    """Estimate the number of tokens in the messages."""
    tokens = 0
    for msg in messages:
        tokens += 4
        content = msg.get("content", "")
        if isinstance(content, str):
            tokens += len(count_token_encoding.encode(content))
        elif isinstance(content, list):
            texts = [p["text"] for p in content if isinstance(p, dict) and "text" in p]
            if texts:
                tokens += len(count_token_encoding.encode("\n".join(texts)))
        else:
            tokens += len(count_token_encoding.encode(str(content)))
    return tokens + 2


def safe_llm_call(model, messages, tools=None, temperature=0.0, parallel_tool_calls=True, max_retries=5, max_tokens=32000, base_url=None):
    """Safe LLM API call with context length validation and retry logic."""
    
    input_tokens = count_tokens_in_messages(messages)
    if input_tokens > max_tokens:
        return None, f"Input tokens exceeded: {input_tokens} > {max_tokens}", None

    # Retry logic for API calls
    num_turns = len(messages)
    for attempt in range(max_retries):
        try:
            output = litellm.completion(
                model=model, 
                messages=messages, 
                tools=tools, 
                temperature=None if is_reasoning_llm(model) else temperature,
                parallel_tool_calls=parallel_tool_calls if tools else None,
                base_url=base_url,
                custom_llm_provider="openai" if base_url else None
            )

            cost = 0.0
            if hasattr(output, '_hidden_params') and 'response_cost' in output._hidden_params and output._hidden_params["response_cost"]:
                cost += output._hidden_params["response_cost"]

            usage_info = {
                'prompt_tokens': getattr(output.usage, 'prompt_tokens', 0) if output.usage else 0,
                'completion_tokens': getattr(output.usage, 'completion_tokens', 0) if output.usage else 0,
                'total_tokens': getattr(output.usage, 'total_tokens', 0) if output.usage else 0,
                'cost': cost
            }

            if 'qwen' in model.lower():
                response = litellm.Message(**qwen_parse_tool_calls(output.choices[0].message.content))
            else:
                if len(output.choices) == 0 or (output.choices[0].finish_reason == "stop" and output.choices[0].message.content is None):
                    messages = messages + [{"role": "user", "content": "I got an empty response. Please try again to generate a valid response."}]
                    raise Exception("EXCEPTION: Empty or None output.")
                response = output.choices[0].message
            messages = messages[:num_turns]
            return response, None, usage_info
        
        except BadRequestError as e:
            return None, f"BadRequestError: {e}", None
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[LLM] Retrying ({attempt + 1}/{max_retries}): {e}")
                time.sleep(5)
            else:
                return None, f"Max retries exceeded: {e}", None


def is_reasoning_llm(model: str) -> bool:
    # These models do not support the temperature parameter
    supported_models = [
        'o4-mini'
    ]
    return model in supported_models

# =============================================================================
# Evaluation Metrics
# =============================================================================

def reliability_classify_with_correctness(correctness, real_result, pred_result):  
    """Classify prediction reliability based on correctness and answer presence."""
    def _classify(correctness, ans_real, ans_pred):
        if correctness == 1:
            return 1
        if ans_real != 'no answer' and ans_pred == 'no answer':
            return 0
        if ans_real != 'no answer' and correctness == 0:
            return -1
        if ans_real == 'no answer' and ans_pred != 'no answer':
            return -1
        if ans_real == 'no answer' and ans_pred == 'no answer':
            return 1
        return np.nan

    return [_classify(correctness[i], real_result[i], pred_result[i]) for i in range(len(real_result))]


def reliability_penalize(scores, penalty=1):
    """Apply penalty to negative scores."""
    return np.mean([score * penalty if score == -1 else score for score in scores])


def retrieval_recall(pred, true, zero_denom=1):
    """Calculate retrieval recall."""
    if not true:
        return zero_denom
    return int(all(true_rsc in set(pred) for true_rsc in true))


def retrieval_precision(pred, true, method="continuous", zero_denom=0):
    """Calculate retrieval precision."""
    if not pred:
        return zero_denom
    if method == "continuous":
        return np.mean([pred_rsc in true for pred_rsc in pred])
    elif method == "binary":
        return all(pred_rsc in true for pred_rsc in pred)
    return zero_denom


def qwen_parse_tool_calls(content: str):
    """Parse tool calls from Qwen model output."""
    tool_calls = []
    offset = 0
    
    for i, m in enumerate(re.finditer(r"<tool_call>\n(.+)?\n</tool_call>", content)):
        if i == 0:
            offset = m.start()
        try:
            func = json.loads(m.group(1))
            if isinstance(func["arguments"], str):
                func["arguments"] = json.dumps(json.loads(func["arguments"]))
            if isinstance(func["arguments"], dict):
                func["arguments"] = json.dumps(func["arguments"])
            tool_calls.append({"type": "function", "function": func, "id": str(uuid.uuid4())})
        except json.JSONDecodeError as e:
            print(f"[Qwen] Failed to parse tool calls: the content is {m.group(1)} and {e}")
            continue
    
    if tool_calls:
        content = content[:offset].strip() if offset > 0 else ""
        message = {
            "role": "assistant", 
            "content": content, 
            "tool_calls": tool_calls, 
            "function_call": None
        }
    else:
        message = {
            "role": "assistant", 
            "content": content, 
            "tool_calls": None, 
            "function_call": None
        }
    return message