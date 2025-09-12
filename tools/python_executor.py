import json
import traceback
from .registry import tool_registry


def normalize_retrieved_resources(resources: dict) -> dict:
    """Auto-parse JSON strings and ensure consistent list format."""
    normalized = {}
    
    for key, value in resources.items():
        try:
            # Parse JSON strings
            if isinstance(value, str) and value.strip().startswith(('{', '[')):
                parsed = json.loads(value)
                normalized[key] = [parsed] if isinstance(parsed, dict) else parsed
            # Wrap single dicts in lists
            elif isinstance(value, dict):
                normalized[key] = [value]
            else:
                normalized[key] = value
        except:
            normalized[key] = value
            
    return normalized


def execute_python_code(code: str, timeout: int = 60, global_vars: dict = None) -> str:
    """Execute Python code and return the result from 'answer' variable.
    When comparing or calculating differences between dates, make sure to set both dates to timezone-naive so the comparison works without code failure."""
    if global_vars is None:
        global_vars = {}
    
    try:
        # Add common imports and the retrieved resources to global scope
        exec_globals = {
            'json': json,
            're': __import__('re'),
            'datetime': __import__('datetime'),
            'math': __import__('math'),
            'statistics': __import__('statistics'),
            'answer': None,  # Initialize answer variable
            **global_vars  # Include any provided global variables (like retrieved_resources)
        }
        
        # Normalize retrieved_resources for consistent data structure
        if 'retrieved_resources' in exec_globals and isinstance(exec_globals['retrieved_resources'], dict):
            exec_globals['retrieved_resources'] = normalize_retrieved_resources(exec_globals['retrieved_resources'])
        
        # Execute the code
        exec(code, exec_globals)
        
        answer = exec_globals.get('answer', None)
        if answer is not None:
            return {"answer": answer}
        else:
            return {"error": "Code executed successfully (no answer variable set)"}
            
    except Exception as e:
        error_info = traceback.format_exc()
        return {"error": f"Error executing code: {e}\n\nFull traceback:\n{error_info}"}


# Register tool
tool_registry.register_tool("execute_python_code", execute_python_code, {
    "type": "function",
    "function": {
        "name": "execute_python_code",
        "description": "Execute Python code with access to retrieved_resources and capture both stdout and answer variable. When comparing or calculating differences between dates, make sure to set both dates to timezone-naive so the comparison works without code failure.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute. Use 'answer' variable to store final result."},
                "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 30}
            },
            "required": ["code"]
        }
    }
})
