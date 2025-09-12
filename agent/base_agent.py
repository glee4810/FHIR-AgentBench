import sys
import os
import litellm

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import setup_api_keys, validate_model_api_key, get_model_provider


class BaseAgent:
    """
    Base class for all FHIR agents.
    
    Provides essential functionality:
    - OpenAI API key setup from config.yml
    - Model configuration
    - Basic message management
    """
    
    def __init__(self, model: str, verbose: bool = False, base_url=None):
        """
        Initialize the base agent.
        
        Args:
            model (str): The LLM model to use
            verbose (bool): Whether to print debug information
        """
        self.model = model
        self.verbose = verbose
        self.messages = []
        self.base_url = base_url
        
        # Initialize usage tracking
        self.total_usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'cost': 0.0,
            'llm_calls': 0
        }
        
        # Setup API keys from config.yml
        setup_api_keys()
        
    def _update_usage(self, usage_info):
        """Update total usage statistics."""
        if usage_info:
            self.total_usage['prompt_tokens'] += usage_info.get('prompt_tokens', 0)
            self.total_usage['completion_tokens'] += usage_info.get('completion_tokens', 0)
            self.total_usage['total_tokens'] += usage_info.get('total_tokens', 0)
            self.total_usage['cost'] += round(usage_info.get('cost', 0.0), 4)
            self.total_usage['llm_calls'] += 1

    def run(self, question: str) -> dict:
        raise NotImplementedError