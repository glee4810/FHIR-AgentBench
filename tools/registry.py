from typing import Dict, List, Callable, Any


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._definitions: List[Dict[str, Any]] = []
    
    def register_tool(self, name: str, function: Callable, definition: Dict[str, Any]):
        self._tools[name] = function
        self._definitions.append(definition)
    
    def get_tool(self, name: str) -> Callable:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found")
        return self._tools[name]
    
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        return self._definitions.copy()


tool_registry = ToolRegistry()
