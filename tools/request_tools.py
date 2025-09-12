import json
from .registry import tool_registry
from .cache import cached_tool
from fhir_client import get_fhir_client


@cached_tool
def fhir_request_get(query_string: str) -> dict:
    """FHIR GET search request."""
    try:
        client = get_fhir_client()
        response = client.session.get(f"{client.fhir_store_url}/{query_string}")
        response.raise_for_status()

        all_resources = []
        if 'entry' in response.json():
            for l in response.json()['entry']:
                all_resources.append(l['resource'])

        # Group by resource type
        resources_by_type = {}
        for resource in all_resources:
            rt = resource.get('resourceType', 'Unknown')
            if rt not in resources_by_type:
                resources_by_type[rt] = []
            resources_by_type[rt].append(resource)
        return resources_by_type
    except Exception as e:
        return {"error": str(e)}

# Register tools
tool_registry.register_tool("fhir_request_get", fhir_request_get, {
    "type": "function",
    "function": {
        "name": "fhir_request_get",
        "description": "FHIR GET request",
        "parameters": {
            "type": "object",
            "properties": {
                "query_string": {"type": "string", "description": "FHIR query string"}
            },
            "required": ["query_string"]
        }
    }
})
