import json
from .registry import tool_registry
from .cache import cached_tool
from fhir_client import get_fhir_client

supported_types = [
    "Patient", "Encounter", "Condition", "MedicationRequest", 
    "Procedure", "Observation", "MedicationAdministration",
    "Location", "Specimen", "Medication"
]

@cached_tool
def get_resources_by_resource_id(resource_type: str, resource_id: str) -> dict:
    """Get FHIR resources by resource ID."""
    try:
        client = get_fhir_client()
        query = f"{resource_type}?id={resource_id}" if resource_id else resource_type
        resources = client.search_with_pagination(query)
        
        resources_by_type = {}
        for resource in resources:
            rt = resource.get('resourceType', 'Unknown')
            if rt not in resources_by_type:
                resources_by_type[rt] = []
            resources_by_type[rt].append(resource)
        
        return resources_by_type
    except Exception as e:
        return {"error": str(e)}

@cached_tool
def get_resources_by_patient_fhir_id(resource_type: str, patient_fhir_id: str) -> dict:
    """Get FHIR resources by patient FHIR ID."""
    try:
        client = get_fhir_client()
        query = f"{resource_type}?patient={patient_fhir_id}" if patient_fhir_id else resource_type
        resources = client.search_with_pagination(query)
        
        resources_by_type = {}
        for resource in resources:
            rt = resource.get('resourceType', 'Unknown')
            if rt not in resources_by_type:
                resources_by_type[rt] = []
            resources_by_type[rt].append(resource)
        
        return resources_by_type
    except Exception as e:
        return {"error": str(e)}

@cached_tool
def get_all_resources_by_patient_fhir_id(patient_fhir_id: str) -> dict:
    """Get all FHIR resources by patient FHIR ID."""
    try:
        client = get_fhir_client()
        all_resources = []        
        for resource_type in supported_types:
            query = f"{resource_type}?patient={patient_fhir_id}" if patient_fhir_id else resource_type
            resources = client.search_with_pagination(query)
            all_resources.extend(resources)
        
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


tool_registry.register_tool("get_resources_by_resource_id", get_resources_by_resource_id, {
    "type": "function",
    "function": {
        "name": "get_resources_by_resource_id",
        "description": "Get FHIR resources by resource ID",
        "parameters": {
            "type": "object",
            "properties": {
                "resource_type": {"type": "string", "description": "FHIR resource type"},
                "resource_id": {"type": "string", "description": "FHIR resource ID"}
            },
            "required": ["resource_type", "resource_id"]
        }
    }
})

tool_registry.register_tool("get_resources_by_patient_fhir_id", get_resources_by_patient_fhir_id, {
    "type": "function",
    "function": {
        "name": "get_resources_by_patient_fhir_id",
        "description": "Get FHIR resources by patient FHIR ID",
        "parameters": {
            "type": "object",
            "properties": {
                "resource_type": {"type": "string", "description": "FHIR resource type"},
                "patient_fhir_id": {"type": "string", "description": "Patient FHIR ID"}
            },
            "required": ["resource_type", "patient_fhir_id"]
        }
    }
})

tool_registry.register_tool("get_all_resources_by_patient_fhir_id", get_all_resources_by_patient_fhir_id, {
    "type": "function",
    "function": {
        "name": "get_all_resources_by_patient_fhir_id",
        "description": "Get all FHIR resources by patient FHIR ID",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_fhir_id": {"type": "string", "description": "Patient FHIR ID"}
            },
            "required": ["patient_fhir_id"]
        }
    }
})
