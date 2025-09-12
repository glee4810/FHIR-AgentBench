import os
import yaml
import sys
sys.path.append("..")

from google.auth import default
from google.auth.transport import requests
from typing import Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import get_fhir_config

JsonObject = list[dict[str, Any]]

def get_auth_session_from_default():
    credentials, project_id = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    return requests.AuthorizedSession(credentials)

def fhir_store_construct_url(project_id, location, dataset_id, fhir_store_id):
    base_url = "https://healthcare.googleapis.com/v1"
    return f"{base_url}/projects/{project_id}/locations/{location}/datasets/{dataset_id}/fhirStores/{fhir_store_id}/fhir"

class FHIRClient:
    """
    Base class to interact with Cloud Healthcare API FHIR search
    """
    def __init__(self, project_id: str, location: str, dataset_id: str, fhir_store_id: str, *, creds=None):
        """
        Initialize a FHIR search client.
        """
        
        # Build URL        
        self.fhir_store_url = fhir_store_construct_url(project_id, location, dataset_id, fhir_store_id)

        # ADC if creds not supplied
        self.session = creds or get_auth_session_from_default()

        # For debug and logging
        self._info = {
            "project_id": project_id,
            "location": location,
            "dataset_id": dataset_id,
            "store_id": fhir_store_id,
            "fhir_store_url": self.fhir_store_url,
        }

    # Utilities below are unchanged except that they use self.session and self.fhir_store_url
    def remove_fields(self, resource: JsonObject, fields: list[str]) -> JsonObject:
        for field in fields:
            if field in resource:
                del resource[field]
        return resource

    def _fetch_resources_with_pagination(self, initial_resource_path: str) -> list[JsonObject]:
        """
        Common function to fetch resources with pagination support.
        """
        all_resources = []
        resource_path = initial_resource_path

        while True:
            response = self.session.get(resource_path)
            response.raise_for_status()
            resources = response.json()

            if resources.get("entry", []):
                all_resources.extend([self.remove_fields(e["resource"], ["text", "meta"]) for e in resources["entry"]])

            next_url = None
            for link in resources.get("link", []):
                if link.get("relation") == "next":
                    next_url = link.get("url")
                    break
            if not next_url:
                break
            resource_path = next_url

        return all_resources

    def get_resources_by_resource_ids(self, resource_type: str, resource_ids: list[str], max_size: int = 1000) -> JsonObject:
        """Get resources by their IDs."""
        if isinstance(resource_ids, str):
            resource_ids = [resource_ids]
        
        resource_ids_str = ",".join(resource_ids)
        resource_path = f"{self.fhir_store_url}/{resource_type}?_id={resource_ids_str}&_count={max_size}"
        return self._fetch_resources_with_pagination(resource_path)


    def search_with_pagination(self, query_string: str) -> list[dict]:
        """
        Perform a FHIR search with automatic pagination handling.
        
        Args:
            query_string (str): The FHIR query string (e.g., "Patient?_count=100")
        
        Returns:
            list[dict]: All paginated resources
        """
        resource_path = f"{self.fhir_store_url}/{query_string}"
        return self._fetch_resources_with_pagination(resource_path)
    

    
def get_fhir_client(
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    dataset_id: Optional[str] = None,
    store_id: Optional[str] = None,
    *,
    creds=None,
) -> FHIRClient:
    """
    Factory function to build a fhir_search from params or env/config.
    Callers can pass explicit params or rely on env/config.
    """

    cfg = get_fhir_config()
    project_id = project_id or cfg.get("PROJECT_ID")
    location = location or cfg.get("LOCATION")
    dataset_id = dataset_id or cfg.get("DATASET_ID")
    store_id = store_id or cfg.get("STORE_ID")

    missing = [k for k, v in [("PROJECT_ID", project_id), ("LOCATION", location), ("DATASET_ID", dataset_id), ("STORE_ID", store_id)] if not v]
    if missing:
        raise ValueError(f"Missing FHIR config keys: {', '.join(missing)}")

    return FHIRClient(project_id, location, dataset_id, store_id, creds=creds)
