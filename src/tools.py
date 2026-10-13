import os
import json
import requests
from crewai.tools import tool

@tool("Execute FHIR Request")
def execute_fhir_query(path: str, method: str = "GET", body: str = None):
    """Executes an HTTP request to the FHIR server."""
    base_url = os.getenv("FHIR_SERVER")
    if not base_url:
        return "Error: Missing 'FHIR_SERVER' variable."
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    
    try:
        headers = {"Content-Type": "application/fhir+json"}
        if method.upper() == "POST":
            if isinstance(body, dict): body = json.dumps(body)
            response = requests.post(url, headers=headers, data=body)
        elif method.upper() == "PUT":
            response = requests.put(url, headers=headers, data=body)
        elif method.upper() == "DELETE":
            response = requests.delete(url, headers=headers)
        else:
            response = requests.get(url, headers=headers)
        
        if response.status_code in [200, 201]:
            return f"Success ({response.status_code}):\n{response.text}"
        else:
            return f"Server Error ({response.status_code}): {response.text}"
    except Exception as e:
        return f"HTTP Exception: {str(e)}"