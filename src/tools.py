import os
import json
import requests
from crewai.tools import tool

@tool("Execute FHIR Request")
def execute_fhir_query(path: str, method: str = "GET", body: str = None):
    """
    Esegue dinamicamente una richiesta HTTP (GET, POST, PUT, DELETE) verso il server FHIR.
    
    Parametri:
    - path: Il percorso o i filtri di ricerca da appendere all'URL base (es. 'Patient?name=Mario').
    - method: Il tipo di operazione HTTP da eseguire (default: "GET").
    - body: Il payload JSON (come stringa o dizionario) necessario per POST e PUT.
    
    Ritorna:
    Una stringa contenente l'esito della chiamata e i dati JSON formattati.
    """

    # Recupera l'indirizzo del server FHIR dalle variabili di ambiente
    base_url = os.getenv("FHIR_SERVER")
    if not base_url:
        return "Error: Missing 'FHIR_SERVER' variable."

    # Costruisce l'URL completo gestendo in automatico eventuali slash ('/') mancanti o di troppo
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    
    try:
        # Intestazione standard richiesta dai server FHIR
        headers = {"Content-Type": "application/fhir+json"}

        # Smistamento della richiesta in base al metodo HTTP specificato
        if method.upper() == "POST":
            if isinstance(body, dict): body = json.dumps(body)
            response = requests.post(url, headers=headers, data=body)
        elif method.upper() == "PUT":
            response = requests.put(url, headers=headers, data=body)
        elif method.upper() == "DELETE":
            response = requests.delete(url, headers=headers)
        else:
            # Operazione di lettura standard (usata dall'agente FDRA)
            response = requests.get(url, headers=headers)

        # Gestione delle risposte: 200 e 201
        if response.status_code in [200, 201]:
            return f"Success ({response.status_code}):\n{response.text}"
        else:
            return f"Server Error ({response.status_code}): {response.text}"
    except Exception as e:
        # Cattura errori di connessione (es. server offline) per evitare crash del programma
        return f"HTTP Exception: {str(e)}"