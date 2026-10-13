"""
Synthea/SyntheticMass Data Mapper per RQ2.

Questo script legge una lista di domande mediche predefinite (domande_rq2.json)
e seleziona casualmente 50 file paziente dal dataset FHIR locale. 
Infine, mappa ogni paziente a una specifica domanda per creare il dataset di test finale (pazienti_rq2.json) 
che verrà utilizzato dagli agenti AI per la Research Question 2.
"""
import os
import random
import json

# Percorsi assoluti predefiniti per l'accesso ai dati e il salvataggio dell'output
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))

fhir_data_dir = "/home/projects/Medical-AI-v2/datasets/data/FHIR/fhir"
domande_base_path = "/home/projects/Medical-AI-v2/datasets/data/domande_rq2.json"
output_path = "/home/projects/Medical-AI-v2/datasets/data/pazienti_rq2.json"

# Controllo di integrità: verifica che il file delle domande esista prima di procedere
if not os.path.exists(domande_base_path):
    print(f"ERRORE CRITICO: Non trovo il file delle domande qui: {domande_base_path}")
    exit()
with open(domande_base_path, "r", encoding="utf-8") as f:
    lista_domande_guida = json.load(f)

# Recupera tutti i file con estensione .json presenti nella directory FHIR
tutti_i_pazienti = [f for f in os.listdir(fhir_data_dir) if f.endswith('.json')]

# Fissa il seed randomico a 42. Questo garantisce la riproducibilità dell'esperimento: 
# eseguendo lo script più volte, verranno estratti sempre gli stessi 50 pazienti.
random.seed(42)

pazienti_selezionati = random.sample(tutti_i_pazienti, 50)

# Ciclo di mapping: associa l'i-esimo paziente selezionato all'i-esima domanda della lista
mapping_rq2 = []
for i in range(50):
    mapping_rq2.append({
        "paziente_file": pazienti_selezionati[i],
        "domanda_id": lista_domande_guida[i]["ID"],
        "domanda_testo": lista_domande_guida[i]["Domanda"]
    })

# Crea le cartelle di destinazione se non esistono e salva il file di output finale
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(mapping_rq2, f, ensure_ascii=False, indent=4)

print(f"Selezionati 50 pazienti e salvato il mapping in: {output_path}")