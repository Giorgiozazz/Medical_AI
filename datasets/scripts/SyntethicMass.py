import os
import random
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))

fhir_data_dir = "/home/tgrauso/projects/Medical-AI-v2/datasets/data/FHIR/fhir"
domande_base_path = "/home/tgrauso/projects/Medical-AI-v2/datasets/data/domande_rq2.json"
output_path = "/home/tgrauso/projects/Medical-AI-v2/datasets/data/pazienti_rq2.json"

if not os.path.exists(domande_base_path):
    print(f"ERRORE CRITICO: Non trovo il file delle domande qui: {domande_base_path}")
    print("Assicurati di aver creato il file con le 50 domande in quella posizione.")
    exit()
with open(domande_base_path, "r", encoding="utf-8") as f:
    lista_domande_guida = json.load(f)

tutti_i_pazienti = [f for f in os.listdir(fhir_data_dir) if f.endswith('.json')]

random.seed(42)

pazienti_selezionati = random.sample(tutti_i_pazienti, 50)

mapping_rq2 = []
for i in range(50):
    mapping_rq2.append({
        "paziente_file": pazienti_selezionati[i],
        "domanda_id": lista_domande_guida[i]["ID"],
        "domanda_testo": lista_domande_guida[i]["Domanda"]
    })

os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(mapping_rq2, f, ensure_ascii=False, indent=4)

print(f"Selezionati 50 pazienti e salvato il mapping in: {output_path}")