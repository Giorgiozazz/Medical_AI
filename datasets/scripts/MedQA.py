"""
MedQA Dataset Downloader per RQ1.

Questo script utilizza la libreria `datasets` di Hugging Face per scaricare 
il dataset medico "MedQA". Estrae casualmente 200 domande dalla porzione di test 
e le salva in un file JSON locale per l'elaborazione della Research Question 1.
"""

import os
import json
from datasets import load_dataset

# Scarica il dataset MedQA (versione inglese 'bigbio_qa') da Hugging Face.
# Richiede un token di autenticazione configurato nelle variabili d'ambiente (.env).
ds = load_dataset(
    "bigbio/med_qa", 
    name="med_qa_en_bigbio_qa", 
    split="test",
    token=os.getenv("HF_TOKEN"),
)

# Mescola il dataset in modo casuale fissando il seed a 42 (per garantire 
# che vengano pescate sempre le stesse 200 domande ad ogni esecuzione) 
# e seleziona il sottoinsieme desiderato.
domande_rq1 = ds.shuffle(seed=42).select(range(200))

output_path = "./datasets/data/domande_rq1.json"

# Verifica che la cartella di destinazione esista, altrimenti la crea
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Esporta i dati estratti nel file JSON locale
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(list(domande_rq1), f, ensure_ascii=False, indent=4)

print(f"File salvato con successo in: {output_path}")