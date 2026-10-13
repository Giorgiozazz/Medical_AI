import os
import json
from datasets import load_dataset

ds = load_dataset(
    "bigbio/med_qa", 
    name="med_qa_en_bigbio_qa", 
    split="test",
    token=os.getenv("HF_TOKEN"),
)

domande_rq1 = ds.shuffle(seed=42).select(range(200))

output_path = "./datasets/data/domande_rq1.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(list(domande_rq1), f, ensure_ascii=False, indent=4)

print(f"File salvato con successo in: {output_path}")