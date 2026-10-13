import json
import os
from datasets import load_dataset

def scarica_gsm8k():
    # Impostiamo il salvataggio nella cartella corrente ("./") per evitare 
    # l'errore [Errno 2] No such file or directory.
    cartella_destinazione = "./"
    file_output = os.path.join(cartella_destinazione, "domande_rq3_gsm8k.json")
    
    print("⏳ Download del dataset GSM8K (OpenAI) in corso...")
    
    try:
        # Il dataset ufficiale è 'openai/gsm8k', configurazione 'main', split 'test'
        ds = load_dataset("openai/gsm8k", "main", split="test")
        
        # Selezioniamo i primi 100 problemi esatti come da tua documentazione
        sample = ds.select(range(100))
        
        domande_formattate = []
        for i, item in enumerate(sample):
            domande_formattate.append({
                "question_id": f"GSM8K_{i+1:03d}",
                "question_text": item.get("question", ""),
                # In GSM8K la risposta corretta include la spiegazione seguita da "#### numero"
                "correct_answer": item.get("answer", "")
            })
            
        with open(file_output, 'w', encoding='utf-8') as f:
            json.dump(domande_formattate, f, indent=4, ensure_ascii=False)
            
        print(f"✅ Finito! Salvate {len(domande_formattate)} domande per la RQ3.")
        print(f"📁 Percorso file salvato in: {os.path.abspath(file_output)}")
        
    except Exception as e:
        print(f"❌ Errore durante il download: {e}")
        print("Assicurati di essere connesso a internet e di aver eseguito:")
        print("pip install datasets")

if __name__ == "__main__":
    scarica_gsm8k()