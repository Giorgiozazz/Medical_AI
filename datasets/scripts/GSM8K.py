"""
GSM8K Dataset Downloader per RQ3.

Questo script scarica il dataset "GSM8K" di OpenAI tramite la libreria Hugging Face. Estrae i primi 
100 problemi dal set di test e li formatta in un file JSON strutturato 
per essere processato dagli agenti matematici nella Research Question 3.
"""
import json
import os
from datasets import load_dataset

def scarica_gsm8k():
    """
    Esegue il download, la selezione e la formattazione dei dati GSM8K,
    salvandoli in un file JSON locale pronto per l'uso.
    """
    cartella_destinazione = "./"
    file_output = os.path.join(cartella_destinazione, "domande_rq3_gsm8k.json")
    
    print(" Download del dataset GSM8K (OpenAI) in corso...")
    
    try:
        # Scarica la configurazione principale ('main') della porzione di test del dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        
        # Selezioniamo i primi 100 problemi esatti 
        sample = ds.select(range(100))

        # Riformatta i dati per renderli compatibili con lo standard di logging del progetto
        domande_formattate = []
        for i, item in enumerate(sample):
            domande_formattate.append({
                "question_id": f"GSM8K_{i+1:03d}",
                "question_text": item.get("question", ""),
                "correct_answer": item.get("answer", "")
            })

        # Esporta la lista di dizionari formattata nel file JSON di destinazione
        with open(file_output, 'w', encoding='utf-8') as f:
            json.dump(domande_formattate, f, indent=4, ensure_ascii=False)
            
        print(f" Finito! Salvate {len(domande_formattate)} domande per la RQ3.")
        print(f" Percorso file salvato in: {os.path.abspath(file_output)}")
        
    except Exception as e:
        # Cattura eventuali errori di connessione API o di permessi di scrittura locale
        print(f" Errore durante il download: {e}")

if __name__ == "__main__":
    scarica_gsm8k()