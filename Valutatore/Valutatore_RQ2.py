import os
import json
import csv
import requests
import numpy as np
from scipy.stats import spearmanr

# ==========================================
# CONFIGURAZIONE OLLAMA E FILE
# ==========================================
MODELLO_OLLAMA = "gemma4:26b"
URL_OLLAMA = "http://localhost:11434/api/generate"
FILE_RQ2_FULL = "./RQ2_full.json"  

def valuta_con_llm_3_livelli(domanda, risposta_corretta, risposta_agente):
    """
    Usa l'LLM come Giudice applicando la rubrica a 3 livelli (0, 1, 2).
    """
    prompt_giudice = f"""Sei un valutatore clinico severo per un esperimento di Intelligenza Artificiale.
    Valuta la RISPOSTA DELL'AGENTE confrontandola con la RISPOSTA CORRETTA (Verità).
    
    DOMANDA: {domanda}
    RISPOSTA CORRETTA: {risposta_corretta}
    RISPOSTA DELL'AGENTE: {risposta_agente}
    
    Usa questa rubrica di valutazione a 3 livelli:
    0 = Risposta errata o mancante.
    1 = Risposta parzialmente corretta.
    2 = Risposta completa e corretta.
    
    Rispondi SOLO ed ESCLUSIVAMENTE con il numero (0, 1 oppure 2), senza testo aggiuntivo.
    """
    
    payload = {
        "model": MODELLO_OLLAMA,
        "prompt": prompt_giudice,
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }
    
    try:
        response = requests.post(URL_OLLAMA, json=payload)
        response.raise_for_status()
        risultato = response.json().get("response", "").strip()
        
        if "2" in risultato: return 2, "2 (Completa)"
        elif "1" in risultato: return 1, "1 (Parziale)"
        else: return 0, "0 (Errata)"
            
    except Exception as e:
        print(f"Errore di connessione a Ollama: {e}")
        return 0, "Errore Valutazione"

def analizza_rq2_full():
    if not os.path.exists(FILE_RQ2_FULL):
        print(f" File non trovato: {FILE_RQ2_FULL}")
        return

    with open(FILE_RQ2_FULL, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    dettagli_csv = []
    ica_scores = []
    correctness_scores = []
    
    for i, item in enumerate(data, 1):
        q_id = item.get("question_id", f"Q_{i}")
        q_text = item.get("question_text", "")
        correct_ans = str(item.get("correct_answer", ""))
        
        vera_risposta = str(item.get("system_answer", "")).strip()
        if "APPROVED" in vera_risposta:
            vera_risposta = vera_risposta.replace("APPROVED:", "").replace("APPROVED\n", "").replace("APPROVED", "").replace("**", "").strip()
        if not vera_risposta:
            vera_risposta = str(item.get("sc_majority_answer", "N/A"))
        
        raw_ica = item.get("sc_ica_score", 0)
        ica_val = float(raw_ica) if raw_ica is not None else 0.0
        if 0 < ica_val <= 1.0: 
            ica_val *= 100
            
        punteggio, giudizio_testo = valuta_con_llm_3_livelli(q_text, correct_ans, vera_risposta)
        
        ica_scores.append(ica_val)
        correctness_scores.append(punteggio)
            
        dettagli_csv.append({
            "ID Domanda": q_id,
            "Punteggio (0/1/2)": punteggio,
            "Punteggio ICA (%)": ica_val,
            "Giudizio LLM": giudizio_testo,
            "Risposta Corretta": correct_ans,
            "Risposta Agente": vera_risposta
        })

    arr_ica = np.array(ica_scores)
    arr_corr = np.array(correctness_scores)
    
    if len(arr_ica) > 1 and len(np.unique(arr_ica)) > 1 and len(np.unique(arr_corr)) > 1:
        corr, pvalue = spearmanr(arr_ica, arr_corr)
    else:
        corr, pvalue = 0.0, 1.0
    
    ica_corrette = arr_ica[arr_corr >= 1]
    ica_errate = arr_ica[arr_corr == 0]
    
    print(f"ICA medio (risposte corrette >=1)     | {np.mean(ica_corrette):.2f}%")
    print(f"ICA medio (risposte errate 0)         | {np.mean(ica_errate):.2f}%")
    print(f"Correlazione di Spearman ρ            | {corr:.3f}")
    print(f"p-value                               | {pvalue:.4e}")

if __name__ == "__main__":
    analizza_rq2_full()