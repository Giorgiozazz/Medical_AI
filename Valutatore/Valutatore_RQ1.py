import os
import json
import re
import string

FILE_RQ1 = "./RQ1_ablA.json"

def normalizza_testo(testo):
    """
    Normalizza il testo per un confronto equo (Exact Match).
    - Converte tutto in minuscolo
    - Rimuove la punteggiatura
    - Rimuove gli spazi doppi o di formattazione
    """
    if not testo:
        return ""
    
    testo = str(testo).lower()
    testo = testo.translate(str.maketrans('', '', string.punctuation))
    
    testo = re.sub(r'\s+', ' ', testo).strip()
    return testo

def calcola_exact_match(risposta_corretta, risposta_agente):
    """
    Calcola l'Exact Match (EM).
    Ritorna 1 se corretto, 0 se errato.
    """
    norm_corretta = normalizza_testo(risposta_corretta)
    norm_agente = normalizza_testo(risposta_agente)
    if norm_corretta == norm_agente:
        return 1
    if norm_corretta and norm_corretta in norm_agente:
        return 1
        
    return 0

def analizza_rq1():
    print(" Valutazione RQ1 (MedQA - Exact Match) in corso...")
    if not os.path.exists(FILE_RQ1):
        print(f" File non trovato: {FILE_RQ1}")
        return

    with open(FILE_RQ1, 'r', encoding='utf-8') as f:
        data = json.load(f)

    totale_domande = len(data)
    risposte_corrette = 0
    
    for item in data:
        correct_ans = item.get("correct_answer", "")
        system_ans = item.get("system_answer", "")
        punteggio = calcola_exact_match(correct_ans, system_ans)
        risposte_corrette += punteggio
    accuratezza_em = (risposte_corrette / totale_domande) * 100 if totale_domande > 0 else 0

    print("\n" + "="*50)
    print(" REPORT FINALE RQ1 (Impatto Chain-of-Thought)")
    print("="*50)
    print(f"Dataset analizzato : MedQA")
    print(f"Condizione         : Ablation A (Solo CoT)")
    print(f"Domande totali     : {totale_domande}")
    print(f"Risposte corrette  : {risposte_corrette}")
    print("-" * 50)
    print(f" Accuratezza (EM%) : {accuratezza_em:.1f}%")
    print("="*50)

if __name__ == "__main__":
    analizza_rq1()