"""
Valutatore RQ1 - Exact Match (EM).

Questo script legge il file di output generato dall'esperimento RQ1 (MedQA) 
e valuta l'accuratezza delle risposte del sistema confrontandole con quelle corrette.
Utilizza una metrica di Exact Match con normalizzazione del testo per ignorare 
differenze di punteggiatura e maiuscole/minuscole.
"""
import os
import json
import re
import string

# Modificare questa variabile per testare altre condizioni (es. "./RQ1_baseline.json")
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

    # 1. Controllo di uguaglianza assoluta
    norm_corretta = normalizza_testo(risposta_corretta)
    norm_agente = normalizza_testo(risposta_agente)
    if norm_corretta == norm_agente:
        return 1

    # 2. Controllo di inclusione (tolleranza):
    # Se il sistema ha aggiunto testo extra (es. "the answer is sickle cell disease"), 
    # la risposta è comunque considerata corretta se contiene l'esatta stringa risolutiva.
    if norm_corretta and norm_corretta in norm_agente:
        return 1
        
    return 0

def analizza_rq1():
    """
    Carica i dati dal file JSON specificato, itera su ogni domanda per calcolare 
    il punteggio e stampa un report finale con l'accuratezza percentuale.
    """

    print(" Valutazione RQ1 (MedQA - Exact Match) in corso...")
    if not os.path.exists(FILE_RQ1):
        print(f" File non trovato: {FILE_RQ1}")
        return

    with open(FILE_RQ1, 'r', encoding='utf-8') as f:
        data = json.load(f)

    totale_domande = len(data)
    risposte_corrette = 0

    # Ciclo di valutazione per ogni log di risposta
    for item in data:
        correct_ans = item.get("correct_answer", "")
        system_ans = item.get("system_answer", "")
        punteggio = calcola_exact_match(correct_ans, system_ans)
        risposte_corrette += punteggio

    # Calcolo percentuale gestendo il potenziale errore di divisione per zero
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