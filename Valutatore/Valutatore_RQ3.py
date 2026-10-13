import os
import json
import csv
import re

FILE_ABL_B = "./RQ3_ablB.json"
FILE_FULL = "./RQ3_full.json"

def estrai_risposta_numerica(testo):
    """Estrae il numero finale da una stringa (formato standard GSM8K o fallback)"""
    match = re.search(r'####\s*(\d+)', testo)
    if match:
        return match.group(1)
    numeri = re.findall(r'\d+', testo)
    return numeri[-1] if numeri else None

def pulisci_risposta(item):
    """Recupera la risposta corretta gestendo il segnale APPROVED del Verificatore."""
    system_ans = str(item.get("system_answer", ""))
    
    if "APPROVED" in system_ans.upper():
        if "verifier_rounds" in item and isinstance(item["verifier_rounds"], list) and len(item["verifier_rounds"]) > 0:
            ultimo_round = item["verifier_rounds"][-1]
            return str(ultimo_round.get("dia_output", system_ans))
        return system_ans.replace("APPROVED", "").strip()
        
    return system_ans

def valuta_gsm8k(domanda, risposta_corretta, risposta_agente):
    if "####" in risposta_corretta:
        target = risposta_corretta.split("####")[-1].strip()
    else:
        target = risposta_corretta.strip()
        
    risposta_sistema = estrai_risposta_numerica(risposta_agente)
    
    if risposta_sistema == target:
        return 1, "CORRETTO"
    else:
        return 0, f"ERRATO (Target: {target}, Estratto: {risposta_sistema})"

def valuta_file(filepath, nome_condizione):
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    punteggi = []
    cicli_totali = []
    
    for item in data:
        correct_ans = str(item.get("correct_answer", ""))
        vera_risposta = pulisci_risposta(item)
        
        punteggio, _ = valuta_gsm8k(item.get("question_text", ""), correct_ans, vera_risposta)
        punteggi.append(punteggio)
        cicli_totali.append(item.get("n_refinement_cycles", 0))

    accuracy = (sum(punteggi) / len(punteggi)) * 100 if punteggi else 0
    media_cicli = (sum(cicli_totali) / len(cicli_totali)) if cicli_totali else 0

    return {"accuracy": accuracy, "media_cicli": media_cicli}

def stampa_report_finale():
    res_ablB = valuta_file(FILE_ABL_B, "Ablation B (No Verifier)")
    res_full = valuta_file(FILE_FULL, "Full System (Con Verifier)")
    
    if res_ablB and res_full:
        print(f"{'Condizione':<30} | {'Accuracy':<10} | {'Media Cicli'}")
        print(f"{'Ablation B':<30} | {res_ablB['accuracy']:>9.2f}% | {'N/A'}")
        print(f"{'Full System':<30} | {res_full['accuracy']:>9.2f}% | {res_full['media_cicli']:.2f}")

if __name__ == "__main__":
    stampa_report_finale()