import os
import json
from datetime import datetime

class MedicalLogger:
    """
    Gestisce la registrazione (logging) standardizzata dei risultati, dei metadati 
    e dei passaggi logici (ragionamento, verifica) per la risoluzione delle domande mediche.
    """

    def __init__(self):
        """
        Inizializza il logger recuperando i parametri dell'esperimento dalle variabili 
        di ambiente e predispone la struttura principale del dizionario di log.
        """

        #Recupero dei metadati di base 
        self.research_question = os.getenv("RESEARCH_QUESTION")
        self.condition = os.getenv("CONDITION")
        self.dataset = os.getenv("DATASET")
        self.model_name = os.getenv("MODEL_NAME")
        self.k_max = os.getenv("K_MAX")

        now = datetime.now()

        # Struttura standard del log per tracciare l'intero ciclo di vita della domanda

        self.log = {
            "run_id": f"{self.research_question}_{self.condition}_{now.strftime('%Y%m%d_%H%M%S')}",
            "condition": self.condition,
            "dataset": self.dataset,
            "question_id": None,
            "question_text": None,
            "correct_answer": None,
            "system_answer": None,
            "cot_qpa": None,
            "cot_dia": None,
            "react_ra": None,
            "fhir_query": None,
            "sc_runs": None,
            "sc_ica_score": None, 
            "sc_majority_answer": None, 
            "verifier_rounds": [], 
            "n_refinement_cycles": 0, 
            "verifier_approved": None, 
            "low_agreement_flag": False, 
            "timestamp": now.isoformat(),
            "model_params": {
                "model": self.model_name,
                "temperature_base": 0,
                "n_sc_runs": 1,
                "k_max": self.k_max
            }
        }

    def add_verifier_round(self, cycle_number=None, dia_output=None, 
                           completeness=None, faithfulness=None, 
                           consistency=None, critique=None, approved=None):
        """
        Aggiunge i risultati di un singolo ciclo di verifica (verifier) allo storico del log.
        Utile per tracciare quante iterazioni fa il modello prima di approvare una risposta.
        """

        verifier_round = {
            "cycle_number": cycle_number,
            "dia_output": dia_output,
            "verifier_completeness": completeness,
            "verifier_faithfulness": faithfulness,
            "verifier_consistency": consistency,
            "verifier_critique": critique,
            "approved": approved
        }
        self.log["verifier_rounds"].append(verifier_round)
        return verifier_round

    def _clean_none_strings(self, data):
        """
        Funzione ricorsiva di supporto per convertire le stringhe "None" nel tipo di dato None.
        Previene errori di formattazione o di interpretazione nei log salvati.
        """

        if isinstance(data, dict):
            return {k: self._clean_none_strings(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._clean_none_strings(item) for item in data]
        elif data == "None":
            return None
        else:
            return data

    def save_log(self):
        """
        Pulisce i dati e salva il dizionario di log come file JSON formattato 
        all'interno della cartella temporanea 'temp'.
        """

        os.makedirs("temp", exist_ok=True)
        log_path = os.path.join("temp", f"{self.log['run_id']}.json")
        
        cleaned_log = self._clean_none_strings(self.log)
        
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(cleaned_log, f, indent=4, ensure_ascii=False)
            print(f"Log salvato con successo in: {log_path}")
        except Exception as e:
            print(f"Errore durante il salvataggio del log: {e}")
            
        return log_path

    def add_attribute(self, attribute, reasoning_text):
        """
        Aggiunge dinamicamente una stringa di testo (es. ragionamento) a un attributo specifico.
        Se l'attributo non esiste, lo inizializza come lista.
        """
        
        if attribute and reasoning_text is not None:
            if self.log.get(attribute) is None:
                self.log[attribute] = []
            self.log[attribute].append(reasoning_text)