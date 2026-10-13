"""
Main Module - Orchestrazione Esperimenti Medici e Logico-Matematici.

Questo script funge da punto di ingresso (entry point) per l'intero sistema.
In base alle variabili di ambiente (.env), carica la condizione sperimentale desiderata 
(es. baseline, full) e avvia l'elaborazione dei dataset specifici per le tre 
Research Question (RQ1, RQ2, RQ3), gestendo dinamicamente l'attivazione degli agenti 
CrewAI e il salvataggio dei log.
"""
import src.logger as logger
import src.agents as agents
import src.tools as tools
import src.tasks as tasks
from crewai import Crew, Process, LLM
import os
import json 
import time
import re
import requests
from dotenv import load_dotenv
load_dotenv()

# Mappa delle condizioni sperimentali che determinano l'attivazione di specifiche funzionalità
# (Chain-of-Thought, Self-Consistency, Verifier) e i parametri del modello.
CONDITIONS_MAP = {
    "baseline": {"agent_mode": "multi", "cot": False, "sc": False, "verifier": False, "temperature": 0.0, "n_sc_runs": 1, "k_max": 5},
    "ablA": {"agent_mode": "multi", "cot": True,  "sc": False, "verifier": False, "temperature": 0.0, "n_sc_runs": 1, "k_max": 5},
    "ablB": {"agent_mode": "multi", "cot": True,  "sc": True,  "verifier": False, "temperature": 0.8, "n_sc_runs": 3, "k_max": 5},
    "full": {"agent_mode": "multi", "cot": True,  "sc": True,  "verifier": True,  "temperature": 1.0, "n_sc_runs": 3, "k_max": 5}
}

def elaborate(question_id: str = None, user_input: str = None, correct_answer: str = None):
    """
    Funzione core per l'elaborazione di una singola domanda (medica o generale).
    Instanzia il modello LLM, inizializza gli agenti e i task necessari in base alla 
    condizione sperimentale, esegue la CrewAI e struttura i risultati nel file di log.
    """

    current_config = CONDITIONS_MAP.get(os.getenv("CONDITION"))

    MODEL_NAME = os.getenv("MODEL_NAME")
    API_SERVER= os.getenv("API_SERVER")
    BASE_URL = os.getenv("BASE_URL")
    CONDITION = os.getenv("CONDITION")

    TEMPERATURE = current_config["temperature"]
    N_SC_RUNS = current_config["n_sc_runs"]
    K_MAX = current_config["k_max"]

    # Configurazione globale del modello di linguaggio
    llm_model = LLM(
        model=MODEL_NAME,
        base_url=API_SERVER,
        temperature=TEMPERATURE,
        timeout=300,
        top_p=K_MAX
    )

    log = logger.MedicalLogger()

    # Registrazione metadati iniziali
    if question_id:
        log.log["question_id"] = question_id
    log.log["question_text"] = user_input
    if correct_answer:
        log.log["correct_answer"] = correct_answer

    log.log["model_params"]["temperature_base"] = TEMPERATURE
    log.log["model_params"]["n_sc_runs"] = N_SC_RUNS
    log.log["model_params"]["k_max"] = K_MAX

    # Inizializzazione degli agenti e dei task base
    med_agents = agents.MedicalAgents(llm=llm_model, logger=log, current_config=current_config)
    med_tasks = tasks.MedicalTasks()

    query_processor = med_agents.query_processing_agent()
    retriever = med_agents.fhir_data_retrieval_agent()
    data_updater = med_agents.data_update_agent()
    interpreter = med_agents.data_interpretation_agent()

    task_query = med_tasks.query_processing_task(query_processor)
    task_retrieval = med_tasks.fhir_data_retrieval_task(retriever)
    task_update = med_tasks.data_update_task(data_updater)
    
    task_interpretation = med_tasks.data_interpretation_task(interpreter)

    base_agents = [query_processor, retriever, data_updater, interpreter]
    base_tasks = [task_query, task_retrieval, task_update]

    manager = None
    manager_llm_model = None
    sc_tasks = []
    ica_task = None

    print(f"Running in {CONDITION} condition")

    # Composizione dinamica della Crew in base alla condizione sperimentale
    match CONDITION:
        case "baseline":
            # Nessun coordinatore avanzato, esecuzione standard
            base_tasks.append(task_interpretation)
            manager_llm_model = llm_model

        case "ablA":
            # Aggiunta del Manager (ReAct) e attivazione del Chain-of-Thought

            manager = med_agents.manager_router()
            base_tasks.insert(0, med_tasks.manager_router_task(manager))
            base_tasks.append(task_interpretation)

        case "ablB":
            # Aggiunta del Manager, Chain-of-Thought e Self-Consistency

            manager = med_agents.manager_router()
            base_agents.insert(0, manager) 
            sc_tasks = [med_tasks.data_interpretation_task(interpreter) for _ in range(N_SC_RUNS)]
            ica_task = med_tasks.ica_scoring_task(interpreter, context_tasks=[task_retrieval] + sc_tasks)
            base_tasks = [med_tasks.manager_router_task(manager)] + base_tasks + sc_tasks + [ica_task]

        case "full":
            # Condizione completa: ReAct + CoT + Self-Consistency + Verifier

            manager = med_agents.manager_router()
            verifier = med_agents.verifier_agent()
            base_agents.extend([manager, verifier])
            trigger_reflexion_by_ica = False
            sc_tasks = [med_tasks.data_interpretation_task(interpreter) for _ in range(N_SC_RUNS)]
            ica_task = med_tasks.ica_scoring_task(interpreter, context_tasks=[task_retrieval] + sc_tasks)
            verification_task = med_tasks.verification_task(verifier, context_tasks=[ica_task] + sc_tasks)

            base_tasks = [
                med_tasks.manager_router_task(manager),
                *base_tasks,
                *sc_tasks,
                ica_task,
                verification_task
            ]

    # Avvio della sequenza di agenti
    crew = Crew(
        agents=base_agents,
        tasks=base_tasks,
        process=Process.sequential,
        manager_agent=manager,
        manager_llm=manager_llm_model,
        verbose=True,
        cache=False
    )

    result = crew.kickoff(inputs={"user_question": user_input})
    result_str = str(result)
    
    log.log["system_answer"] = result_str
    

    def extract_json(text):
        """Helper per estrarre blocchi JSON grezzi dall'output formattato in markdown."""

        match = re.search(r'```(?:json)?(.*?)```', text, re.DOTALL)
        if match: return match.group(1).strip()
        match_braces = re.search(r'\{.*\}', text, re.DOTALL)
        if match_braces: return match_braces.group(0).strip()
        return text.strip()
    
    # Valutazione della Self-Consistency e dell' ICA
    if sc_tasks and current_config["sc"]:
        responses = [task.output.raw for task in sc_tasks if hasattr(task, 'output') and task.output]
        log.log["sc_runs"] = [{"response": r, "score": None} for r in responses]

        ica_raw_output = ica_task.output.raw if (ica_task and hasattr(ica_task, 'output') and ica_task.output) else result_str

        try:
            clean_ica_output = extract_json(ica_raw_output)
            ica_result = json.loads(clean_ica_output)
            
            scores = ica_result.get("scores", [])
            best_idx = ica_result.get("best_response_index", 0)
            best_score = ica_result.get("best_score", None)

            # Se l'accordo è troppo basso (< 85%), si flagga la necessità di riflessione o revisione
            if best_score is not None and best_score < 0.85:
                trigger_reflexion_by_ica = True
                log.log["low_agreement_flag"] = True  
                print("Avviso: ICA sotto la soglia dell'85%. Attivazione Verifier e Human-in-the-Loop.")
            
            detailed_runs = []
            for i, resp in enumerate(responses):
                s = scores[i] if isinstance(scores, list) and i < len(scores) else None
                detailed_runs.append({"response": resp, "score": s})

            log.log["sc_runs"] = detailed_runs
            log.log["sc_ica_score"] = best_score

            # Salvataggio della risposta ritenuta migliore tra le corse multiple
            if isinstance(best_idx, int) and 0 <= best_idx < len(responses):
                best_response_text = responses[best_idx]
                log.log["sc_majority_answer"] = best_response_text
                log.log["system_answer"] = best_response_text
                
        except Exception as e:
            print(f"Errore nel parsing del JSON di ICA Task: {e}")
            print(f"Testo analizzato che ha causato l'errore: {ica_raw_output}")

    # Gestione del ciclo di verifica (Verifier) ed eventuale raffinamento (Reflexion) per la condizione 'full'
    if CONDITION == "full" and 'verification_task' in locals() and verification_task and verification_task.output:
        v_output = verification_task.output.raw
        v_output_upper = v_output.upper()

        if "REJECTED" in v_output_upper or "FAIL" in v_output_upper:
            is_approved = False
        else:
            is_approved = any(word in v_output.lower() for word in ["approved", "sufficient", "correct", "valid"])
        log.add_verifier_round(
            cycle_number=1,
            dia_output=log.log.get("sc_majority_answer") or log.log.get("system_answer"),
            critique=v_output,
            approved=is_approved
        )

        # Se l'ICA è basso, forza un ciclo extra di ragionamento
        if trigger_reflexion_by_ica:
            
            ref_task = med_tasks.reflexion_task(
                agent=interpreter, 
                context_tasks=[verification_task], 
                original_question=user_input
            )

            ref_crew = Crew(agents=[interpreter], tasks=[ref_task], verbose=True, cache=False)
            refined_result = str(ref_crew.kickoff())

            log.log["system_answer"] = refined_result
            log.log["n_refinement_cycles"] = 1

            log.log["verifier_approved"] = True
        else:
            log.log["verifier_approved"] = is_approved

    # Registrazione query finale ed esportazione su file
    log.log["fhir_query"] = task_query.output.raw if hasattr(task_query, 'output') and task_query.output else None

    log.save_log()
    print("Log salvato con successo.")
    return log.log

def research_question_1():
    """
    Esegue l'esperimento relativo alla Research Question 1 (RQ1).
    Legge un set di domande a risposta multipla da un JSON locale e invia ogni query 
    al sistema per la valutazione della capacità di risposta di base del modello.
    """

    with open("datasets/data/domande_rq1.json", 'r', encoding='utf-8') as file:
        data = json.load(file)
        
    for item in data:
        question_id = item.get("question_id")
        
        question_text = item.get("question", "")
        choices_list = item.get("choices", [])
        
        choices_formatted = "\n".join([f"- {choice}" for choice in choices_list])
        user_input = f"{question_text}\n\ Choices: \n{choices_formatted}"
        
        answers_list = item.get("answer", [])
        correct_answer = answers_list[0] if answers_list else None

        elaborate(
            question_id=question_id,
            user_input=user_input,
            correct_answer=correct_answer
        )

    print("Tutte le domande di RQ1 sono state elaborate.")

def research_question_2():
    """
    Esegue l'esperimento relativo alla Research Question 2 (RQ2).
    Inietta nei prompt del modello dati clinici sintetici (FHIR JSON) letti 
    da file locali per testare la capacità di interpretazione contestuale.
    """

    cartella_pazienti_locali = "./datasets/data/FHIR/fhir/"
    with open("datasets/data/pazienti_rq2.json", 'r', encoding='utf-8') as file:
        data = json.load(file)

    risultati_finali = []

    for item in data:
        question_id = item.get("domanda_id")
        question_text = item.get("domanda_testo", "")
        paziente_file = item.get("paziente_file", "")
        risposta_corretta = item.get("correct_answer", "")
        percorso_completo_json = os.path.join(cartella_pazienti_locali, paziente_file)

        try:
            with open(percorso_completo_json, 'r', encoding='utf-8') as f:
                dati_paziente = json.load(f)
        
            prompt_completo = f"""
            Domanda clinica: {question_text}
            
            Dati clinici del paziente (Formato FHIR):
            {json.dumps(dati_paziente, indent=2)[:4000]} 
            (Nota: Analizza questi dati per rispondere alla domanda).
            """
            
            risultato_domanda = elaborate(
                question_id=question_id,
                user_input=prompt_completo,
                correct_answer=risposta_corretta
            )

            if risultato_domanda:
                risultati_finali.append(risultato_domanda)
                
        except Exception as e:
            print(f"[{question_id}] Errore lettura file: {e}")

    condizione = os.getenv("CONDITION", "baseline")

    # Salva il dump complessivo dei risultati per la RQ2
    with open(f"RQ2_{condizione}.json", 'w', encoding='utf-8') as f:
        json.dump(risultati_finali, f, indent=4, ensure_ascii=False)

def research_question_3():
    """
    Esegue l'esperimento relativo alla Research Question 3 (RQ3).
    Questo modulo devia dal dominio clinico per testare l'architettura su problemi 
    matematici e logici puri (GSM8K dataset), verificando se i workflow (es. full condition) 
    migliorano le performance a prescindere dal contesto medico.
    """

    import os
    import json
    from crewai import Agent, Task, Crew, Process, LLM
    
    modello_scelto = os.getenv("MODEL", "ollama/gemma4:26b")
    llm_rq3 = LLM(model=modello_scelto, base_url="http://localhost:11434")
    
    agente_logico = Agent(
        role='Mathematics and Logic Expert',
        goal='Solve complex step-by-step math problems correctly.',
        backstory='You are an expert mathematician. You solve problems directly and provide the final exact number clearly at the end of your reasoning.',
        llm=llm_rq3,
        verbose=True,
        max_iter=3  
    )
    agente_verificatore = Agent(
        role='Math Validation Expert',
        goal='Validate math steps. Approve unless there is a clear calculation error.',
        backstory='''You are a precise math validator. 
        1. If the final answer is correct and the reasoning is sound, output ONLY: 'APPROVED'.
        2. If there is a calculation error, politely explain it and provide the correct result.
        3. Do NOT reject for style or minor verbosity. Trust the agent unless the math is wrong.''',
        llm=llm_rq3,
        verbose=True,
        memory=False
    )

    with open("datasets/data/domande_gsm8k.json", 'r', encoding='utf-8') as file:
        data = json.load(file)

    risultati_finali = []
    condizione_attuale = os.getenv("CONDITION", "baseline")
    
    for item in data:
        question_id = item.get("question_id")
        question_text = item.get("question_text", "")
        correct_answer = item.get("correct_answer", "")
        
        print(f"\n[{question_id}] Elaborazione in condizione: {condizione_attuale}...")
        
        risoluzione_task = Task(
            description=f"Solve this math problem step by step: {question_text}",
            expected_output="A step-by-step calculation ending with the final exact answer.",
            agent=agente_logico
        )
        
        agenti_attivi = [agente_logico]
        task_attivi = [risoluzione_task]

        # Nella condizione 'full', si innesca il doppio controllo matematico
        if condizione_attuale == "full":
            verifica_task = Task(
                description=f"Verify the mathematical correctness of the previous answer against the original problem: {question_text}. Correct the answer if the math is wrong.",
                expected_output="Verification feedback and the definitive corrected math answer.",
                agent=agente_verificatore
            )
            agenti_attivi.append(agente_verificatore)
            task_attivi.append(verifica_task)
       
        mini_crew = Crew(
            agents=agenti_attivi,
            tasks=task_attivi,
            process=Process.sequential,
            verbose=True
        )
        
        try:
            risultato_crew = mini_crew.kickoff()
            if condizione_attuale == "full":

               # Estrazione condizionale della risposta in base all'esito del Verificatore
                ultimo_task = risultato_crew.tasks_output[-1].raw
                if "APPROVED" in ultimo_task.upper():
                    
                    testo_risposta = risultato_crew.tasks_output[-2].raw
                else:
                    
                    testo_risposta = ultimo_task
            else:
                testo_risposta = risultato_crew.tasks_output[0].raw
        except Exception as e:
            testo_risposta = f"Errore durante l'esecuzione: {str(e)}"
            print(f" Errore su {question_id}, passo alla successiva.")
        
        log_rq3 = {
            "question_id": question_id,
            "question_text": question_text,
            "correct_answer": correct_answer,
            "system_answer": testo_risposta,
            "condition": condizione_attuale,
            "n_refinement_cycles": 1 if condizione_attuale == "full" else 0
        }
        risultati_finali.append(log_rq3)
        
       
        nome_file_output = f"RQ3_{condizione_attuale}.json"
        with open(nome_file_output, 'w', encoding='utf-8') as f_out:
            json.dump(risultati_finali, f_out, indent=4, ensure_ascii=False)

    print(f"\n Esecuzione RQ3 completata per la condizione {condizione_attuale}!")

# Entry point dello script
if __name__ == "__main__":
    # Avvia l'esperimento modificando la funzione chiamata qui sotto a seconda delle necessità.
    research_question_2()