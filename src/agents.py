from crewai import Agent
import src.tools as tools

class MedicalAgents:
    """
    Classe che definisce e istanzia gli agenti AI specializzati (basati su CrewAI) 
    per l'elaborazione, il recupero e l'interpretazione di dati medici standardizzati (FHIR).
    Ogni agente ha un ruolo specifico all'interno della pipeline di ragionamento.
    """

    def __init__(self, llm, logger, current_config):
        """
        Inizializza la classe fornendo agli agenti il modello di linguaggio (LLM), 
        il sistema di logging e la configurazione corrente dell'esperimento.
        """

        self.llm = llm
        self.logger = logger
        self.current_config = current_config

    def catch_reasoning(self, log_attribute=None):
        """
        Funzione di callback che intercetta l'output di ogni step di ragionamento dell'agente.
        Se la configurazione prevede il Chain-of-Thought (cot), salva l'output nel logger.
        """

        def callback(step_output):
            if log_attribute:
                output = str(step_output.output) if step_output else "None"
                print(output)
                if self.current_config["cot"]:
                    self.logger.add_attribute(log_attribute, str(output))
        return callback

    def manager_router(self):
        """
        Crea e restituisce l'agente Manager.
        Applica il framework ReAct per valutare la domanda dell'utente, 
        selezionare l'agente più adatto per il compito e delegare l'azione senza inventare dati.
        """
        return Agent(
            role="Medical Workflow Manager",
            goal="Coordinate the agents to respond to the demand: {user_question}",
            backstory="""You are an experienced medical coordinator.
                Your task is to analyse the request, REASON (Thought) about which agent is most suitable, 
                TAKE ACTION (Act) by delegating the task, and evaluate the result.
                
                !!! CRITICAL MANDATORY RULES !!!
                1. NEVER SIMULATE OR INVENT THE OUTPUT OF OTHER AGENTS.
                2. NEVER INVENT FHIR JSON DATA OR CLINICAL RECORDS.
                3. When you delegate a task, YOU MUST STOP GENERATING TEXT and wait for the actual real response from the tool/agent.
                4. Do NOT write "Agent Output:", "Result:" or pretend to be the other agent. Let the actual agent do its job.
                
                The agents at your disposal are:
                1. Query Processing Agent: Translates natural language into valid FHIR search paths.
                2. FHIR Data Retrieval Agent: Executes read queries on the server.
                3. Data Update Agent: Creates valid FHIR JSON resources and executes upload.
                4. Data Interpretation Agent: Evaluates the real JSON and generates the clinical report.
                """,
            llm=self.llm,
            verbose=True,
            step_callback=self.catch_reasoning("react_ra")
        )

    def query_processing_agent(self):
        """
        Crea e restituisce lo specialista sintattico (QPA).
        Traduce la richiesta in linguaggio naturale in una stringa di interrogazione 
        API valida per i server FHIR, implementando strategie adattive se la ricerca è troppo restrittiva.
        """

        return Agent(
            role='Query Processing Agent',
            goal='Translate natural language into valid FHIR search paths.',
            backstory="""You are a FHIR syntax expert. 
                      CRITICAL RULES FOR FHIR SYNTAX:
                      - Use kebab-case for search parameters (e.g., clinical-status, verification-status).
                      - Use _sort=-onset-date for descending order.
                      
                      ADAPTIVE STRATEGY:
                      1. Start with the requested filters (e.g., clinical-status=active).
                      2. IF the user asks for a specific diagnosis (like 'Stress') AND your strict filter returns 0 results OR 
                         doesn't match the clinical context expected, RELAX the status filters to 'clinical-status=active,resolved' 
                         or remove the verification-status constraint to perform a broader search.
                      3. Your primary goal is to retrieve the record that contains the diagnosis, not to strictly filter it out.""",
            llm=self.llm,
            verbose=True,
            memory=False,
            step_callback=self.catch_reasoning("cot_qpa")
        )

    def fhir_data_retrieval_agent(self):
        """
        Crea e restituisce l'agente di recupero dati (FDRA).
        Utilizza i tool di sistema per eseguire materialmente le chiamate HTTP (GET) 
        al database FHIR, evitando severamente la simulazione dei file JSON.
        """

        return Agent(
            role='FHIR Data Retrieval Agent',
            goal='Execute read queries on the server using tools.',
            backstory="""You execute the GET call if the path is provided.
                      !!! CRITICAL RULE !!!
                      YOU MUST USE THE 'Execute FHIR Request' TOOL TO FETCH REAL DATA.
                      YOU MUST NEVER INVENT, GUESS, OR SIMULATE JSON DATA YOURSELF.
                      Once you use the tool, STOP writing and output ONLY the exact, raw text returned by the tool. 
                      If the tool fails or returns total: 0, output exactly that.""",
            tools=[tools.execute_fhir_query],
            llm=self.llm,
            verbose=True,
            memory=False,
            step_callback=self.catch_reasoning()
        )

    def data_update_agent(self):
        """
        Crea e restituisce l'agente per l'aggiornamento dati (DUA).
        Si occupa della generazione di risorse JSON conformi allo standard FHIR 
        e dell'esecuzione delle chiamate POST per caricare i dati sul server.
        """

        return Agent(
            role='Data Update Agent',
            goal='Create valid FHIR JSON resources and execute upload.',
            backstory="Responsible for data integrity. You create JSON and POST it.",
            tools=[tools.execute_fhir_query],
            llm=self.llm,
            verbose=True,
            memory=False,
            step_callback=self.catch_reasoning()
        )

    def data_interpretation_agent(self):
        """
        Crea e restituisce l'agente di interpretazione clinica (DIA).
        Agisce come un medico esperto: analizza i risultati JSON grezzi restituiti dal server 
        e genera un report sintetico e conclusivo sulla base delle evidenze estratte.
        """

        return Agent(
            role='Data Interpretation Agent / Medical Expert',
            goal='Translate technical JSON into a concise report, or answer general medical questions directly and briefly.',
            backstory="""You are a clinical expert. 
                      When answering, ALWAYS provide a brief 1-2 sentence explanation of YOUR reasoning using your own words, 
                      and then clearly state the final diagnosis in bold. Do not just output the diagnosis name.
                      If the provided JSON says 'total: 0', you MUST answer that the patient DOES NOT have the diagnosis.""",
            llm=self.llm,
            verbose=True,
            memory=False,
            step_callback=self.catch_reasoning("cot_dia")
        )

    def verifier_agent(self):
        """
        Crea e restituisce l'agente verificatore.
        Valuta se l'output del DIA è clinicamente corretto e aderente ESCLUSIVAMENTE 
        ai dati JSON ottenuti. Se la risposta non è corretta, fornisce feedback per la riformulazione.
        """
        
        return Agent(
            role='Verifier Agent',
            goal='Evaluate the quality of the final answer, providing feedback and suggestions for improvement.',
            backstory="""You are a Clinical Validation Assistant. Your goal is to confirm if the 
                      Data Interpretation Agent's conclusion is medically sound based ONLY on the provided JSON data. 
                      You focus on the 'What' (the correct diagnosis) rather than the 'How' (the format). 
                      If the agent provides the correct medical answer based on the clinical vignette or 
                      the FHIR data, you should approve it. 
                      Do not reject for lack of formatting, length, or minor redundancies. 
                      Your priority is medical correctness and agreement between the agents.""",
            llm=self.llm,
            verbose=True,
            memory=False,
            step_callback=self.catch_reasoning()
        )