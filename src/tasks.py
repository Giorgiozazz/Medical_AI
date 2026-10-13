from crewai import Task

class MedicalTasks:

    """
    Classe dedicata alla definizione dei Task (compiti) assegnati agli agenti di CrewAI.
    Ogni metodo restituisce un oggetto Task configurato con una descrizione chiara, 
    l'output atteso e, se necessario, il contesto derivante dai task precedenti.
    """

    def manager_router_task(self, agent):
        """
        Task per l'agente orchestratore. 
        Definisce l'obiettivo di analizzare la richiesta dell'utente e coordinare 
        il flusso di lavoro decidendo a quali agenti delegare le operazioni.
        """

        return Task(
            name="Manager Router Task",
            description=(
                "As the Medical Workflow Manager, your task is to coordinate the efforts of the medical agents to respond to the user's question: '{user_question}'. "
                "Analyze the request, reason about which agent is most suitable for each part of the task, delegate accordingly, and evaluate the results before proceeding to the next step or providing the final response."
            ),
            agent=agent,
            expected_output="A well-coordinated workflow that effectively utilizes the capabilities of each agent to address the user's question comprehensively and accurately."
        )

    def query_processing_task(self, agent):
        """
        Task per lo specialista sintattico (QPA). 
        Analizza la domanda dell'utente e la traduce in una query API (FHIR search path) 
        o determina che non è necessario interrogare il server.
        """

        return Task(
            name="Query Processing Task",
            description=(
                "Analyze the user's question: '{user_question}'. "
                "If the question requires data from the server (e.g., blood pressure, patients), "
                "translate the natural language into a valid FHIR search path. "
                "If it's a general medical question, state that no FHIR query is needed."
            ),
            agent=agent,
            expected_output="A valid FHIR search path or a JSON string for POST/PUT. Or 'None' if not needed."
        )

    def fhir_data_retrieval_task(self, agent):
        """
        Task per il recupero dati (FDRA). 
        Prende in input la query generata dal task precedente e la esegue sul server,
        restituendo i dati grezzi in formato JSON.
        """

        return Task(
            name="FHIR Data Retrieval Task",
            description="Receive the FHIR path and use the tool to execute the GET query on the server.",
            agent=agent,
            expected_output="The JSON response from the FHIR server or an error message if the query fails. If query is not needed, return 'None'."
        )

    def data_update_task(self, agent):
        """
        Task opzionale per l'aggiornamento dei dati (DUA). 
        Si attiva solo se la richiesta dell'utente implica la creazione o l'aggiornamento 
        di una risorsa sul server (operazioni POST/PUT).
        """

        return Task(
            name="Data Update Task",
            description=(
                "If the user's request «{user_question}» specifies that data needs to be updated or created, "
                "create the JSON FHIR resource and upload it."
            ),
            agent=agent,
            expected_output="Server response after the update (POST/PUT) or 'None' if the request does not explicitly require an update"
        )

    def data_interpretation_task(self, agent):
        """
        Task di sintesi clinica (DIA). 
        Converte i dati JSON tecnici in una risposta narrativa o fornisce direttamente 
        una diagnosi medica basata sulle conoscenze generali se non ci sono dati FHIR.
        """

        return Task(
            name="Data Interpretation Task",
            description=(
                "Answer the user's question: '{user_question}'.\n"
                "If FHIR JSON data is provided by other agents, translate this technical data "
                "into a brief, human-readable narrative.\n"
                "If no FHIR data is available because the user asked a general medical question, "
                "provide a direct and concise medical answer based on your knowledge "
                "(e.g., 'The most likely diagnosis is **Posttraumatic stress disorder**.')."
                "Do not provide explanations or a line of reasoning, just the final answer in a clear and concise manner."
            ),
            agent=agent,
            expected_output=(
                "A natural language medical summary containing a brief reasoning path "
                "followed by the definitive answer or diagnosis in bold."
            ),
        )

    def ica_scoring_task(self, agent, context_tasks):
        """
        Task di valutazione. 
        Confronta il JSON originale con le interpretazioni mediche prodotte 
        e restituisce un punteggio di accuratezza in formato JSON strutturato.
        """

        return Task(
            name="ICA Scoring Task",
            description=(
                "Evaluate the correlation between the raw data extracted from the server (FHIR JSON) and the different medical interpretations provided. "
                "For each interpretation, calculate an ICA score (from 0 to 100) that represents the percentage of correlation and accuracy with respect to the original JSON data. "
                "Return the output EXACTLY in the following JSON format (no additional text, no extra markdown):\n"
                "{\n"
                '  "scores": [score_run1, score_run2, ...],\n'
                '  "best_response_index": index_of_the_best_response_starting_from_zero,\n'
                '  "best_score": maximum_score\n'
                "}\n\n"
                "User question that the interpretations respond to: '{user_question}'"
            ),
            agent=agent,
            context=context_tasks,
            expected_output="A JSON object containing the array 'scores', 'best_response_index', and 'best_score'."
        )

    def verification_task(self, agent, context_tasks=None):
        """
        Task di verifica medica. 
        Controlla se la diagnosi finale è clinicamente coerente con i dati.
        Il task tollera problemi di formattazione ma rigetta le inesattezze cliniche.
        """

        return Task(
            name="Verification Task",
            description=(
            "Review the final medical answer provided. "
            "Check if the diagnosis or conclusion aligns with the clinical facts presented in the question. "
            "If the answer is medically correct and addresses the user's request, mark it as APPROVED. "
            "Only reject (REJECTED) if there is a clear medical error or if the answer is completely unrelated. "
            "Be permissive with formatting: if the core information is accurate, the task is successful."
),
            agent=agent,
            context=context_tasks,
            expected_output="An approval (APPROVED) if the answer is medically sound, or a brief rejection (REJECTED) only in case of clear errors."
        )

    def reflexion_task(self, agent, context_tasks, original_question):
        """
        Task di raffinamento (Reflexion). 
        Si attiva solo se il 'Verification Task' ha rigettato la risposta precedente.
        L'agente analizza la critica ricevuta nel contesto e formula una risposta corretta.
        """
        return Task(
        name="Reflexion and Refinement Task",
        description=(
            f"Your previous answer to: '{original_question}' was REJECTED. "
            "Read the Verifier's critique in the context and fix the following:\n"
            "1. REMOVE all repetitions (don't print the same thing 3 times).\n"
            "2. EXPLAIN your reasoning (Chain-of-Thought).\n"
            "3. Be professional and concise."
        ),
        agent=agent,
        context=context_tasks, # Qui l'agente riceve la "bocciatura" del Verifier
        expected_output="A single, high-quality clinical summary with no redundancies."
    )