# Medical AI v2: Multi-Agent System per FHIR

Un sistema avanzato basato su architettura multi-agente (tramite CrewAI) progettato per interrogare, analizzare e interpretare dati medici standardizzati (server FHIR). Il progetto esplora le capacita dei Large Language Models (LLM) nel ragionamento clinico e logico applicando tecniche come ReAct, Chain-of-Thought (CoT), Self-Consistency e cicli di Reflexion (Verifier).

## Architettura degli Agenti

Il sistema utilizza i seguenti agenti specializzati:
* Medical Workflow Manager: Interpreta la richiesta dell'utente e delega i compiti usando il pattern ReAct, prevenendo allucinazioni sui dati.
* Query Processing Agent (QPA): Traduce il linguaggio naturale in percorsi di ricerca sintatticamente validi per server FHIR.
* FHIR Data Retrieval Agent (FDRA): Esegue materialmente le chiamate HTTP tramite tool integrati.
* Data Update Agent (DUA): Genera e carica risorse JSON FHIR valide.
* Data Interpretation Agent (DIA): Agisce come esperto clinico, traducendo i JSON tecnici in referti e diagnosi narrative.
* Verifier Agent: Valida l'output finale rispetto ai dati clinici. In caso di errore o basso accordo (ICA score < 85%), innesca un ciclo di raffinamento.

## Condizioni Sperimentali

Il sistema e testato su tre diverse Research Question (RQ1: MedQA, RQ2: Synthea FHIR, RQ3: GSM8K) modulando le seguenti condizioni sperimentali per misurare l'impatto delle singole tecniche:

Le condizioni sperimentali testate nel sistema sono quattro, ognuna progettata per misurare l'impatto di specifiche tecniche di ragionamento.

*Baseline: il sistema opera nella sua configurazione di base senza l'utilizzo di ReAct/Manager, Chain-of-Thought, Self-Consistency o Verifier.

*Ablation A: introduce l'uso di ReAct/Manager e del Chain-of-Thought, ma non utilizza la Self-Consistency o il Verifier.

*Ablation B: aggiunge la Self-Consistency alla configurazione di Ablation A, mantenendo attivi ReAct/Manager e Chain-of-Thought, ma omettendo ancora il Verifier.

*Full System: implementa tutte le tecniche disponibili contemporaneamente: ReAct/Manager, Chain-of-Thought, Self-Consistency e Verifier.

---

## Installazione

Installa le dipendenze necessarie eseguendo:

```bash
pip install -r requirements.txt
```

---

## Installazione e avvio di Ollama

1. Installa Ollama.

2. Avvia Ollama con il modello desiderato, ad esempio qwen3.5:27b:

```bash
ollama run qwen3.5:27b
```

---

## Avvio del progetto

Esegui il file principale:

```bash
python main.py
```

---

## Utilizzo

- Avvia il programma
- Inserisci una richiesta per il server FHIR

---

## Configurazione

L’URL del server FHIR è configurabile tramite il file .env; nello stesso file puoi anche scegliere il modello LLM da usare.