# Medical AI v2

Sistema multi-agente per l’interrogazione di server FHIR.

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