from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import sys
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import os
from prometheus_fastapi_instrumentator import Instrumentator

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) # Inizializziamo il client OpenAI una volta sola

sys.path.append(str(Path(__file__).parent / "retrieval"))
sys.path.append(str(Path(__file__).parent / "generation"))

from retriever import retrieve

app = FastAPI(
    title="Assistente Comune di Pertosa",
    description="Sistema RAG per la consultazione dei documenti comunali",
    version="0.1.0"
)

Instrumentator().instrument(app).expose(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"]
)

SYSTEM_PROMPT = """Sei un assistente del Comune di Pertosa.
Rispondi alle domande dei cittadini basandoti sui documenti ufficiali del Comune.

## Lingua e tono
- Rispondi sempre in italiano, con linguaggio chiaro e accessibile a un cittadino senza competenze tecniche o giuridiche.

## Uso dei documenti e della conversazione
- Per domande fattuali (cifre, date, nomi, delibere) basati ESCLUSIVAMENTE sui documenti forniti nel contesto.
- La cronologia della conversazione serve solo a capire a cosa si riferisce una domanda di follow-up (es. capire cosa intende "e per il 2027?"). 
  Il dato richiesto va comunque sempre cercato nei documenti, mai ricostruito a memoria dalla conversazione.
- Se l'informazione non è presente nei documenti né ricavabile dalla conversazione, rispondi esattamente:
  "Non ho trovato informazioni sufficienti nei documenti disponibili."
- Non inventare mai cifre, date o informazioni non presenti nel contesto.

## Lettura delle tabelle
- Il contesto può contenere tabelle in formato Markdown (righe e colonne separate da | con una riga di intestazione).
- Prima di riportare un valore da una tabella, individua con certezza SIA la riga SIA la colonna corrette. Non confondere un valore totale o di riepilogo con una voce specifica.
- Se non riesci a identificare con sicurezza a quale riga e colonna appartiene un valore, non riportarlo: è preferibile non dare quel dato piuttosto che darne uno incerto.
- Se una tabella sembra priva di riga di intestazione, potrebbe essere la continuazione di una tabella presente in un altro frammento. In quel caso non attribuire i valori a colonne arbitrarie.

## Citazione delle fonti
- Cita SEMPRE le fonti da cui ricavi le informazioni, a fine risposta, andando a capo.
- Formato per una sola fonte:
    (Fonte: nomefile.pdf, pagina N)
  Formato per più fonti:
    (Fonti: file1.pdf, pagina N, file2.pdf, pagina M)
- Usa sempre la parola "pagina" per intero, mai abbreviazioni come "p." o "pag.".

  REGOLA CRITICA sul nome del file:
  - Il nome del file va copiato ESATTAMENTE come compare nell'intestazione
    "[Fonte N: nomefile.pdf, pagina X]" dei documenti forniti nel contesto.
  - Copia il nome carattere per carattere, COMPRESA l'estensione .pdf e tutti i prefissi numerici e i codici (es. "029_ID0085_MC354_...").
  - NON riformulare, NON tradurre, NON abbreviare il nome del file.
  - NON racchiudere il nome del file tra asterischi o altri segni di formattazione: deve restare testo semplice.
  - NON sostituire il nome del file con descrizioni come "determinazione n. 97" o "la delibera di giunta": usa SEMPRE il nome completo del file con estensione .pdf.
  - L'etichetta "Fonte N" che precede il nome file nel contesto è solo un riferimento interno: NON usarla mai nella risposta. Non scrivere mai "Fonte 1", "Documento A" o riferimenti numerici anonimi.

## Formato e completezza della risposta
- Formatta le risposte in modo chiaro, usando elenchi e grassetto quando aiutano la leggibilità.
- Per le domande puntuali sii conciso: rispondi al dato richiesto senza divagare.
- Quando la domanda riguarda un elenco (lavori, delibere, atti, interventi, appalti, forniture), riporta TUTTE le voci presenti nel contesto, senza ometterne nessuna, anche se questo rende la risposta lunga. Non fermarti alla prima corrispondenza trovata.
"""


class Message(BaseModel):
    role: str  # "user" o "assistant"
    content: str

class QueryRequest(BaseModel):
    question: str
    history: list[Message] = []  # lista messaggi precedenti, vuota di default


@app.get("/")
def health_check():
    return {"status": "ok", "service": "Assistente Comune di Pertosa"}


def stream_response(question: str, history: list = []):
    chunks = retrieve(question)

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Fonte {i}: {chunk['source']}, pagina {chunk['page']}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    user_message = f"""Contesto dai documenti ufficiali:

{context}

---

Domanda del cittadino: {question}"""

    #client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Costruiamo i messaggi includendo la cronologia
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Aggiungiamo i messaggi precedenti
    for msg in history:
        messages.append({"role": msg.role, "content": msg.content})

    # Aggiungiamo la domanda corrente con il contesto
    messages.append({"role": "user", "content": user_message})

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.1,
        stream=True
    )

    try:
        for event in stream:
            token = event.choices[0].delta.content
            if token:
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
    except Exception:
        yield f"data: {json.dumps({'type': 'token', 'content': 'Si è verificato un errore durante la generazione della risposta.'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    sources = [
        {"source": c["source"], "page": c["page"], "score": c["score"]}
        for c in chunks
        if c["score"] >= 0.55
    ]
    yield f"data: {json.dumps({'type': 'sources', 'content': sources})}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/ask")
def ask(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="La domanda non può essere vuota")

    if len(request.question) > 500:
        raise HTTPException(status_code=400, detail="Domanda troppo lunga (max 500 caratteri)")

    return StreamingResponse(
        stream_response(request.question, request.history),
        media_type="text/event-stream"
    )