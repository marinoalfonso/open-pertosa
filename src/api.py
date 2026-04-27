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

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) # Inizializziamo il client OpenAI una volta sola

def extract_financials_llm(text: str) -> dict:
    """
    Estrae dati finanziari da testo usando LLM.
    Usata dal parser (NON dagli endpoint API).
    """

    text = text[:8000]  # limite sicurezza

    prompt = f"""
Estrai tutte le voci finanziarie dal testo.

Restituisci SOLO JSON valido nel formato:
{{ "voce": numero }}

Regole:
- Numeri in formato float (es: 120000.0)
- Converti numeri italiani (120.000,00 → 120000.0)
- Normalizza quando possibile (imu, tari, irpef, entrate, spese)
- Mantieni anche voci non standard se presenti
- NON aggiungere testo fuori dal JSON

TESTO:
{text}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": "Esperto di bilanci pubblici italiani"},
                {"role": "user", "content": prompt}
            ]
        )

        content = response.choices[0].message.content.strip()

        # pulizia markdown
        content = content.replace("```json", "").replace("```", "").strip()

        data = json.loads(content)

        # validazione
        cleaned = {}
        for k, v in data.items():
            try:
                cleaned[k] = float(v)
            except:
                continue

        return cleaned

    except Exception as e:
        print(f"⚠️ LLM financial extraction error: {e}")
        return {}

sys.path.append(str(Path(__file__).parent / "retrieval"))
sys.path.append(str(Path(__file__).parent / "generation"))

from retriever import retrieve

app = FastAPI(
    title="Assistente Comune di Pertosa",
    description="Sistema RAG per la consultazione dei documenti comunali",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"]
)

SYSTEM_PROMPT = """Sei un assistente del Comune di Pertosa.
Rispondi alle domande dei cittadini basandoti sui documenti ufficiali del comune.

Regole:
- Rispondi sempre in italiano
- Per domande fattuali (cifre, date, nomi, delibere), basati esclusivamente 
  sui documenti forniti e cita sempre la fonte
- Per domande di chiarimento o follow-up su qualcosa già detto nella 
  conversazione, puoi usare la cronologia del dialogo per rispondere
- Se l'informazione non è presente nei documenti e non è stata menzionata 
  nella conversazione, rispondi:
  "Non ho trovato informazioni sufficienti nei documenti disponibili."
- Non inventare mai cifre o informazioni non presenti nel contesto
- Cita sempre le fonti alla fine della risposta, a capo, in questo formato esatto:
  (Fonte: nome_del_file.pdf, pagina X), con "Fonte" in corsivo mentre "nome_del_file.pdf" e "pagina X" in grassetto.
  Se usi più fonti: (Fonti: nome_file1.pdf p.X, nome_file2.pdf p.Y)
  Non usare mai "Fonte 1", "Fonte 2" o riferimenti numerici anonimi.
- Formatta le risposte in modo chiaro usando elenchi e grassetto quando utile
- Sii preciso, conciso e completo nelle risposte, evitando ambiguità o mancanze di informazione"""


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