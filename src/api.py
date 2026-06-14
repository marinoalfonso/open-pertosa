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
- Per qualsiasi informazione specifica del Comune di Pertosa (cifre, date, nomi, delibere, regolamenti, decisioni amministrative), basati esclusivamente sui documenti forniti nel contesto.
- Per domande di follow-up, puoi usare la cronologia del dialogo.
- Se l'informazione non è presente nei documenti né ricavabile dalla conversazione, rispondi esattamente:
  "Non ho trovato informazioni sufficienti nei documenti disponibili."

## Lettura delle tabelle
- Il contesto può contenere tabelle in formato Markdown (righe e colonne separate da | con una riga di intestazione).
- Prima di riportare un valore da una tabella, individua con certezza SIA la riga SIA la colonna corrette. Non confondere un valore totale o di riepilogo con una voce specifica.
- Se non riesci a identificare con sicurezza a quale riga e colonna appartiene un valore, non riportarlo: è preferibile non dare quel dato piuttosto che darne uno incerto.

## Regole temporali
- I documenti forniti potrebbero essere stati selezionati con filtri 
  temporali (es. ultimi 6 mesi, anno specifico): basati su quanto vedi
  in essi, considerando la data odierna fornita più in basso.
- Quando la domanda contiene "attualmente", "in corso", "ultimo", "recente":
  * cita esplicitamente la data dell'atto a cui ti riferisci
  * se nei chunk vedi più atti con date diverse, scegli quello con data
    più vicina ad oggi
  * se la domanda riguarda lo STATO di un lavoro o procedimento (es. "in
    corso", "ancora aperto", "concluso") e i documenti non ti permettono
    di confermare lo stato corrente, dichiaralo onestamente: indica la
    data dell'atto più recente di cui hai notizia, e suggerisci al
    cittadino di consultare gli uffici comunali per informazioni
    aggiornate.
- Quando la domanda specifica un anno (es. "nel 2024"), considera solo i
  documenti di quell'anno presenti nel contesto.

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
- Adatta la lunghezza alla domanda: risposta secco per domande chiuse, risposta articolata per domande aperte.
- Quando la domanda chiede un elenco, riporta tutte le voci presenti nel contesto fornito. Non sintetizzare con "tra cui" o "ad esempio": aggrega esplicitamente.
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

import re

# Parole troppo comuni che non aggiungono significato
STOPWORDS_IT = {
    "il", "la", "lo", "i", "gli", "le", "un", "una", "uno",
    "di", "a", "da", "in", "con", "su", "per", "tra", "fra",
    "e", "o", "ma", "se", "che", "chi", "cosa", "come", "dove",
    "quando", "quanto", "quale", "qual", "è", "sono", "ha", "ho"
}

# Indicatori che la query È GIÀ specifica al Comune o a un'entità chiara
SPECIFIC_MARKERS = {
    "pertosa", "comune", "comunale",
    "tari", "imu", "tasi", "cosap", "tosap",  # tributi specifici
    "delibera", "determina", "ordinanza",      # tipi atto
    "bilancio", "pec", "stipendio"             # ricerche tecniche specifiche
}
# Nota: "pec" e "stipendio" NON sono marker di specificità,
# ma di intent tecnico. Li tratto a parte.


def _count_significant_words(query: str) -> int:
    """Conta parole non-stopword nella query."""
    words = re.findall(r"\b\w+\b", query.lower())
    return sum(1 for w in words if w not in STOPWORDS_IT)


def _is_query_specific(query: str) -> bool:
    words_lower = set(re.findall(r"\b\w+\b", query.lower()))
    has_comune_marker = bool(words_lower & {"pertosa", "comune", "comunale"})
    has_tax_marker = bool(words_lower & {"tari", "imu", "tasi", "cosap", "tosap"})
    return has_comune_marker or has_tax_marker


def needs_rewriting(query: str, history: list) -> bool:
    """Decide se la query è candidata all'espansione."""
    sig_words = _count_significant_words(query)
    
    # Criterio principale: query breve E non specifica al Comune
    if sig_words < 5 and not _is_query_specific(query):
        return True
    
    # Criterio aggiuntivo: query con pronomi vaghi e nessuna history
    pronouns = {"quanto", "quando", "chi", "cosa", "dove", "qual"}
    words_lower = set(re.findall(r"\b\w+\b", query.lower()))
    if (sig_words < 4 
        and words_lower & pronouns 
        and not history 
        and not _is_query_specific(query)):
        return True
    
    return False

REWRITE_PROMPT = """Riformula questa domanda di un cittadino al Comune di Pertosa, rendendola esplicita.

REGOLE:
- Aggiungi "del Comune di Pertosa" solo se manca completamente un riferimento all'ente.
- NON aggiungere informazioni che il cittadino non ha menzionato.
- Mantieni la domanda concisa.

Domanda originale: {query}

Rispondi SOLO con la domanda riformulata, niente altro."""


def expand_query(query: str, client: OpenAI) -> str:
    """Espande la query con una chiamata LLM. Fallback alla query originale
    in caso di errore."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": REWRITE_PROMPT.format(query=query)}],
            max_tokens=80,
            temperature=0
        )
        expanded = response.choices[0].message.content.strip()
        # Sanity check: l'output non deve essere assurdamente lungo
        if len(expanded) > 300 or not expanded:
            return query
        return expanded
    except Exception as e:
        print(f"[rewriter] errore: {e}")
        return query

def stream_response(question: str, history: list = []):
    # ───── QUERY REWRITING CONDIZIONALE ─────
    # Le query brevi e generiche (es. "qual è la PEC?") non hanno
    # abbastanza segnali per un retrieval efficace. Le espandiamo
    # esplicitando il contesto del Comune. Le query già specifiche
    # passano intatte.
    effective_query = question
    if needs_rewriting(question, history):
        effective_query = expand_query(question, client)
        print(f"[rewriter] '{question}' → '{effective_query}'")

    # Retrieval con la query espansa (se applicabile)
    chunks = retrieve(effective_query)

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        # Costruzione header del chunk con metadati temporali quando disponibili
        header_parts = [f"Fonte {i}: {chunk['source']}, pagina {chunk['page']}"]
        if chunk.get("data_atto"):
            header_parts.append(f"data atto: {chunk['data_atto']}")
        elif chunk.get("anno"):
            header_parts.append(f"anno: {chunk['anno']}")
        if chunk.get("tipo_atto") and chunk["tipo_atto"] != "altro":
            header_parts.append(f"tipo: {chunk['tipo_atto']}")
        header = " | ".join(header_parts)

        context_parts.append(f"[{header}]\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_parts)

    # Il modello vede la domanda ORIGINALE del cittadino, non quella
    # riformulata: così la risposta resta in linea col tono di chi ha
    # chiesto.
    user_message = f"""Contesto dai documenti ufficiali:

{context}

---

Domanda del cittadino: {question}"""

    # ───── INIEZIONE DATA ODIERNA NEL SYSTEM PROMPT ─────
    # Permette al modello di interpretare correttamente espressioni
    # come "attualmente", "ultimo mese", "recente".
    from datetime import date
    oggi = date.today().strftime("%d/%m/%Y")
    system_with_date = SYSTEM_PROMPT + f"\n\nData odierna: {oggi}."

    # Costruzione messaggi per OpenAI
    messages = [{"role": "system", "content": system_with_date}]

    # Cronologia conversazione
    for msg in history:
        messages.append({"role": msg.role, "content": msg.content})

    # Domanda corrente con contesto recuperato
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