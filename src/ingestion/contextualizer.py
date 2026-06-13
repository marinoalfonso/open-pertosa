"""
Contextualizer — arricchisce ogni chunk con un breve contesto generato
dal LLM prima della vettorizzazione.

Si innesta tra il chunker e il vectorizer senza modificarne i confini:
riceve i chunk prodotti dal chunker e restituisce gli stessi chunk con
due campi in più:
  - "context"             → 2-3 frasi che situano il chunk nel documento
  - "text_contextualized" → context + "\n\n" + testo originale

Il campo "text" originale resta INVARIATO: è quello che verrà mostrato
all'LLM in fase di risposta e usato per la citazione del PDF. Solo
"text_contextualized" viene vettorizzato, così il contesto migliora il
retrieval senza inquinare la citazione delle fonti.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Allineato al resto della pipeline. In produzione puntare allo stesso
# endpoint Azure OpenAI EU usato per embedding e generazione (vedi nota
# GDPR a fine messaggio).
CONTEXT_MODEL = os.getenv("OPENAI_CONTEXT_MODEL", "gpt-4o-mini")
MAX_WORKERS = 2          # chiamate concorrenti; basso per non saturare i rate limit
HEADER_MAX_WORDS = 300   # parole di intestazione documento passate al modello

CONTEXT_PROMPT = """Sei un assistente che analizza documenti amministrativi \
del Comune di Pertosa (Salerno).

Ti vengono forniti: l'intestazione del documento sorgente e un frammento \
(chunk) estratto da quel documento. Scrivi 2-3 frasi brevi che situano il \
frammento nel suo documento, in modo che sia ritrovabile da una ricerca \
semantica.

Le frasi devono indicare:
1. di quale documento fa parte (tipo di atto, oggetto, anno se deducibili);
2. di cosa tratta specificamente il frammento;
3. le informazioni chiave che contiene (cifre, date, soggetti, codici).

Non ripetere il testo del frammento. Non aggiungere introduzioni o commenti. \
Rispondi solo con le 2-3 frasi di contesto, in italiano.

--- INTESTAZIONE DEL DOCUMENTO ---
{header}

--- FRAMMENTO DA CONTESTUALIZZARE ---
{chunk}"""


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words])


def _contextualize_one(chunk: dict, client: OpenAI) -> dict:
    """Genera il contesto per un singolo chunk. In caso di errore restituisce
    il chunk con contesto vuoto: il frammento resta comunque vettorizzabile
    col solo testo originale (degradazione controllata)."""
    header = _truncate_words(chunk.get("document_header", ""), HEADER_MAX_WORDS)

    prompt = CONTEXT_PROMPT.format(
        header=header or "(intestazione non disponibile)",
        chunk=chunk["text"]
    )

    try:
        for attempt in range(5):
            try:
                response = client.chat.completions.create(
                    model=CONTEXT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0
                )
                context = response.choices[0].message.content.strip()
                break
            except Exception as e:
                if "429" in str(e) and attempt < 4:
                    wait = 2 ** attempt  # 1s, 2s, 4s, 8s
                    time.sleep(wait)
                else:
                    raise
    except Exception as e:
        print(f"  [contextualizer] errore su {chunk['source']} "
              f"pag.{chunk['page']} chunk#{chunk['chunk_index']}: {e}")
        context = ""

    text_ctx = f"{context}\n\n{chunk['text']}" if context else chunk["text"]

    return {**chunk, "context": context, "text_contextualized": text_ctx}


def contextualize_chunks(chunks: list[dict],
                         client: OpenAI = None,
                         max_workers: int = MAX_WORKERS) -> list[dict]:
    """Contestualizza tutti i chunk in parallelo (thread pool).
    L'ordine di output rispecchia l'ordine di input."""
    if not chunks:
        return chunks

    if client is None:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    total = len(chunks)
    print(f"  Contestualizzazione di {total} chunk "
          f"(modello={CONTEXT_MODEL}, workers={max_workers})...")

    results = [None] * total
    done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_contextualize_one, chunk, client): i
            for i, chunk in enumerate(chunks)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            done += 1
            if done % 50 == 0 or done == total:
                print(f"  [contextualizer] {done}/{total} completati")

    empty = sum(1 for r in results if not r["context"])
    print(f"  Contestualizzazione completata in {time.time() - t0:.0f}s "
          f"({empty} chunk senza contesto per errori/header mancante)")

    return results