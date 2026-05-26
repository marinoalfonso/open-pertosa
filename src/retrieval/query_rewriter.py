import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Client locale al modulo — stesso pattern di retriever.py e vectorizer.py.
# Istanziato all'import, non a ogni chiamata: evita overhead di setup
# per ogni domanda del cittadino.
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Modello e parametri: gpt-4o-mini in coerenza con la generazione.
# Temperature 0 perché il rewriting non vuole creatività, vuole determinismo.
# max_tokens 80 cattura query realistiche (5-15 parole) ma blocca runaway.
_MODEL = "gpt-4o-mini"
_TEMPERATURE = 0.0
_MAX_TOKENS = 80

# Quanto della cronologia passare al rewriter.
# Solo l'ultimo turno (user + assistant) copre il 95% dei follow-up.
# Più contesto = più rischio che il rewriter agganci il referente sbagliato.
_HISTORY_TURNS = 1

# Tronchiamo la risposta assistant per non sprecare token: l'inizio
# della risposta contiene il topic, il resto è dettaglio che il rewriter non usa.
_ASSISTANT_TRUNCATE = 300

# Se l'output del rewriter è più lungo di questo, è sospetto:
# probabilmente ha "risposto" alla domanda invece di riformularla.
# Fallback alla domanda originale.
_MAX_REWRITTEN_LENGTH = 300


REWRITER_PROMPT = """Sei un assistente che riformula domande dei cittadini per migliorare la ricerca semantica in un archivio di documenti del Comune di Pertosa (delibere, determine, bilanci, regolamenti, ordinanze).

Il tuo unico compito è produrre una query di ricerca ottimizzata. Non rispondere alla domanda.

Regole:
- Se la domanda contiene riferimenti impliciti a quanto detto prima (es. "e per il 2018?", "e gli altri?", "invece quella nuova?"), risolvili usando la cronologia.
- Avvicina il linguaggio colloquiale al lessico amministrativo italiano (es. "stanno facendo lavori" → "interventi lavori pubblici"; "quanto si spende" → "spese"; "il sindaco" → "sindaco Comune Pertosa").
- Non inventare termini, nomi, date o numeri non presenti nella domanda o nella cronologia.
- Se la domanda è già autonoma e ben formulata, restituiscila invariata.
- Rispondi SOLO con la query riformulata, su una sola riga, senza preamboli, virgolette, spiegazioni o markdown.

Cronologia recente:
{history}

Domanda dell'utente: {question}

Query riformulata:"""


def _format_history(history: list) -> str:
    """
    Formatta gli ultimi N turni della cronologia per il rewriter.
    Trunca le risposte assistant per limitare i token.
    """
    if not history:
        return "(nessuna conversazione precedente)"

    # Un turno = 2 messaggi (user + assistant). Prendiamo gli ultimi.
    recent = history[-(_HISTORY_TURNS * 2):]

    lines = []
    for msg in recent:
        content = msg.content
        if msg.role == "assistant" and len(content) > _ASSISTANT_TRUNCATE:
            content = content[:_ASSISTANT_TRUNCATE].rstrip() + "..."
        lines.append(f"{msg.role}: {content}")

    return "\n".join(lines)


def rewrite_query(question: str, history: list) -> str:
    """
    Riformula la domanda dell'utente in una query ottimizzata per il retrieval.
    
    Risolve riferimenti deittici usando la cronologia (es. "e per il 2018?")
    e normalizza il linguaggio colloquiale verso il lessico amministrativo.
    
    In caso di errore o output sospetto restituisce la domanda originale
    (fail-open): il rewriter è un'ottimizzazione, non un componente critico.
    Meglio un retrieval mediocre che un errore 500 all'utente.
    """
    start = time.monotonic()

    prompt = REWRITER_PROMPT.format(
        history=_format_history(history),
        question=question
    )

    try:
        response = _client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=_TEMPERATURE,
            max_tokens=_MAX_TOKENS,
        )
        rewritten = response.choices[0].message.content.strip()

        # Safety net: output vuoto o anomalo → fallback alla domanda originale.
        # Casistiche tipiche: il modello "risponde" alla domanda invece di
        # riformularla, o aggiunge preamboli del tipo "Query: ...".
        if not rewritten or len(rewritten) > _MAX_REWRITTEN_LENGTH:
            print(f"[rewriter] output sospetto (len={len(rewritten)}), uso originale")
            return question

        # Rimuoviamo eventuali virgolette esterne che il modello a volte aggiunge
        # nonostante l'istruzione, perché degradano il retrieval semantico.
        rewritten = rewritten.strip('"\'')

        elapsed_ms = (time.monotonic() - start) * 1000
        print(f"[rewriter] original={question!r}")
        print(f"[rewriter] rewritten={rewritten!r} ({elapsed_ms:.0f}ms)")

        return rewritten

    except Exception as e:
        # Fail-open: log dell'errore e fallback. L'utente non deve mai
        # vedere un 500 perché il rewriter è giù.
        elapsed_ms = (time.monotonic() - start) * 1000
        print(f"[rewriter] errore dopo {elapsed_ms:.0f}ms: {type(e).__name__}: {e}")
        print(f"[rewriter] fallback alla domanda originale: {question!r}")
        return question