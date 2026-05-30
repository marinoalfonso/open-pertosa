import os
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector
from fastembed import SparseTextEmbedding
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = "pertosa_docs"
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Numero di chunk finali restituiti al modello, dopo fusione RRF e
# diversificazione.
TOP_K = 15

# Nomi dei vettori nella collezione: devono coincidere con quelli usati
# in ingestion (vectorizer.py).
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"

# Pre-fetch per ciascun ramo (denso e sparso). Più alto rispetto a prima
# (era 30) perché la diversificazione ha bisogno di un pool più ampio:
# se più chunk dello stesso documento vengono tagliati dal limite, servono
# candidati di riserva da altri documenti per riempire i TOP_K.
PREFETCH_LIMIT = 50

# Quanti risultati post-fusione raccogliere PRIMA della diversificazione.
# La fusione RRF di Qdrant produce un ranking unificato dei candidati dei
# due rami; ne prendiamo molti (60) e poi filtriamo. Se chiedessimo solo
# TOP_K=15 a Qdrant, non avremmo materiale di riserva per sostituire i
# chunk eliminati dal limite per documento.
FETCH_BEFORE_DIVERSIFY = 60

# Diversificazione: massimo N chunk dello stesso documento nei risultati
# finali. Spezza il monopolio dei documenti lunghi (DUPS, PIAO, elenchi
# residui) garantendo che il TOP_K rappresenti più documenti diversi.
# Valore 2 = "intestazione + dettaglio" se servono, ma niente monopolio.
MAX_CHUNKS_PER_DOC = 2

# Flag operativo: se False, salta la diversificazione e si comporta come
# l'hybrid puro. Utile per confronti A/B e diagnosi.
DIVERSIFY_ENABLED = True

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

_bm25_model = None


def _get_bm25():
    global _bm25_model
    if _bm25_model is None:
        _bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25", language="italian")
    return _bm25_model


def _diversify_by_source(points, max_per_doc: int, top_k: int):
    """
    Diversificazione stile MMR sull'asse SOURCE: scorre i candidati in
    ordine di score (già ordinati da RRF) e tiene al massimo `max_per_doc`
    chunk per ciascun documento, fermandosi a `top_k` totali.

    Non valuta similarità tra i chunk: è un MMR "leggero" basato solo sul
    documento di provenienza. Semplice, prevedibile, sufficiente per il
    problema del monopolio dei documenti programmatici lunghi.

    Policy esplicita: se i candidati diversificati sono meno di top_k,
    si restituisce MENO di top_k chunk. NON si ripescano i chunk eliminati
    dal limite, perché vanificherebbe la diversificazione stessa (il caso
    tipico è un corpus con pochi documenti dominanti: meglio 10 chunk
    diversificati che 15 monopolizzati).
    """
    kept = []
    counts = {}

    for p in points:
        source = p.payload.get("source", "")
        if counts.get(source, 0) < max_per_doc:
            kept.append(p)
            counts[source] = counts.get(source, 0) + 1
            if len(kept) >= top_k:
                break

    return kept


def retrieve(query: str) -> list[dict]:
    """
    Hybrid search (denso + sparso BM25) con fusione RRF e diversificazione
    per documento. Restituisce TOP_K chunk al massimo, da almeno
    TOP_K/MAX_CHUNKS_PER_DOC documenti diversi (in condizioni normali).
    """
    # ── Vettorizzazione della query in due forme ──
    dense_resp = openai_client.embeddings.create(input=[query], model=EMBEDDING_MODEL)
    dense_query = dense_resp.data[0].embedding

    bm25 = _get_bm25()
    sparse = next(iter(bm25.embed([query])))
    sparse_query = SparseVector(
        indices=sparse.indices.tolist(),
        values=sparse.values.tolist(),
    )

    # ── Hybrid query con fusione RRF lato server ──
    # Recuperiamo FETCH_BEFORE_DIVERSIFY risultati post-fusione (non solo
    # TOP_K), per avere materiale di riserva quando la diversificazione
    # taglia i chunk in eccesso dello stesso documento.
    response = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=dense_query, using=DENSE_VECTOR_NAME, limit=PREFETCH_LIMIT),
            Prefetch(query=sparse_query, using=SPARSE_VECTOR_NAME, limit=PREFETCH_LIMIT),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=FETCH_BEFORE_DIVERSIFY,
        with_payload=True,
    )

    raw_points = response.points

    # ── Diversificazione per documento ──
    if DIVERSIFY_ENABLED:
        final_points = _diversify_by_source(
            raw_points,
            max_per_doc=MAX_CHUNKS_PER_DOC,
            top_k=TOP_K,
        )
    else:
        final_points = raw_points[:TOP_K]

    chunks = []
    for r in final_points:
        chunks.append({
            "text": r.payload["text"],
            "source": r.payload["source"],
            "page": r.payload["page"],
            "type": r.payload.get("type", "paragraph"),
            "score": round(r.score, 4),
        })

    # ───── DEBUG RETRIEVAL (temporaneo) ─────
    # Mostra anche le statistiche di diversificazione: quanti documenti
    # diversi nei risultati finali.
    print(f"\n[retrieval] query={query!r}")
    n_unique_sources = len({c["source"] for c in chunks})
    diversify_tag = "ON" if DIVERSIFY_ENABLED else "OFF"
    print(f"[retrieval] {len(chunks)} chunk recuperati (hybrid RRF, top_k={TOP_K}, "
          f"prefetch={PREFETCH_LIMIT}, fetch={FETCH_BEFORE_DIVERSIFY}, "
          f"diversify={diversify_tag} max/doc={MAX_CHUNKS_PER_DOC}, "
          f"{n_unique_sources} documenti unici)")
    for i, c in enumerate(chunks):
        preview = c["text"][:100].replace("\n", " ")
        tag = "T" if c["type"] == "table" else "p"
        print(f"[retrieval] #{i:02d} score={c['score']:.4f} [{tag}] "
              f"file={c['source']} pag={c['page']} | {preview}...")
    print("[retrieval] ─" * 20, flush=True)

    return chunks