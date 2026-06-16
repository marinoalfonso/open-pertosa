import os
import re
from datetime import date, timedelta
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Prefetch, FusionQuery, Fusion, SparseVector,
    Filter, FieldCondition, Range, MatchValue, DatetimeRange,
)
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

# ─────────────────────────────────────────────────────────────
# RICONOSCIMENTO INTENTI TEMPORALI NELLA QUERY
# ─────────────────────────────────────────────────────────────

# Quando applichiamo il filtro "recente", restringiamo agli ultimi N giorni.
# 180 giorni (~6 mesi) è un compromesso: cattura "ultimo trimestre",
# "ultimi mesi", "attualmente in corso" senza essere troppo stretto.
RECENT_DAYS_WINDOW = 180

KEYWORDS_RECENT = {
    "attualmente", "attuale", "adesso", "ora", "oggi",
    "recente", "recenti", "in corso", "ancora",
    "vigente", "vigenti", "aperto", "aperti", "aperta", "aperte",
}

KEYWORDS_MOST_RECENT = {
    "più recente", "piu recente", "ultima", "ultimo", "ultimi", "ultime",
}

_RE_ANNO_QUERY = re.compile(r"\b(20\d{2})\b")

TIPO_ATTO_KEYWORDS = {
    "delibera di giunta": "delibera_giunta",
    "delibere di giunta": "delibera_giunta",
    "delibera di consiglio": "delibera_consiglio",
    "delibere di consiglio": "delibera_consiglio",
    "consiglio comunale": "delibera_consiglio",
    "determina": "determina",
    "determine": "determina",
    "determinazione": "determina",
    "determinazioni": "determina",
    "ordinanza": "ordinanza",
    "ordinanze": "ordinanza",
    "decreto": "decreto",
    "decreti": "decreto",
    "bando": "bando_avviso",
    "bandi": "bando_avviso",
    "avviso": "bando_avviso",
    "avvisi": "bando_avviso",
}


def _detect_temporal_intent(query: str) -> dict:
    """Analizza la query e individua intenti temporali e di tipologia.
    Ritorna un dict con: recent, most_recent, year, tipo_atto."""
    query_lower = query.lower()

    intent = {
        "recent": any(kw in query_lower for kw in KEYWORDS_RECENT),
        "most_recent": any(kw in query_lower for kw in KEYWORDS_MOST_RECENT),
        "year": None,
        "tipo_atto": None,
    }

    m = _RE_ANNO_QUERY.search(query)
    if m:
        anno = int(m.group(1))
        if 2000 <= anno <= 2030:
            intent["year"] = anno

    # Cerco la chiave più lunga che matcha (per gestire "delibera di giunta"
    # prima di "delibera")
    sorted_keys = sorted(TIPO_ATTO_KEYWORDS.keys(), key=len, reverse=True)
    for kw in sorted_keys:
        if kw in query_lower:
            intent["tipo_atto"] = TIPO_ATTO_KEYWORDS[kw]
            break

    return intent


def _build_qdrant_filter(intent: dict) -> Filter | None:
    """Costruisce un filtro Qdrant a partire dall'intent temporale.
    Ritorna None se nessun filtro è applicabile."""
    must = []

    # 'recente' senza anno specifico → ultimi 180 giorni
    if intent["recent"] and intent["year"] is None:
        cutoff = (date.today() - timedelta(days=RECENT_DAYS_WINDOW)).isoformat()
        must.append(FieldCondition(
            key="data_atto",
            range=DatetimeRange(gte=cutoff)
        ))

    # Anno specifico
    if intent["year"] is not None:
        must.append(FieldCondition(
            key="anno",
            match=MatchValue(value=intent["year"])
        ))

    # Tipo atto menzionato esplicitamente
    if intent["tipo_atto"] is not None:
        must.append(FieldCondition(
            key="tipo_atto",
            match=MatchValue(value=intent["tipo_atto"])
        ))

    if not must:
        return None

    return Filter(must=must)

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
    Hybrid search (denso + sparso BM25) con fusione RRF, diversificazione
    per documento, e filtri temporali condizionali. Se la query contiene
    riferimenti temporali ("attualmente", "ultimo", "nel 2024", ecc.),
    applica filtri sui metadati prima della ricerca semantica.
    """

    # ── Analisi della query: intenti temporali e di tipologia ──
    intent = _detect_temporal_intent(query)
    query_filter = _build_qdrant_filter(intent)

    if query_filter or intent["most_recent"]:
        print(f"[retrieval] intent: {intent}")

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
    # Il filtro temporale viene applicato a ciascun Prefetch: Qdrant
    # restringe i candidati nei due rami prima della fusione RRF.
    response = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(
                query=dense_query,
                using=DENSE_VECTOR_NAME,
                limit=PREFETCH_LIMIT,
                filter=query_filter,  # ← nuovo
            ),
            Prefetch(
                query=sparse_query,
                using=SPARSE_VECTOR_NAME,
                limit=PREFETCH_LIMIT,
                filter=query_filter,  # ← nuovo
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=FETCH_BEFORE_DIVERSIFY,
        with_payload=True,
    )

    raw_points = response.points
    
    # Se il filtro azzera i risultati, ritenta senza filtro.
    if query_filter and not raw_points:
        print("[retrieval] filtro → 0 candidati, retry SENZA filtro")
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
        query_filter = None

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
            # ── nuovi campi temporali ──
            "data_atto": r.payload.get("data_atto"),
            "anno": r.payload.get("anno"),
            "tipo_atto": r.payload.get("tipo_atto"),
        })

    # ── Ordinamento per data se richiesto "il più recente" ──
    # Lo applichiamo DOPO la diversificazione, così operiamo su un set
    # già limitato e bilanciato. Chunk senza data finiscono in fondo.
    if intent["most_recent"]:
        chunks.sort(
            key=lambda c: c["data_atto"] or "0000-00-00",
            reverse=True
        )
        print(f"[retrieval] ordinamento per data (most_recent=True)")

    # ───── DEBUG RETRIEVAL ─────
    print(f"\n[retrieval] query={query!r}")
    n_unique_sources = len({c["source"] for c in chunks})
    diversify_tag = "ON" if DIVERSIFY_ENABLED else "OFF"
    filter_tag = "ON" if query_filter else "OFF"
    print(f"[retrieval] {len(chunks)} chunk recuperati (hybrid RRF, top_k={TOP_K}, "
          f"prefetch={PREFETCH_LIMIT}, fetch={FETCH_BEFORE_DIVERSIFY}, "
          f"diversify={diversify_tag} max/doc={MAX_CHUNKS_PER_DOC}, "
          f"filter={filter_tag}, "
          f"{n_unique_sources} documenti unici)")
    for i, c in enumerate(chunks):
        preview = c["text"][:80].replace("\n", " ")
        tag = "T" if c["type"] == "table" else "p"
        print(f"[retrieval] #{i:02d} score={c['score']:.4f} [{tag}] "
              f"data={c['data_atto']} tipo={c['tipo_atto']} "
              f"file={c['source']} pag={c['page']} | {preview}...")
    print("[retrieval] ─" * 20, flush=True)

    return chunks