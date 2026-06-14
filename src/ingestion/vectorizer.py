import os
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    SparseVectorParams, SparseVector, Modifier,
)
from fastembed import SparseTextEmbedding
from dotenv import load_dotenv
import uuid

load_dotenv()

EMBEDDING_DIM = 1536  # dimensione di text-embedding-3-small
COLLECTION_NAME = "pertosa_docs"

# Nomi dei due vettori della collezione. Con la Query API di Qdrant ogni
# punto porta DUE rappresentazioni: il vettore denso (semantico, OpenAI) e
# il vettore sparso (lessicale, BM25). I nomi servono a riferirsi a ciascuno
# in fase di ricerca.
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"

# Upsert in batch: Qdrant rifiuta payload singoli oltre 32MB.
UPSERT_BATCH_SIZE = 100

# Modello BM25 sparso, in locale via FastEmbed. language="italian" attiva
# lo stemming e le stopword italiane: senza, l'italiano verrebbe trattato
# come inglese e lo stemming sbaglierebbe (es. "comunale"/"comune" non
# verrebbero ricondotti correttamente).
# Istanziato pigro: il modello si carica al primo uso e resta in memoria.
_bm25_model = None


def _get_bm25():
    global _bm25_model
    if _bm25_model is None:
        _bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25", language="italian")
    return _bm25_model


def get_openai_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_qdrant_client():
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    return QdrantClient(host=host, port=port)


def create_collection_if_not_exists(qdrant: QdrantClient):
    """
    Crea la collezione hybrid (denso + sparso) se non esiste.

    ATTENZIONE: la struttura dei vettori si definisce alla creazione e non
    è modificabile dopo. Una collezione creata con la vecchia struttura
    (solo denso, senza vettori nominati) NON è compatibile con l'hybrid e
    va eliminata prima. Vedi recreate_collection() per la re-indicizzazione.
    """
    existing = [c.name for c in qdrant.get_collections().collections]

    if COLLECTION_NAME not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(
                    # IDF: Qdrant pesa i termini per rarità lato server.
                    # È ciò che rende "Pertosa" (presente ovunque) quasi
                    # irrilevante e fa emergere i termini distintivi.
                    modifier=Modifier.IDF,
                )
            },
        )
        print(f"Collection '{COLLECTION_NAME}' creata (hybrid: denso + bm25)")
    else:
        print(f"Collection '{COLLECTION_NAME}' già esistente")


def recreate_collection(qdrant: QdrantClient):
    """
    Elimina e ricrea la collezione con la struttura hybrid.

    Da usare per la PRIMA ingestion hybrid, quando esiste già una collezione
    con la vecchia struttura (solo denso). Cancella tutti i dati esistenti:
    va lanciata consapevolmente.
    """
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME in existing:
        qdrant.delete_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' eliminata (ricreazione hybrid)")
    create_collection_if_not_exists(qdrant)


def embed_chunks(chunks: list[dict], batch_size: int = 20) -> list[dict]:
    """
    Calcola per ogni chunk DUE vettori:
      - denso: embedding semantico OpenAI (come prima)
      - sparso: BM25 locale via FastEmbed (italiano)
    """
    client = get_openai_client()
    model = os.getenv("OPENAI_EMBEDDING_MODEL")
    bm25 = _get_bm25()

    embedded = []
    total = len(chunks)

    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c.get("text_contextualized", c["text"]) for c in batch]

        print(f"  Embedding batch {i//batch_size + 1}/{-(-total//batch_size)}")

        # Denso (OpenAI, una chiamata per batch)
        response = client.embeddings.create(input=texts, model=model)
        dense_vectors = [d.embedding for d in response.data]

        # Sparso BM25 (locale, nessuna chiamata di rete)
        sparse_vectors = list(bm25.embed(texts))

        for chunk, dense, sparse in zip(batch, dense_vectors, sparse_vectors):
            embedded.append({
                **chunk,
                "dense": dense,
                "sparse_indices": sparse.indices.tolist(),
                "sparse_values": sparse.values.tolist(),
            })

    return embedded


def save_to_qdrant(embedded_chunks: list[dict], qdrant: QdrantClient):
    points = []
    for chunk in embedded_chunks:
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector={
                DENSE_VECTOR_NAME: chunk["dense"],
                SPARSE_VECTOR_NAME: SparseVector(
                    indices=chunk["sparse_indices"],
                    values=chunk["sparse_values"],
                ),
            },
            payload={
                "text": chunk["text"],
                "context": chunk.get("context", ""),
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"],
                "tipo_atto": chunk.get("tipo_atto", "altro"),
                "data_atto": chunk.get("data_atto"),
                "anno": chunk.get("anno"),
                "data_precisione": chunk.get("data_precisione", "ignoto"),
            },
        ))

    total = len(points)
    for i in range(0, total, UPSERT_BATCH_SIZE):
        batch = points[i:i + UPSERT_BATCH_SIZE]
        qdrant.upsert(collection_name=COLLECTION_NAME, points=batch)

    print(f"  Salvati {total} punti in Qdrant ({-(-total//UPSERT_BATCH_SIZE)} batch)")