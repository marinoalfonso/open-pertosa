import os
from openai import OpenAI
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = "pertosa_docs"
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
TOP_K = 25  # numero di chunk da recuperare per ogni domanda

# Client istanziati una volta sola all'import del modulo, non a ogni
# richiesta: evita di riaprire connessioni a ogni domanda del cittadino.
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def retrieve(query: str) -> list[dict]:
    """
    Converte la domanda in vettore e cerca i chunk
    più semanticamente vicini in Qdrant.
    """
    
    # Vettorizziamo la domanda con lo stesso modello usato in ingestion
    response = openai_client.embeddings.create(
        input=[query],
        model=EMBEDDING_MODEL
    )
    query_vector = response.data[0].embedding

    # Ricerca semantica in Qdrant
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=TOP_K,
        with_payload=True  # vogliamo il testo e i metadati, non solo i vettori
    )

    # Restituiamo i chunk con il loro score di similarità
    chunks = []
    for r in results:
        chunks.append({
            "text": r.payload["text"],
            "source": r.payload["source"],
            "page": r.payload["page"],
            "score": round(r.score, 3)
        })
        
    # ───── DEBUG RETRIEVAL (temporaneo) ─────
    # Stampa posizione, score, file e prime parole di ogni chunk recuperato.
    # Da rimuovere dopo aver completato la diagnosi.
    print(f"\n[retrieval] query={query!r}")
    print(f"[retrieval] {len(chunks)} chunk recuperati su top_k={TOP_K}")
    for i, c in enumerate(chunks):
        preview = c["text"][:100].replace("\n", " ")
        print(f"[retrieval] #{i:02d} score={c['score']:.3f} "
              f"file={c['source']} pag={c['page']} | {preview}...")
    print("[retrieval] ─" * 20, flush=True)

    return chunks