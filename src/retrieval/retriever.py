import os
from openai import OpenAI
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = "pertosa_docs"
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
TOP_K = 10  # numero di chunk da recuperare per ogni domanda

#openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#qdrant_client = QdrantClient(host="localhost", port=6333)

def get_clients():
    openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    qdrant = QdrantClient(host="localhost", port=6333)
    return openai, qdrant


def retrieve(query: str) -> list[dict]:
    """
    Converte la domanda in vettore e cerca i chunk
    più semanticamente vicini in Qdrant.
    """
    openai, qdrant = get_clients()
    
    # Vettorizziamo la domanda con lo stesso modello usato in ingestion
    response = openai.embeddings.create(
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

    return chunks