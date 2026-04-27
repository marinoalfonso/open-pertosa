import os
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
import uuid

load_dotenv()

EMBEDDING_DIM = 1536  # dimensione di text-embedding-3-small
COLLECTION_NAME = "pertosa_docs"


def get_openai_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_qdrant_client():
    return QdrantClient(host="localhost", port=6333)


def create_collection_if_not_exists(qdrant: QdrantClient):
    existing = [c.name for c in qdrant.get_collections().collections]

    if COLLECTION_NAME not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE
            )
        )
        print(f"Collection '{COLLECTION_NAME}' creata")
    else:
        print(f"Collection '{COLLECTION_NAME}' già esistente")


def embed_chunks(chunks: list[dict], batch_size: int =20) -> list[dict]:
    client = get_openai_client()
    model = os.getenv("OPENAI_EMBEDDING_MODEL")

    embedded = []
    total = len(chunks)

    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]

        print(f"  Embedding batch {i//batch_size + 1}/{-(-total//batch_size)}")

        response = client.embeddings.create(
            input=texts,
            model=model
        )

        for chunk, embedding_obj in zip(batch, response.data):
            embedded.append({
                **chunk,
                "vector": embedding_obj.embedding
            })

    return embedded


def save_to_qdrant(embedded_chunks: list[dict], qdrant: QdrantClient):
    points = []
    for chunk in embedded_chunks:
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=chunk["vector"],
            payload={
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"]
            }
        ))

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print(f"  Salvati {len(points)} punti in Qdrant")