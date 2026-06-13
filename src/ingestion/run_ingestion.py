from pathlib import Path
from parser import parse_pdf
from normalizer import normalize_blocks
from chunker import chunk_blocks
from contextualizer import contextualize_chunks
from vectorizer import get_qdrant_client, create_collection_if_not_exists, embed_chunks, save_to_qdrant, recreate_collection

DATA_DIR = Path("../../data/raw")


def main():
    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print("Nessun PDF trovato in data/raw/")
        return

    print(f"Trovati {len(pdf_files)} PDF\n")

    qdrant = get_qdrant_client()
    #create_collection_if_not_exists(qdrant)
    recreate_collection(qdrant)

    total_chunks = 0

    for pdf_path in pdf_files:
        print(f"\nInizio processing: {pdf_path.name}")
        blocks = parse_pdf(str(pdf_path))

        if not blocks:
            print(f"  Nessun blocco estratto, salto.")
            continue

        blocks = normalize_blocks(blocks)

        chunks = chunk_blocks(blocks)

        if not chunks:
            print(f"  Nessun chunk generato, salto.")
            continue

        print(f"  Chunk generati: {len(chunks)}")
        
        #Contextual Retrieval: arricchiamo i chunk prima della vettorizzazione
        chunks = contextualize_chunks(chunks)
        
        print(f"  Vettorizzazione...")

        embedded = embed_chunks(chunks, batch_size=10)
        save_to_qdrant(embedded, qdrant)

        total_chunks += len(chunks)
        print(f"  Completato: {pdf_path.name}")

    print(f"\nTotale chunk indicizzati: {total_chunks}")
    print("Ingestion completata.")

if __name__ == "__main__":
    main()