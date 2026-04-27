from pathlib import Path
from parser import parse_pdf
from chunker import chunk_pages
from vectorizer import get_qdrant_client, create_collection_if_not_exists, embed_chunks, save_to_qdrant

DATA_DIR = Path("../../data/raw")


def main():
    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print("Nessun PDF trovato in data/raw/")
        return

    print(f"Trovati {len(pdf_files)} PDF\n")

    qdrant = get_qdrant_client()
    create_collection_if_not_exists(qdrant)

    total_chunks = 0

    for pdf_path in pdf_files:
        print(f"\nInizio processing: {pdf_path.name}")
        pages = parse_pdf(str(pdf_path))

        if not pages:
            print(f"  Nessuna pagina estratta, salto.")
            continue

        chunks = chunk_pages(pages)

        if not chunks:
            print(f"  Nessun chunk generato, salto.")
            continue

        print(f"  Chunk generati: {len(chunks)}")
        print(f"  Vettorizzazione...")

        embedded = embed_chunks(chunks, batch_size=10)
        save_to_qdrant(embedded, qdrant)

        total_chunks += len(chunks)
        print(f"  Completato: {pdf_path.name}")

    print(f"\nTotale chunk indicizzati: {total_chunks}")
    print("Ingestion completata.")

if __name__ == "__main__":
    main()