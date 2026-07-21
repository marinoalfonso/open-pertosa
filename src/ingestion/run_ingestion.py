from pathlib import Path
from parser import parse_pdf
from normalizer import normalize_blocks
from chunker import chunk_blocks
from contextualizer import contextualize_chunks
from vectorizer import (
    get_qdrant_client,
    create_collection_if_not_exists,
    embed_chunks,
    save_to_qdrant,
    recreate_collection,
    is_already_indexed, 
)
from metadata_extractor import build_parent_index

DATA_DIR = Path("../../data/raw")


def main():
    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print("Nessun PDF trovato in data/raw/")
        return

    print(f"Trovati {len(pdf_files)} PDF\n")

    qdrant = get_qdrant_client()
    create_collection_if_not_exists(qdrant)

    # ─── Pre-scan: costruzione indice padri per ereditarietà allegati ───
    print("Costruzione indice padri...")
    filenames = [f.name for f in pdf_files]
    parent_index = build_parent_index(
        filenames=filenames,
        data_dir=DATA_DIR,
        parse_pdf_fn=parse_pdf,
    )
    print(f"Indice padri pronto: {len(parent_index)} entry\n")

    total_chunks = 0

    for pdf_path in pdf_files:
        if is_already_indexed(qdrant, pdf_path.name):
            print(f"  [SKIP] {pdf_path.name} già indicizzato")
            continue
        print(f"\nInizio processing: {pdf_path.name}")
        blocks = parse_pdf(str(pdf_path), parent_index=parent_index)

        if not blocks:
            print(f"  Nessun blocco estratto, salto.")
            continue

        blocks = normalize_blocks(blocks)
        chunks = chunk_blocks(blocks)

        if not chunks:
            print(f"  Nessun chunk generato, salto.")
            continue

        print(f"  Chunk generati: {len(chunks)}")

        # Contextual Retrieval: arricchiamo i chunk prima della vettorizzazione
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