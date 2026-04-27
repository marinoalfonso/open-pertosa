def chunk_pages(pages: list[dict], chunk_size: int = 1500,
                overlap: int = 150) -> list[dict]:
    """
    Suddivide il testo estratto in chunk con overlap.

    chunk_size: caratteri massimi per chunk
    overlap: caratteri ripetuti tra chunk consecutivi
              (evita di perdere contesto ai bordi)
    """
    chunks = []

    for page in pages:
        text = page["text"]
        if not text.strip():
            continue

        start = 0
        while start < len(text):
            end = start + chunk_size

            # Cerchiamo di tagliare su un punto o a capo
            # invece che a metà parola
            if end < len(text):
                for sep in ["\n\n", "\n", ". ", " "]:
                    pos = text.rfind(sep, start, end)
                    if pos != -1:
                        end = pos + len(sep)
                        break

            chunk_text = text[start:end].strip()

            # Filtriamo chunk troppo corti — frammenti inutili
            # come firme spezzate o residui di formattazione
            if chunk_text and len(chunk_text) >= 80:
                chunks.append({
                    "text": chunk_text,
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_index": len(chunks)
                })

            start = end - overlap

    return chunks