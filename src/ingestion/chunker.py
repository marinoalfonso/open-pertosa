"""
chunker.py — Chunking a partire da blocchi tipizzati.

Riceve i blocchi tipizzati prodotti dal parser e li trasforma in chunk
pronti per l'embedding. NON conosce PDF, PyMuPDF o OCR: conosce solo il
formato canonico dei blocchi.

Strategia di chunking PER TIPO di blocco:

  • paragraph → chunking a finestra con overlap (come il comportamento
    storico): si taglia su separatori naturali (\\n\\n, \\n, ". ", " ")
    per non spezzare le parole, con un overlap di caratteri tra chunk
    consecutivi per non perdere contesto ai bordi.

  • table → chunking per RIGHE con HEADER PROPAGATION:
      - se la tabella (header incluso) sta in chunk_size → un chunk unico;
      - altrimenti si spezza su confini di riga (mai a metà cella) e
        l'intestazione (header + separatrice Markdown) viene REPLICATA
        in cima a ogni chunk derivato, così ogni pezzo resta interpretabile
        da solo: le colonne non diventano mai anonime.
    Nessun overlap tra i pezzi-tabella: ogni riga è già autosufficiente
    grazie all'header replicato, l'overlap aggiungerebbe solo ridondanza.

Ogni chunk prodotto ha la forma:

    {
        "text":        str,
        "source":      str,
        "page":        int,
        "chunk_index": int,    # progressivo globale
        "type":        "paragraph" | "table",   # utile a valle (debug, reranking)
    }
"""

MIN_CHUNK_CHARS = 80   # sotto questa soglia il chunk è un frammento inutile


def _chunk_paragraph(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Chunking a finestra con overlap su testo discorsivo."""
    pieces = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size

        # Tagliamo su un separatore naturale invece che a metà parola.
        if end < text_len:
            for sep in ["\n\n", "\n", ". ", " "]:
                pos = text.rfind(sep, start, end)
                if pos != -1 and pos > start:
                    end = pos + len(sep)
                    break

        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)

        next_start = end - overlap
        # Garanzia anti-loop: avanziamo sempre.
        start = next_start if next_start > start else end

    return pieces


def _chunk_table(content: str, header: str | None, chunk_size: int) -> list[str]:
    """
    Chunking di una tabella Markdown per righe, con header propagation.

    `content` include già header + separatrice + righe dati (così come
    estratto dal parser). `header` è l'intestazione isolata da replicare.
    """
    content = content.strip()

    # Caso semplice: la tabella intera sta in un chunk.
    if len(content) <= chunk_size:
        return [content]

    lines = [l for l in content.split("\n") if l.strip() != ""]

    # Determiniamo header e righe-dati. Se il parser ha fornito l'header
    # esplicito lo usiamo; altrimenti deduciamo le prime due righe.
    if header:
        header_lines = header.split("\n")
        n_header = len(header_lines)
    else:
        # Fallback prudente: prime 2 righe come header se plausibile.
        n_header = 2 if len(lines) >= 2 else 0
        header_lines = lines[:n_header]

    header_block = "\n".join(header_lines).strip()
    data_lines = lines[n_header:]

    # Se non ci sono righe dati separabili, restituiamo tutto intero
    # (meglio un chunk grande che una tabella senza header).
    if not data_lines:
        return [content]

    chunks = []
    current = list(header_lines)         # ogni chunk parte dall'header
    current_len = len(header_block)

    for row in data_lines:
        row_len = len(row) + 1  # +1 per il newline
        # Se aggiungere questa riga sfora chunk_size e abbiamo già
        # almeno una riga dati, chiudiamo il chunk corrente.
        has_data = len(current) > n_header
        if current_len + row_len > chunk_size and has_data:
            chunks.append("\n".join(current).strip())
            current = list(header_lines)     # ricomincia con l'header replicato
            current_len = len(header_block)

        current.append(row)
        current_len += row_len

    # Ultimo chunk se contiene righe dati.
    if len(current) > n_header:
        chunks.append("\n".join(current).strip())

    return chunks


def chunk_blocks(blocks: list[dict], chunk_size: int = 1500,
                 overlap: int = 150) -> list[dict]:
    """
    Trasforma i blocchi tipizzati in chunk pronti per l'embedding.

    chunk_size: caratteri massimi indicativi per chunk
    overlap:    sovrapposizione tra chunk di PARAGRAFO (non usato sulle tabelle)
    """
    chunks = []

    for block in blocks:
        content = block.get("content", "")
        if not content or not content.strip():
            continue

        btype = block.get("type", "paragraph")

        if btype == "table":
            pieces = _chunk_table(content, block.get("header"), chunk_size)
        else:
            pieces = _chunk_paragraph(content, chunk_size, overlap)

        for piece in pieces:
            if piece and len(piece) >= MIN_CHUNK_CHARS:
                chunks.append({
                    "text": piece,
                    "source": block["source"],
                    "page": block["page"],
                    "chunk_index": len(chunks),
                    "type": btype,
                })

    return chunks