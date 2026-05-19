import re
import fitz
import pymupdf4llm
from pathlib import Path

MIN_CHARS_PER_PAGE = 10
OCR_ARTIFACTS_RATIO = 0.15

def _pulisci_markdown_tabelle(text: str) -> str:
    """
    Pulisce l'output Markdown di PyMuPDF4LLM dal rumore tipico
    sulle tabelle, SENZA toccare i <br> legittimi (a capo dentro
    una cella multi-riga, es. intestazioni lunghe).

    Interviene solo sui <br> appiccicati subito dopo l'apertura
    di una cella (eventualmente dopo il ** del grassetto):
      "|<br>665.502,59|"   -> "|665.502,59|"
      "|**<br>0,00**|"     -> "|**0,00**|"
    Un <br> a inizio cella è sempre un artefatto: un a-capo
    legittimo non sta mai all'inizio del contenuto di una cella.
    """
    return re.sub(r'\|(\*\*)?<br>', r'|\1', text)


def _clean_artifacts(text: str) -> str:
    text = re.sub(r'\b([a-zA-ZÀ-ùÀ-ÿ])\s([a-zA-ZÀ-ùÀ-ÿ])\b', r'\1\2', text)
    text = re.sub(r'\b(\d)\s(\d)\b', r'\1\2', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r' ([.,;:!?»])', r'\1', text)
    return text.strip()


def _has_artifacts(text: str) -> bool:
    if not text:
        return False
    tokens = text.split()
    if len(tokens) < 10:
        return False
    single_char_tokens = sum(1 for t in tokens if len(t) == 1 and t.isalpha())
    return (single_char_tokens / len(tokens)) > OCR_ARTIFACTS_RATIO


def _parse_with_pymupdf(pdf_path: str) -> list[dict]:
    """Estrazione nativa con PyMuPDF — veloce, leggera."""
    pages = []
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    strategy_counts = {}
    table_page_indices = []   # indici 0-based delle pagine con tabelle

    for page_num, page in enumerate(doc):
        text = page.get_text("text").strip()

        # Rilevamento tabelle — find_tables() è nativo e leggero.
        # Se fallisce su una pagina non blocchiamo l'estrazione.
        try:
            if page.find_tables().tables:
                table_page_indices.append(page_num)
        except Exception:
            pass

        if len(text) >= MIN_CHARS_PER_PAGE:
            if _has_artifacts(text):
                text = _clean_artifacts(text)
                strategy = "native+clean"
            else:
                strategy = "native"
        else:
            strategy = "empty"

        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        if text and strategy != "empty":
            pages.append({
                "page": page_num + 1,
                "text": text,
                "char_count": len(text),
                "source": Path(pdf_path).name,
                "strategy": strategy
            })

    doc.close()
    return pages, total_pages, strategy_counts, table_page_indices


def _parse_with_pymupdf4llm(pdf_path: str, pages: list[int] = None) -> list[dict]:
    """
    Estrazione con PyMuPDF4LLM → Markdown.

    Se `pages` è None: processa tutto il PDF (fallback pagine-vuote).
    Se `pages` è una lista di indici 0-based: processa solo quelle
    pagine (riconversione tabelle).

    IMPORTANTE — allineamento pagine:
    con il parametro `pages`, PyMuPDF4LLM rinumera metadata['page']
    in base alla POSIZIONE nella lista (0,1,2...), non all'indice
    reale nel PDF. Per questo, quando `pages` è fornita, NON usiamo
    metadata['page']: associamo il chunk j-esimo all'indice pages[j].
    """
    try:
        kwargs = {
            "page_chunks": True,
            "ocr_language": "ita+eng",
        }
        if pages is not None:
            kwargs["pages"] = pages

        chunks = pymupdf4llm.to_markdown(pdf_path, **kwargs)

        result = []
        for j, chunk in enumerate(chunks):
            text = chunk.get("text", "").strip()
            if not text or len(text) < MIN_CHARS_PER_PAGE:
                continue

            if pages is not None:
                # riconversione tabelle: pagina reale = pages[j] + 1
                # (allineamento per posizione, non per metadata)
                if j < len(pages):
                    pagina = pages[j] + 1
                else:
                    pagina = chunk.get("metadata", {}).get("page", 0) + 1
                text = _pulisci_markdown_tabelle(text)
                strategy = "pymupdf4llm-table"
            else:
                # fallback pagine-vuote: comportamento originale
                pagina = chunk.get("metadata", {}).get("page", 0) + 1
                strategy = "pymupdf4llm"

            result.append({
                "page": pagina,
                "text": text,
                "char_count": len(text),
                "source": Path(pdf_path).name,
                "strategy": strategy
            })
        return result
    except Exception as e:
        print(f"  PyMuPDF4LLM fallito: {e}")
        return []


def parse_pdf(pdf_path: str) -> list[dict]:
    path = Path(pdf_path)
    print(f"\nParsing: {path.name}")

    # Step 1 — PyMuPDF nativo
    pages, total_pages, strategy_counts, table_page_indices = _parse_with_pymupdf(pdf_path)

    total_chars = sum(p["char_count"] for p in pages)
    avg_chars = total_chars / len(pages) if pages else 0

    print(f"  Pagine: {total_pages} totali, {len(pages)} con testo")
    print(f"  Caratteri medi: {avg_chars:.0f}")
    print(f"  Strategie: {strategy_counts}")

    # Step 2 — Fallback PyMuPDF4LLM se testo insufficiente
    empty_count = strategy_counts.get("empty", 0)
    if empty_count > 0:
        print(f"  → {empty_count} pagine vuote, provo PyMuPDF4LLM...")
        pages_4llm = _parse_with_pymupdf4llm(pdf_path)
        if pages_4llm:
            # Sostituiamo solo le pagine vuote con quelle di pymupdf4llm
            existing_pages = {p["page"] for p in pages}
            for p in pages_4llm:
                if p["page"] not in existing_pages:
                    pages.append(p)
            pages.sort(key=lambda x: x["page"])
            print(f"  → PyMuPDF4LLM ha aggiunto {len(pages_4llm)} pagine")
    
    # Step 3 — Riconversione tabelle con PyMuPDF4LLM (Markdown)
    # Le pagine con tabelle vengono ri-estratte in Markdown, che
    # preserva la struttura riga/colonna, e SOSTITUISCONO la
    # versione nativa (che appiattisce le tabelle in testo lineare).
    if table_page_indices:
        print(f"  → {len(table_page_indices)} pagine con tabelle, riconverto in Markdown...")
        pages_tbl = _parse_with_pymupdf4llm(pdf_path, pages=table_page_indices)

        if pages_tbl:
            tbl_by_page = {p["page"]: p for p in pages_tbl}
            sostituite = 0
            for i, p in enumerate(pages):
                if p["page"] in tbl_by_page:
                    pages[i] = tbl_by_page[p["page"]]
                    sostituite += 1
            print(f"  → {sostituite} pagine-tabella sostituite con Markdown")

    return pages

    return pages