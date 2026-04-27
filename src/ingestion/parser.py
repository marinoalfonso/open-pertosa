import re
import fitz
import pymupdf4llm
from pathlib import Path

MIN_CHARS_PER_PAGE = 10
OCR_ARTIFACTS_RATIO = 0.15


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

    for page_num, page in enumerate(doc):
        text = page.get_text("text").strip()

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
    return pages, total_pages, strategy_counts


def _parse_with_pymupdf4llm(pdf_path: str) -> list[dict]:
    """Fallback con PyMuPDF4LLM — più potente, senza OCR."""
    try:
        chunks = pymupdf4llm.to_markdown(
            pdf_path,
            page_chunks=True,
            ocr_language="ita+eng"
        )
        pages = []
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if not text or len(text) < MIN_CHARS_PER_PAGE:
                continue
            pages.append({
                "page": chunk.get("metadata", {}).get("page", 0) + 1,
                "text": text,
                "char_count": len(text),
                "source": Path(pdf_path).name,
                "strategy": "pymupdf4llm"
            })
        return pages
    except Exception as e:
        print(f"  PyMuPDF4LLM fallito: {e}")
        return []


def parse_pdf(pdf_path: str) -> list[dict]:
    path = Path(pdf_path)
    print(f"\nParsing: {path.name}")

    # Step 1 — PyMuPDF nativo
    pages, total_pages, strategy_counts = _parse_with_pymupdf(pdf_path)

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

    return pages