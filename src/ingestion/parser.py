"""
parser.py — Estrazione PDF a blocchi tipizzati.

Il suo unico compito è trasformare un PDF in una lista di BLOCCHI
SEMANTICI TIPIZZATI, indipendenti dal resto della pipeline.

Un blocco è un dizionario con questa forma canonica:

    {
        "type":    "paragraph" | "table",
        "content": str,              # testo del paragrafo, o tabella in Markdown
        "page":    int,              # numero di pagina reale (1-based)
        "source":  str,              # nome file PDF
        # solo per type == "table":
        "header":  str | None,       # righe di intestazione della tabella
                                     # (header markdown + riga separatrice),
                                     # da replicare se la tabella viene spezzata
    }

Gli stadi successivi (chunker, ecc.) consumano SOLO questa struttura e
non sanno nulla di PyMuPDF, OCR o PDF. Se un domani si cambia estrattore,
basta che il nuovo produca blocchi in questo formato: nient'altro cambia.

Strategia di estrazione PER PAGINA (decisa una volta sola, qui):
  1. Se la pagina contiene tabelle "vere" (>= 2 righe e >= 2 colonne):
     si estrae la pagina in Markdown con PyMuPDF4LLM, che preserva la
     struttura riga/colonna. I blocchi-tabella vengono isolati dal testo.
  2. Altrimenti: estrazione nativa PyMuPDF (veloce), con pulizia
     artefatti OCR se necessario.
  3. Se la pagina nativa è vuota (scansione pura): fallback PyMuPDF4LLM
     con OCR sull'intera pagina.

Ogni pagina è processata in modo indipendente: il numero di pagina reale
è sempre noto e non viene mai ricalcolato per posizione. Questo elimina
per costruzione il bug di disallineamento dei metadati di pagina.
"""

import re
import fitz
import pymupdf4llm
from pathlib import Path

MIN_CHARS_PER_PAGE = 10
OCR_ARTIFACTS_RATIO = 0.15

# Una tabella è considerata "vera" solo se ha almeno questo numero di
# righe e colonne. find_tables() di PyMuPDF è generoso e segnala come
# tabella anche allineamenti di testo, intestazioni di delibere, blocchi
# di firme. Questa soglia evita di riconvertire in Markdown pagine che
# tabelle non sono (sostituendo testo nativo buono con Markdown peggiore).
MIN_TABLE_ROWS = 2
MIN_TABLE_COLS = 2


# ─────────────────────────────────────────────────────────────────────
# Pulizia testo
# ─────────────────────────────────────────────────────────────────────

def _clean_artifacts(text: str) -> str:
    """Ricuce gli artefatti OCR tipici (lettere/cifre spezzate da spazi)."""
    text = re.sub(r'\b([a-zA-ZÀ-ùÀ-ÿ])\s([a-zA-ZÀ-ùÀ-ÿ])\b', r'\1\2', text)
    text = re.sub(r'\b(\d)\s(\d)\b', r'\1\2', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r' ([.,;:!?»])', r'\1', text)
    return text.strip()


def _has_artifacts(text: str) -> bool:
    """Euristica: troppi token di un solo carattere ⇒ probabile OCR sporco."""
    if not text:
        return False
    tokens = text.split()
    if len(tokens) < 10:
        return False
    single_char_tokens = sum(1 for t in tokens if len(t) == 1 and t.isalpha())
    return (single_char_tokens / len(tokens)) > OCR_ARTIFACTS_RATIO


def _pulisci_markdown_tabelle(text: str) -> str:
    """
    Rimuove i <br> spuri a inizio cella dall'output Markdown di PyMuPDF4LLM,
    senza toccare i <br> legittimi (a-capo dentro celle multi-riga).
      "|<br>665.502,59|" -> "|665.502,59|"
      "|**<br>0,00**|"   -> "|**0,00**|"
    """
    return re.sub(r'\|(\*\*)?<br>', r'|\1', text)


# ─────────────────────────────────────────────────────────────────────
# Riconoscimento e isolamento delle tabelle Markdown
# ─────────────────────────────────────────────────────────────────────

def _is_table_row(line: str) -> bool:
    """Una riga di tabella Markdown inizia (a meno di spazi) con '|'."""
    return line.lstrip().startswith("|")


def _is_separator_row(line: str) -> bool:
    """
    Riga separatrice Markdown: |---|---| (eventualmente con : per
    l'allineamento). È la riga che segue l'header e definisce le colonne.
    """
    s = line.strip()
    if not s.startswith("|"):
        return False
    # tutti i caratteri tra le pipe sono '-', ':' o spazi
    cells = [c.strip() for c in s.strip("|").split("|")]
    return len(cells) > 0 and all(
        c != "" and set(c) <= set("-:") for c in cells
    )


def _split_blocks_from_markdown(md_text: str, page: int, source: str) -> list[dict]:
    """
    Scompone il Markdown di una pagina in blocchi tipizzati, separando
    le tabelle dal testo discorsivo che le circonda.

    Per ogni tabella isola anche la sua intestazione (riga header + riga
    separatrice), che il chunker replicherà se deve spezzare la tabella.
    """
    blocks = []
    lines = md_text.split("\n")
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        if _is_table_row(line):
            # Inizio di un blocco-tabella: raccogliamo tutte le righe
            # consecutive che appartengono alla tabella.
            start = i
            while i < n and (_is_table_row(lines[i]) or lines[i].strip() == ""):
                # Una riga vuota dentro il range potrebbe separare due
                # tabelle distinte: ci fermiamo se dopo la vuota non
                # ricomincia una tabella.
                if lines[i].strip() == "":
                    if i + 1 < n and _is_table_row(lines[i + 1]):
                        i += 1
                        continue
                    else:
                        break
                i += 1

            table_lines = [l for l in lines[start:i] if l.strip() != ""]

            # Isoliamo l'header: riga 0 = header, riga 1 = separatrice
            # (se presente). Il resto sono righe dati.
            header = None
            if len(table_lines) >= 2 and _is_separator_row(table_lines[1]):
                header = "\n".join(table_lines[0:2])

            table_md = "\n".join(table_lines).strip()
            if table_md:
                blocks.append({
                    "type": "table",
                    "content": table_md,
                    "page": page,
                    "source": source,
                    "header": header,
                })
        elif line.strip() == "":
            i += 1
        else:
            # Testo discorsivo: raccogliamo fino alla prossima tabella
            start = i
            while i < n and not _is_table_row(lines[i]):
                i += 1
            para = "\n".join(lines[start:i]).strip()
            if para:
                blocks.append({
                    "type": "paragraph",
                    "content": para,
                    "page": page,
                    "source": source,
                    "header": None,
                })

    return blocks


# ─────────────────────────────────────────────────────────────────────
# Estrazione pagina per pagina
# ─────────────────────────────────────────────────────────────────────

def _page_has_real_table(page) -> bool:
    """
    True se la pagina contiene almeno una tabella "vera", cioè con un
    numero minimo di righe e colonne. Filtra i falsi positivi di
    find_tables() (allineamenti, firme, moduli).
    """
    try:
        found = page.find_tables()
        for tab in found.tables:
            # tab.row_count / tab.col_count disponibili in PyMuPDF recenti.
            rows = getattr(tab, "row_count", None)
            cols = getattr(tab, "col_count", None)
            if rows is None or cols is None:
                # fallback: deduci dalle celle estratte
                extracted = tab.extract()
                rows = len(extracted)
                cols = max((len(r) for r in extracted), default=0)
            if rows >= MIN_TABLE_ROWS and cols >= MIN_TABLE_COLS:
                return True
    except Exception:
        pass
    return False


def _markdown_for_pages(pdf_path: str, page_indices: list[int]) -> dict[int, str]:
    """
    Estrae in Markdown solo le pagine indicate (indici 0-based).
    Restituisce una mappa {pagina_reale_1based: markdown}.

    L'allineamento NON si basa sulla posizione nella lista di output
    (fragile se una pagina viene saltata), ma sui metadata di pagina che
    PyMuPDF4LLM riporta. Estraendo pagina per pagina con una lista di un
    solo elemento eliminiamo ogni ambiguità: una chiamata = una pagina.
    """
    out = {}
    for idx in page_indices:
        try:
            chunks = pymupdf4llm.to_markdown(
                pdf_path,
                pages=[idx],
                page_chunks=True,
                ocr_language="ita+eng",
            )
        except Exception as e:
            print(f"    PyMuPDF4LLM fallito su pagina {idx + 1}: {e}")
            continue

        # Con pages=[idx] e page_chunks=True ci aspettiamo 1 chunk.
        text_parts = [c.get("text", "") for c in chunks]
        md = "\n".join(t for t in text_parts if t).strip()
        if md:
            out[idx + 1] = _pulisci_markdown_tabelle(md)
    return out


def _extract_blocks(pdf_path: str) -> tuple[list[dict], int, dict]:
    """
    Estrae il PDF in blocchi tipizzati, una pagina alla volta.

    Ritorna: (blocchi, numero_pagine_totali, conteggio_strategie)
    """
    source = Path(pdf_path).name
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    strategy_counts = {}

    # 1° passaggio: per ogni pagina decidiamo la strategia e raccogliamo
    # gli indici delle pagine che richiedono estrazione Markdown
    # (tabelle vere) o OCR (pagine vuote).
    native_text = {}      # pagina_1based -> testo nativo pulito
    table_pages = []      # indici 0-based con tabelle vere
    empty_pages = []      # indici 0-based senza testo (scansioni)

    for page_num, page in enumerate(doc):
        text = page.get_text("text").strip()
        has_table = _page_has_real_table(page)

        if has_table:
            table_pages.append(page_num)
            strategy_counts["table-markdown"] = strategy_counts.get("table-markdown", 0) + 1
        elif len(text) >= MIN_CHARS_PER_PAGE:
            if _has_artifacts(text):
                text = _clean_artifacts(text)
                strategy_counts["native+clean"] = strategy_counts.get("native+clean", 0) + 1
            else:
                strategy_counts["native"] = strategy_counts.get("native", 0) + 1
            native_text[page_num + 1] = text
        else:
            empty_pages.append(page_num)
            strategy_counts["empty"] = strategy_counts.get("empty", 0) + 1

    doc.close()

    # 2° passaggio: estrazione Markdown per le pagine con tabelle
    table_md = _markdown_for_pages(pdf_path, table_pages) if table_pages else {}

    # 3° passaggio: OCR fallback per le pagine vuote (scansioni)
    ocr_md = _markdown_for_pages(pdf_path, empty_pages) if empty_pages else {}

    # Componiamo i blocchi in ordine di pagina
    blocks = []
    for page in range(1, total_pages + 1):
        if page in table_md:
            blocks.extend(_split_blocks_from_markdown(table_md[page], page, source))
        elif page in native_text:
            blocks.append({
                "type": "paragraph",
                "content": native_text[page],
                "page": page,
                "source": source,
                "header": None,
            })
        elif page in ocr_md:
            # Le pagine OCR potrebbero anch'esse contenere tabelle: le
            # scomponiamo con lo stesso meccanismo.
            blocks.extend(_split_blocks_from_markdown(ocr_md[page], page, source))

    return blocks, total_pages, strategy_counts


# ─────────────────────────────────────────────────────────────────────
# API pubblica
# ─────────────────────────────────────────────────────────────────────

def parse_pdf(pdf_path: str) -> list[dict]:
    """
    Estrae un PDF in una lista di blocchi tipizzati (paragraph | table).

    È l'unica funzione che il resto della pipeline deve conoscere.
    """
    path = Path(pdf_path)
    print(f"\nParsing: {path.name}")

    blocks, total_pages, strategy_counts = _extract_blocks(pdf_path)

    n_tables = sum(1 for b in blocks if b["type"] == "table")
    n_paras = sum(1 for b in blocks if b["type"] == "paragraph")
    total_chars = sum(len(b["content"]) for b in blocks)

    print(f"  Pagine: {total_pages} totali")
    print(f"  Strategie pagina: {strategy_counts}")
    print(f"  Blocchi: {len(blocks)} ({n_paras} paragrafi, {n_tables} tabelle)")
    print(f"  Caratteri totali: {total_chars}")

    return blocks