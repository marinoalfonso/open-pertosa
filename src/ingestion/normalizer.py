"""
normalizer.py — Normalizzazione dei blocchi (Stadio 2 della pipeline).

Si inserisce TRA il parser (estrazione → blocchi tipizzati) e il chunker
(blocchi → chunk). Riceve blocchi nel formato canonico e restituisce
blocchi nello stesso formato, eventualmente corretti.

Compito attuale: PROPAGAZIONE DELL'INTESTAZIONE tra tabelle multi-pagina.

Il problema che risolve:
    Una tabella lunga che attraversa più pagine viene estratta dal parser
    come più blocchi-tabella separati (uno per pagina), perché ogni pagina
    è processata indipendentemente. Spesso la pagina di continuazione NON
    ripete l'intestazione: il suo blocco-tabella è una sequenza di righe
    con colonne ANONIME (numeri senza sapere cosa numerano).

La soluzione (volutamente prudente — "presta", non "fonde"):
    Se un blocco-tabella è privo di intestazione propria ed è immediatamente
    preceduto (sulla pagina precedente) da un blocco-tabella CON intestazione
    e con lo STESSO numero di colonne, gli si presta quell'intestazione.

    NON si fondono i blocchi: ogni blocco resta sulla sua pagina reale, così
    le citazioni restano corrette. Si aggiunge solo l'header mancante, in
    modo che il chunker possa poi replicarlo su ogni chunk (header propagation).

Perché prudente:
    Propagare un header sbagliato è peggio che non propagarlo. Per questo i
    criteri di compatibilità sono restrittivi: pagine consecutive, secondo
    blocco senza header proprio, stesso numero di colonne. Nel dubbio, non
    si tocca nulla.
"""


def _count_columns(table_md: str) -> int | None:
    """
    Deduce il numero di colonne da una tabella Markdown, contando le celle
    della prima riga (escludendo i bordi). Ritorna None se non determinabile.
    """
    for line in table_md.split("\n"):
        s = line.strip()
        if s.startswith("|"):
            # "| a | b | c |" -> celle = ['a', 'b', 'c']
            cells = [c for c in s.strip("|").split("|")]
            n = len(cells)
            if n > 0:
                return n
    return None


def _first_data_row_columns(table_md: str, has_header: bool) -> int | None:
    """
    Conta le colonne guardando la prima riga DATI (saltando header +
    separatrice se presenti). Più robusto di _count_columns quando il
    blocco orfano inizia direttamente con i dati.
    """
    lines = [l for l in table_md.split("\n") if l.strip().startswith("|")]
    if not lines:
        return None
    # se has_header, le prime due righe sono header+separatrice
    idx = 2 if has_header and len(lines) >= 2 else 0
    if idx >= len(lines):
        idx = 0
    cells = [c for c in lines[idx].strip().strip("|").split("|")]
    return len(cells) if cells else None


def _has_own_header(block: dict) -> bool:
    """Un blocco-tabella ha un'intestazione propria?"""
    return bool(block.get("header"))


def normalize_blocks(blocks: list[dict]) -> list[dict]:
    """
    Applica la propagazione dell'intestazione tra tabelle multi-pagina.

    Ritorna la stessa lista di blocchi, con gli header prestati dove
    appropriato. I blocchi non-tabella e quelli già provvisti di header
    restano invariati. Nessun blocco viene fuso, spostato o eliminato.
    """
    propagati = 0

    for i, block in enumerate(blocks):
        if block.get("type") != "table":
            continue
        if _has_own_header(block):
            continue  # ha già la sua intestazione, niente da fare

        # Cerchiamo il candidato "donatore": il blocco-tabella immediatamente
        # precedente, su una pagina consecutiva, con intestazione propria.
        donor = _find_donor(blocks, i)
        if donor is None:
            continue

        # Verifica di compatibilità: stesso numero di colonne.
        donor_cols = _count_columns(donor["header"])
        orphan_cols = _first_data_row_columns(block["content"], has_header=False)

        if donor_cols is None or orphan_cols is None:
            continue
        if donor_cols != orphan_cols:
            continue  # strutture diverse: non è una continuazione, non tocco

        # Prestito dell'intestazione: aggiorniamo header E anteponiamo
        # l'header al contenuto, così il blocco diventa autosufficiente
        # (il chunker lo tratterà come una tabella completa di header).
        block["header"] = donor["header"]
        block["content"] = donor["header"] + "\n" + block["content"]
        propagati += 1

    if propagati:
        print(f"  → Header propagato a {propagati} tabelle di continuazione")

    return blocks


def _find_donor(blocks: list[dict], orphan_idx: int) -> dict | None:
    """
    Trova il blocco-tabella "donatore" per il blocco orfano in posizione
    orphan_idx: dev'essere un blocco-tabella, con header proprio, situato
    su una pagina IMMEDIATAMENTE precedente (consecutiva).

    Si guarda indietro saltando eventuali blocchi non-tabella sulla stessa
    pagina dell'orfano (es. un'intestazione di pagina), ma ci si ferma se
    si cambia pagina di più di 1 o se si incontra un blocco non riconducibile.
    """
    orphan_page = blocks[orphan_idx].get("page")
    if orphan_page is None:
        return None

    j = orphan_idx - 1
    while j >= 0:
        prev = blocks[j]
        prev_page = prev.get("page")

        # Il donatore deve stare sulla pagina immediatamente precedente.
        if prev_page is None:
            return None
        if prev_page < orphan_page - 1:
            return None  # troppo indietro: non è una continuazione diretta
        if prev_page > orphan_page:
            return None  # ordine inatteso

        if prev.get("type") == "table" and _has_own_header(prev):
            # Donatore valido solo se sta proprio sulla pagina precedente
            # (o sulla stessa, caso raro di due frammenti già divisi).
            if prev_page in (orphan_page - 1, orphan_page):
                return prev
            return None

        # Se tra le due tabelle c'è del testo discorsivo, è improbabile che
        # la seconda sia una continuazione diretta della prima: interrompiamo.
        # Soglia bassa (50 char): vogliamo SALTARE solo boilerplate di pagina
        # (numero pagina, data in testata, "segue tabella") ma FERMARCI su
        # qualsiasi frase di senso compiuto, che di solito introduce una
        # tabella nuova e diversa. Nel dubbio, non propaghiamo.
        if prev.get("type") == "paragraph" and len(prev.get("content", "").strip()) > 50:
            return None

        j -= 1

    return None
