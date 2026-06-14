"""
Estrazione metadati temporali e di tipologia dai documenti amministrativi.

Logica a cascata:
  Step 1 — Regex sul nome file (gratis, deterministico)
  Step 2 — Regex sul testo del documento (gratis, deterministico)
  Step 3 — LLM sull'intestazione e calce (costoso, robusto)

Restituisce un dizionario con:
  - tipo_atto:        una delle categorie di TIPI_ATTO
  - data_atto:        stringa ISO "YYYY-MM-DD" oppure None
  - anno:             intero oppure None
  - data_precisione:  "giorno" | "anno" | "ignoto"
  - extraction_log:   stringa che spiega quale step ha prodotto cosa
                      (utile per debug, da rimuovere in produzione se vuoi)
"""

import os
import re
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

EXTRACT_MODEL = os.getenv("OPENAI_EXTRACT_MODEL", "gpt-4o-mini")

TIPI_ATTO = [
    "delibera_giunta",
    "delibera_consiglio",
    "determina",
    "decreto",
    "ordinanza",
    "regolamento",
    "bando_avviso",
    "progetto_tecnico",
    "parere",
    "comunicazione",
    "altro",
]

# ─────────────────────────────────────────────────────────────
# STEP 1 — Regex sul nome file
# ─────────────────────────────────────────────────────────────

# Famiglia A: "determinazioni_tec_n._49-2026.pdf", "decreti_dec_n._5-2026.pdf"
# Estrae tipo + numero + anno
_RE_NAME_TECNICO = re.compile(
    r"_(determinazioni_tec|determinazioni|delibera_di_giunta|"
    r"delibera_di_consiglio|decreti|ordinanze?|regolamento|"
    r"bando|avviso)_(?:tec_)?n\.?_(\d+)[-_](\d{4})",
    re.IGNORECASE,
)

# Famiglia B: "ordinanza-n-5-2026-...", "avviso-di-convocazione-..."
_RE_NAME_DESCRITTIVO = re.compile(
    r"\b(ordinanza|determinazion[ei]|delibera|decreto|"
    r"bando|avviso|regolamento|parere|progetto|capitolato)\b",
    re.IGNORECASE,
)

# Anno isolato nel nome file (es. "...-2026-..." o "..._2026.pdf")
_RE_ANNO_NAME = re.compile(r"[-_/](\d{4})(?:[-_/]|\.pdf$)")


_TIPO_ATTO_DA_NAMING = {
    "determinazioni": "determina",
    "determinazioni_tec": "determina",
    "determinazione": "determina",
    "determinazioni": "determina",
    "delibera_di_giunta": "delibera_giunta",
    "delibera_di_consiglio": "delibera_consiglio",
    "delibera": "altro",  # ambiguo, sarà raffinato dall'LLM
    "decreti": "decreto",
    "decreto": "decreto",
    "ordinanza": "ordinanza",
    "ordinanze": "ordinanza",
    "regolamento": "regolamento",
    "bando": "bando_avviso",
    "avviso": "bando_avviso",
    "parere": "parere",
    "progetto": "progetto_tecnico",
    "capitolato": "progetto_tecnico",
}


def _step1_nome_file(filename: str) -> dict:
    """Tentativo di estrazione dal nome file. Ritorna sempre un dizionario;
    i campi non determinati restano None."""
    result = {"tipo_atto": None, "anno": None}

    # Tentativo Famiglia A — formato tecnico con numero e anno
    m = _RE_NAME_TECNICO.search(filename)
    if m:
        raw_tipo = m.group(1).lower()
        result["tipo_atto"] = _TIPO_ATTO_DA_NAMING.get(raw_tipo, "altro")
        result["anno"] = int(m.group(3))
        return result

    # Tentativo Famiglia B — naming descrittivo
    m = _RE_NAME_DESCRITTIVO.search(filename)
    if m:
        raw_tipo = m.group(1).lower()
        # Normalizziamo varianti: "determinazioni" e "determinazioni" → "determina"
        if raw_tipo.startswith("determinazion"):
            result["tipo_atto"] = "determina"
        else:
            result["tipo_atto"] = _TIPO_ATTO_DA_NAMING.get(raw_tipo, "altro")

    # Anno (anche se il tipo non è stato trovato)
    m = _RE_ANNO_NAME.search(filename)
    if m:
        anno_candidato = int(m.group(1))
        if 2000 <= anno_candidato <= 2030:
            result["anno"] = anno_candidato

    return result


# ─────────────────────────────────────────────────────────────
# STEP 2 — Regex sul testo del documento
# ─────────────────────────────────────────────────────────────

# "Pertosa, lì 15/05/2026" e varianti tipiche dei testi PA
_RE_DATA_TESTO = re.compile(
    r"(?:Pertosa[,\s]+)?(?:lì|li')\s+(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})",
    re.IGNORECASE,
)

# Data generica DD/MM/YYYY (fallback, più rumoroso)
_RE_DATA_GENERICA = re.compile(r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})\b")


def _step2_data_dal_testo(document_header: str, 
                          expected_year: int | None = None) -> tuple[str | None, int | None]:
    snippet = document_header[:2000]

    for pattern in (_RE_DATA_TESTO, _RE_DATA_GENERICA):
        for m in pattern.finditer(snippet):
            gg, mm, aaaa = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if 1 <= gg <= 31 and 1 <= mm <= 12 and 2000 <= aaaa <= 2030:
                if expected_year is not None and aaaa != expected_year:
                    continue
                data_iso = f"{aaaa:04d}-{mm:02d}-{gg:02d}"
                return data_iso, aaaa

    return None, None


# ─────────────────────────────────────────────────────────────
# STEP 3 — Fallback LLM
# ─────────────────────────────────────────────────────────────

_EXTRACT_PROMPT = """Sei un assistente che analizza documenti amministrativi \
del Comune di Pertosa (Salerno).

Dato l'estratto di un documento, identifica:
1. il tipo di atto;
2. la data dell'atto (vicino al luogo "Pertosa, lì DD/MM/AAAA" oppure in calce \
accanto alla firma).

Categorie possibili per il tipo di atto:
{tipi}

Estratto del documento:
{header}

Rispondi SOLO con un JSON valido, senza commenti né markdown:
{{
  "tipo_atto": "una delle categorie sopra, oppure altro",
  "data_atto": "YYYY-MM-DD se trovata, altrimenti null"
}}"""


def _step3_llm(document_header: str, client: OpenAI) -> dict:
    """Fallback LLM. In caso di errore restituisce campi None."""
    if not document_header.strip():
        return {"tipo_atto": None, "data_atto": None}

    prompt = _EXTRACT_PROMPT.format(
        tipi=", ".join(TIPI_ATTO),
        header=document_header[:3000],  # limitiamo per controllare i costi
    )

    try:
        for attempt in range(5):
            try:
                response = client.chat.completions.create(
                    model=EXTRACT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=120,
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content.strip()
                data = json.loads(content)

                tipo = data.get("tipo_atto")
                if tipo not in TIPI_ATTO:
                    tipo = "altro"

                data_atto = data.get("data_atto")
                if data_atto and not re.match(r"^\d{4}-\d{2}-\d{2}$", data_atto):
                    data_atto = None

                return {"tipo_atto": tipo, "data_atto": data_atto}

            except Exception as e:
                if "429" in str(e) and attempt < 4:
                    time.sleep(2 ** attempt)
                else:
                    raise
    except Exception as e:
        print(f"  [metadata_extractor] errore LLM: {e}")
        return {"tipo_atto": None, "data_atto": None}


# ─────────────────────────────────────────────────────────────
# ORCHESTRATORE — questa è l'unica funzione che il resto del codice chiama
# ─────────────────────────────────────────────────────────────

def extract_metadata(filename: str,
                     document_header: str,
                     client: OpenAI = None,
                     parent_index: dict = None) -> dict:
    """Estrae i metadati di un documento usando la cascata in tre step.
    L'LLM viene invocato solo se necessario.
    
    Se il file è un allegato e parent_index contiene il suo padre,
    i metadati TEMPORALI (data_atto, anno, data_precisione) sono ereditati 
    dal padre. Il tipo_atto resta estratto autonomamente dall'allegato.
    """
    log = []
    
    # ─── EREDITARIETÀ ALLEGATI ───
    inherited_temporal = None
    if parent_index and is_allegato(filename):
        prefix = get_parent_prefix(filename)
        if prefix and prefix in parent_index:
            inherited_temporal = parent_index[prefix]
            log.append(f"ereditarietà: dati temporali dal padre {prefix}")

    # ─── Step 1 ───
    s1 = _step1_nome_file(filename)
    if s1["tipo_atto"]:
        log.append(f"step1: tipo_atto={s1['tipo_atto']} (da nome file)")
    if s1["anno"]:
        log.append(f"step1: anno={s1['anno']} (da nome file)")

    tipo_atto = s1["tipo_atto"]
    anno = s1["anno"]
    data_atto = None
    data_precisione = "ignoto"

    # ─── Step 2: SOLO se non abbiamo dati ereditati dal padre ───
    if inherited_temporal is None:
        data_iso, anno_dal_testo = _step2_data_dal_testo(
            document_header, 
            expected_year=anno
        )
        if data_iso:
            data_atto = data_iso
            if anno is None:
                anno = anno_dal_testo
            data_precisione = "giorno"
            log.append(f"step2: data_atto={data_iso} (regex sul testo)")
    
    # ─── Step 3 — LLM solo se ancora mancano informazioni essenziali ───
    # Anche qui: skippa se abbiamo dati ereditati
    serve_llm = (tipo_atto is None) or (
        inherited_temporal is None and data_atto is None and anno is None
    )
    if serve_llm:
        if client is None:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        s3 = _step3_llm(document_header, client)
        
        if tipo_atto is None and s3["tipo_atto"]:
            tipo_atto = s3["tipo_atto"]
            log.append(f"step3: tipo_atto={tipo_atto} (LLM)")
        
        if inherited_temporal is None and data_atto is None and s3["data_atto"]:
            data_atto = s3["data_atto"]
            data_precisione = "giorno"
            if anno is None:
                anno = int(s3["data_atto"][:4])
            log.append(f"step3: data_atto={data_atto} (LLM)")

    # ─── Applica ereditarietà se presente (sovrascrive temporali) ───
    if inherited_temporal is not None:
        data_atto = inherited_temporal["data_atto"]
        anno = inherited_temporal["anno"]
        data_precisione = inherited_temporal["data_precisione"]

    # ─── Finalizzazione ───
    if tipo_atto is None:
        tipo_atto = "altro"
        log.append("fallback: tipo_atto=altro (nessuno step ha riconosciuto)")

    if data_atto is None and anno is not None and data_precisione == "ignoto":
        data_precisione = "anno"

    return {
        "tipo_atto": tipo_atto,
        "data_atto": data_atto,
        "anno": anno,
        "data_precisione": data_precisione,
        "extraction_log": " | ".join(log),
    }

# ─────────────────────────────────────────────────────────────
# COSTRUZIONE INDICE PADRI (per ereditarietà metadati negli allegati)
# ─────────────────────────────────────────────────────────────

_RE_ALLEGATO = re.compile(r"^(.+?)__allegato-", re.IGNORECASE)


def is_allegato(filename: str) -> bool:
    """Riconosce se un file è un allegato dal pattern di naming."""
    return "__allegato-" in filename.lower()


def get_parent_prefix(filename: str) -> str | None:
    """Estrae il prefisso del documento padre da un nome di allegato.
    Es: '0118_delibera_n._16-2026__allegato-01-piano.pdf'
        → '0118_delibera_n._16-2026'
    Restituisce None se il file non è un allegato."""
    m = _RE_ALLEGATO.match(filename)
    if m:
        return m.group(1)
    return None


def build_parent_index(filenames: list[str],
                       data_dir,
                       parse_pdf_fn,
                       client: OpenAI = None) -> dict:
    """
    Costruisce un indice dei metadati temporali per i documenti padre.
    Solo i campi temporali (data_atto, anno, data_precisione) sono salvati.
    
    Per ciascun padre identificato:
      - se è presente nel corpus, estrae i metadati dal suo contenuto
      - se NON è presente, estrae solo dal nome (anno) e marca data_precisione="anno"
    
    Ritorna: dict {prefix_padre: {data_atto, anno, data_precisione}}
    """
    from pathlib import Path
    
    # Identifica i padri di cui abbiamo bisogno (cioè quelli referenziati 
    # dagli allegati presenti nel corpus)
    needed_parents = set()
    for fn in filenames:
        if is_allegato(fn):
            prefix = get_parent_prefix(fn)
            if prefix:
                needed_parents.add(prefix)
    
    if not needed_parents:
        print("  [parent_index] nessun allegato trovato, indice vuoto")
        return {}
    
    print(f"  [parent_index] identificati {len(needed_parents)} padri da indicizzare")
    
    # Mappa: prefix → nome file effettivo del padre (se presente nel corpus)
    # Es: '0118_delibera_n._16-2026' → '0118_delibera_n._16-2026.pdf'
    filenames_set = set(filenames)
    parent_to_file = {}
    for prefix in needed_parents:
        candidate = f"{prefix}.pdf"
        if candidate in filenames_set:
            parent_to_file[prefix] = candidate
    
    print(f"  [parent_index] padri presenti nel corpus: {len(parent_to_file)} su {len(needed_parents)}")
    
    if client is None:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    parent_index = {}
    
    # Caso A: padre presente nel corpus → estrazione completa
    for prefix, parent_file in parent_to_file.items():
        try:
            parent_path = Path(data_dir) / parent_file
            blocks = parse_pdf_fn(str(parent_path))
            
            if not blocks:
                # Parser non ha estratto nulla, fallback solo sul nome
                meta = extract_metadata(parent_file, "", client)
            else:
                header = blocks[0].get("document_header", "")
                meta = extract_metadata(parent_file, header, client)
            
            parent_index[prefix] = {
                "data_atto": meta["data_atto"],
                "anno": meta["anno"],
                "data_precisione": meta["data_precisione"],
            }
        except Exception as e:
            print(f"  [parent_index] errore su padre {parent_file}: {e}")
            # Fallback: estrai dal nome
            meta = extract_metadata(parent_file, "", client)
            parent_index[prefix] = {
                "data_atto": meta["data_atto"],
                "anno": meta["anno"],
                "data_precisione": meta["data_precisione"],
            }
    
    # Caso B: padre assente dal corpus → estrazione solo dal nome
    missing_parents = needed_parents - set(parent_to_file.keys())
    for prefix in missing_parents:
        # Trattiamo il prefisso come un nome file fittizio per riusare le regex
        pseudo_filename = f"{prefix}.pdf"
        meta = extract_metadata(pseudo_filename, "", client)
        parent_index[prefix] = {
            "data_atto": meta["data_atto"],
            "anno": meta["anno"],
            "data_precisione": meta["data_precisione"],
        }
    
    # Statistiche
    with_data = sum(1 for v in parent_index.values() if v["data_atto"])
    print(f"  [parent_index] {with_data}/{len(parent_index)} padri con data al giorno")
    
    return parent_index