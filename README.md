# Open Pertosa

Sistema RAG (Retrieval-Augmented Generation) per la consultazione in linguaggio naturale dei documenti ufficiali del Comune di Pertosa (SA, Italia).

Il progetto nasce come iniziativa civica personale con l'obiettivo di rendere il patrimonio documentale della pubblica amministrazione — bilanci, delibere, determine, regolamenti — accessibile a qualsiasi cittadino senza competenze tecniche o giuridiche.

---

## Il problema

I comuni italiani pubblicano per legge centinaia di documenti all'anno sull'Albo Pretorio e su Amministrazione Trasparente. Questi documenti sono formalmente accessibili ma praticamente inaccessibili: sono PDF non indicizzati, spesso scansionati, con un linguaggio tecnico-amministrativo che scoraggia la consultazione autonoma da parte dei cittadini.

Open Pertosa risolve questo problema trasformando l'archivio documentale in una base di conoscenza interrogabile in linguaggio naturale, con risposte citate e verificabili.

---

## Come funziona

```
Domanda del cittadino (italiano)
           ↓
      FastAPI backend
           ↓
  Ricerca semantica su Qdrant
           ↓
  Top-k chunk più rilevanti
           ↓
  GPT-4o mini (OpenAI) — risposta in streaming
           ↓
  Risposta con citazione delle fonti
```

Il sistema è composto da due pipeline indipendenti:

**Pipeline di ingestion (offline, locale)**
I documenti PDF vengono estratti, suddivisi in chunk semantici, vettorizzati e caricati su Qdrant. Questa operazione avviene una volta per documento, offline, sulla macchina dello sviluppatore.

**Pipeline di inferenza (online, server)**
Ad ogni domanda del cittadino, la query viene vettorizzata, confrontata con i chunk indicizzati, e i più rilevanti vengono passati al modello linguistico che genera la risposta in streaming.

---

## Stack tecnologico

| Componente | Tecnologia |
|---|---|
| Backend API | FastAPI + Uvicorn |
| Vector database | Qdrant (self-hosted, Docker) |
| Parsing PDF | PyMuPDF + PyMuPDF4LLM |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI `gpt-4o-mini` (streaming SSE) |
| Frontend | HTML/CSS/JS vanilla (single file) |
| Reverse proxy | Nginx |
| Process manager | systemd |
| Infrastruttura | Hetzner Cloud CPX21, Germania (EU) |

**Target produzione:** Azure OpenAI EU (Sweden Central) per conformità GDPR con data residency europea.

---

## Ambienti di deployment

### Prototipo (attuale)

| Componente | Configurazione |
|---|---|
| Server | Hetzner CPX21 — 3 vCPU, 4GB RAM, 80GB SSD, Germania |
| LLM | OpenAI API diretta (`gpt-4o-mini`) |
| Embeddings | OpenAI API diretta (`text-embedding-3-small`) |
| Ingestion | Offline sulla macchina dello sviluppatore |
| HTTPS | Non attivo — accesso via IP pubblico |
| Dominio | Non configurato |

### Produzione (target)

| Componente | Configurazione |
|---|---|
| Server | Hetzner CPX31 — 4 vCPU, 8GB RAM, 160GB SSD, Germania |
| LLM | Azure OpenAI EU — `gpt-4o-mini`, region Sweden Central |
| Embeddings | Azure OpenAI EU — `text-embedding-3-small`, region Sweden Central |
| GDPR | Data Processing Agreement firmato con Microsoft, EU Data Boundary attivo |
| HTTPS | Let's Encrypt via Certbot |
| Dominio | `assistente.comune.pertosa.sa.it` |
| Ingestion | Semi-automatizzata — monitoraggio albo pretorio |

La migrazione da prototipo a produzione richiede principalmente l'aggiornamento delle credenziali API da OpenAI a Azure OpenAI EU e la configurazione del dominio con HTTPS. L'architettura applicativa rimane invariata.

---

## Struttura del progetto

```
pertosa-rag/
├── src/
│   ├── api.py                       # FastAPI — endpoint RAG con streaming
│   ├── ingestion/
│   │   ├── parser.py                # Estrazione testo PDF con fallback OCR
│   │   ├── chunker.py               # Chunking con overlap semantico
│   │   ├── vectorizer.py            # Embedding e upsert su Qdrant
│   │   └── run_ingestion.py         # Entrypoint pipeline di ingestion
│   ├── retrieval/
│   │   └── retriever.py             # Ricerca semantica su Qdrant
│   └── frontend/
│       ├── index.html               # Interfaccia chat (streaming, markdown)
│       └── cos-e-open-pertosa.html  # Pagina informativa per i cittadini
├── requirements.txt
└── .gitignore
```

---

## Decisioni di progettazione

### Parser PDF a strategia progressiva

I documenti amministrativi italiani presentano una varietà di formati difficile da gestire con un approccio unico: PDF nativi con testo selezionabile, PDF con OCR incorporato dallo scanner (spesso con artefatti come lettere spezzate `d i` invece di `di`), e PDF puramente scansionati senza testo incorporato.

La soluzione adottata applica una strategia per pagina:

1. **Estrazione nativa con PyMuPDF** — veloce, zero dipendenze esterne
2. **Rilevamento e pulizia artefatti OCR** — regex calibrate su testi amministrativi italiani, attivata solo se necessario
3. **Fallback PyMuPDF4LLM con OCR** — applicato solo alle pagine che restituiscono zero caratteri con l'estrazione nativa

Questo approccio evita di applicare OCR su documenti che non ne hanno bisogno, riducendo i tempi di elaborazione e il consumo di risorse.

### Chunking con chunk_size adattivo

Il chunking a dimensione fissa standard (800 caratteri) produceva chunk che spezzavano liste di requisiti e articoli di legge in punti arbitrari, compromettendo la qualità del retrieval. Aumentando `chunk_size` a 1500 caratteri con overlap di 150, i blocchi logici dei documenti amministrativi (elenchi, articoli, dispositivi di delibera) rimangono coerenti all'interno dello stesso chunk.

### Streaming SSE con typewriter adattivo

Le risposte vengono trasmesse token per token via Server-Sent Events. Il frontend implementa un buffer di caratteri con un loop a 60fps che regola dinamicamente la velocità di rendering in base alla dimensione del buffer — più lento quando il buffer è piccolo, più veloce quando si accumula un arretrato. Il risultato è uno scorrimento visivamente fluido indipendentemente dalla latenza irregolare del modello.

### Memoria della conversazione

La cronologia della conversazione viene mantenuta in memoria nel browser (non persistita) e inviata ad ogni richiesta API. Il system prompt istruisce il modello a distinguere tra domande fattuali (rispondi solo dai documenti) e domande di follow-up (puoi usare il contesto della conversazione). La cronologia è limitata agli ultimi 20 messaggi per contenere la dimensione del contesto.

---

## Setup locale

### Prerequisiti

- Python 3.11+
- Docker
- API key OpenAI

### Installazione

```bash
git clone https://github.com/marinoalfonso/pertosa-rag.git
cd pertosa-rag

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configurazione

Crea un file `.env` nella root del progetto:

```env
OPENAI_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

### Avvio

```bash
# Avvia Qdrant
docker run -d --name qdrant \
  -p 6333:6333 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# Aggiungi i PDF in data/raw/ e indicizza
cd src/ingestion
python run_ingestion.py

# Avvia il server API
cd ../..
uvicorn src.api:app --reload --port 8000

# Apri il frontend
open src/frontend/index.html
```

---

## Deploy su Hetzner

```bash
# Dipendenze di sistema
apt update && apt upgrade -y
apt install -y python3-pip python3-venv nginx
curl -fsSL https://get.docker.com | sh

# Setup progetto
git clone https://github.com/marinoalfonso/pertosa-rag.git /opt/pertosa-rag
cd /opt/pertosa-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Qdrant
docker run -d --name qdrant --restart always \
  -p 6333:6333 \
  -v /opt/pertosa-rag/qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# Systemd service per uvicorn
# Configura /etc/systemd/system/pertosa-rag.service
# Configura /etc/nginx/sites-available/pertosa-rag
```

---

## Limitazioni note

| Problema | Stato |
|---|---|
| Sensibilità alla formulazione della domanda | Retrieval semantico non sempre robusto su query molto specifiche — riformulare in termini più generali risolve nella maggior parte dei casi |
| Tabelle presenze con segni grafici (X) | Non risolto — i segni sono pixel, non testo |
| Data residency EU (GDPR formale) | Pianificato — migrazione Azure OpenAI EU all'adozione ufficiale |

---

## Roadmap

- [ ] Migrazione Azure OpenAI EU
- [ ] Pipeline di ingestion automatizzata (monitoraggio albo pretorio)
- [ ] Chunking semantico sui confini logici del documento
- [ ] Pannello amministrativo per la gestione dei documenti
- [ ] HTTPS e dominio istituzionale

---

## Licenza

Rilasciato sotto licenza **GNU Affero General Public License v3.0 (AGPL-3.0)**, coerente con la licenza della dipendenza PyMuPDF.

---

## Autore

**Alfonso Marino**
[github.com/marinoalfonso](https://github.com/marinoalfonso)
