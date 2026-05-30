# Open Pertosa

Sistema RAG (Retrieval-Augmented Generation) per la consultazione in linguaggio naturale dei documenti ufficiali del Comune di Pertosa (SA, Italia).

Il progetto nasce come iniziativa civica personale con l'obiettivo di rendere il patrimonio documentale della pubblica amministrazione — bilanci, delibere, determine, regolamenti — accessibile a qualsiasi cittadino senza competenze tecniche o giuridiche.

![Interfaccia di Open Pertosa](assets/interfaccia.png)

---

## Il problema

I comuni italiani pubblicano per legge centinaia di documenti all'anno sull'Albo Pretorio e su Amministrazione Trasparente. Questi documenti sono formalmente accessibili ma praticamente inaccessibili: sono PDF non indicizzati, spesso con tabelle complesse o scansionati, con un linguaggio tecnico-amministrativo che scoraggia la consultazione autonoma da parte dei cittadini.

Open Pertosa risolve questo problema trasformando l'archivio documentale in una base di conoscenza interrogabile in linguaggio naturale, con risposte citate e verificabili.

---

## Come funziona

```
Domanda del cittadino (italiano)
           ↓
      FastAPI backend
           ↓
  Hybrid search su Qdrant
  (denso OpenAI + sparso BM25)
           ↓
  Fusione RRF + diversificazione per documento
           ↓
  Top-k chunk più rilevanti
           ↓
  GPT-4o mini (OpenAI) — risposta in streaming
           ↓
  Risposta con citazione delle fonti
```

Il sistema è composto da due pipeline indipendenti.

**Pipeline di ingestion (offline, locale)**
I documenti PDF attraversano cinque stadi: estrazione in blocchi tipizzati, normalizzazione (con propagazione delle intestazioni tra tabelle multi-pagina), chunking consapevole del tipo di contenuto, vettorizzazione duale (denso semantico + sparso BM25), e indicizzazione su Qdrant. Questa operazione avviene una volta per documento, offline, sulla macchina dello sviluppatore.

**Pipeline di inferenza (online, server)**
Ad ogni domanda del cittadino, la query viene vettorizzata in due forme (denso e sparso), confrontata con i chunk indicizzati tramite ricerca ibrida, fusa con Reciprocal Rank Fusion lato server e diversificata per documento. I chunk più rilevanti vengono passati al modello linguistico che genera la risposta in streaming.

---

## Stack tecnologico

| Componente | Tecnologia |
|---|---|
| Backend API | FastAPI + Uvicorn |
| Vector database | Qdrant (self-hosted, Docker) |
| Parsing PDF | PyMuPDF + PyMuPDF4LLM (tabelle in Markdown) |
| Embedding denso | OpenAI `text-embedding-3-small` |
| Embedding sparso | BM25 italiano via FastEmbed (locale) |
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
| Embedding denso | OpenAI API diretta (`text-embedding-3-small`) |
| Embedding sparso | BM25 via FastEmbed (locale, on-server) |
| Ingestion | Offline sulla macchina dello sviluppatore |
| HTTPS | Non attivo — accesso via IP pubblico |
| Dominio | Non configurato |

### Produzione (target)

| Componente | Configurazione |
|---|---|
| Server | Hetzner CPX31 — 4 vCPU, 8GB RAM, 160GB SSD, Germania |
| LLM | Azure OpenAI EU — `gpt-4o-mini`, region Sweden Central |
| Embedding denso | Azure OpenAI EU — `text-embedding-3-small`, region Sweden Central |
| Embedding sparso | BM25 via FastEmbed (locale, invariato) |
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
│   ├── api.py                       # FastAPI — endpoint RAG con streaming + metriche Prometheus
│   ├── ingestion/
│   │   ├── parser.py                # Stadio 1 — estrazione PDF in blocchi tipizzati (paragraph/table)
│   │   ├── normalizer.py            # Stadio 2 — propagazione header tra tabelle multi-pagina
│   │   ├── chunker.py               # Stadio 3 — chunking consapevole del tipo, con header propagation
│   │   ├── vectorizer.py            # Stadio 4-5 — embedding duale (denso + BM25) e upsert su Qdrant
│   │   └── run_ingestion.py         # Orchestrazione della pipeline di ingestion
│   ├── retrieval/
│   │   └── retriever.py             # Hybrid search (BM25 + denso) con RRF + diversificazione MMR
│   └── frontend/
│       ├── index.html               # Interfaccia chat (streaming, markdown, typewriter)
│       ├── cos-e-open-pertosa.html  # Pagina informativa per i cittadini
│       └── logo-pertosa.png         # Logo ufficiale del Comune di Pertosa
├── monitoring/
│   ├── docker-compose.yml           # Stack Prometheus + Grafana + Node Exporter
│   └── prometheus.yml               # Configurazione scrape targets
├── deployment/
│   └── nginx.conf                   # Configurazione Nginx — reverse proxy
├── LICENSE                          # GNU AGPL-3.0
├── requirements.txt
└── .gitignore
```

---

## Decisioni di progettazione

### Pipeline di ingestion a stadi

L'ingestion è organizzata in cinque stadi indipendenti che comunicano attraverso un modello-dati canonico (blocchi tipizzati). Ogni stadio ha un unico compito ben definito e non conosce l'implementazione degli altri:

1. **Parser** — trasforma il PDF in una lista di blocchi `{type, content, page, source, header}` dove `type` è `paragraph` o `table`. La decisione tabella-vs-paragrafo è presa una sola volta, qui, ed è codificata nel tipo del blocco.
2. **Normalizer** — applica trasformazioni che operano sui blocchi, come la propagazione dell'intestazione tra tabelle multi-pagina adiacenti.
3. **Chunker** — applica strategie di chunking diverse per tipo di blocco.
4. **Vectorizer** — calcola due rappresentazioni vettoriali per ogni chunk (denso semantico, sparso BM25).
5. **Indexer** — fa l'upsert su Qdrant in batch.

Questa separazione rende ogni componente sostituibile: cambiare estrattore PDF, modello di embedding o strategia di chunking richiede di toccare un solo stadio, lasciando gli altri invariati.

### Parser PDF con riconoscimento tabellare

I documenti amministrativi italiani presentano una varietà di formati difficile da gestire con un approccio unico: PDF nativi con testo selezionabile, PDF con tabelle complesse a celle unite, e PDF scansionati. La soluzione adottata applica una strategia per pagina:

- **Estrazione nativa con PyMuPDF** per pagine di testo discorsivo
- **Estrazione Markdown con PyMuPDF4LLM** per pagine con tabelle riconosciute (filtro minimo 2×2 celle effettive per evitare falsi positivi)
- **Fallback OCR via PyMuPDF4LLM** per pagine senza testo selezionabile

L'output è una lista di blocchi tipizzati: i blocchi-tabella conservano la struttura markdown con l'intestazione isolata e disponibile per le trasformazioni successive.

### Chunking table-aware con header propagation

Le tabelle dei documenti amministrativi (bilanci, programmi triennali, elenchi) vengono spezzate su confini di riga e l'intestazione viene replicata in ogni chunk derivato. Questo garantisce che ogni frammento di tabella resti interpretabile da solo: le colonne non diventano mai anonime e il modello sa sempre cosa sta leggendo, indipendentemente da dove la tabella sia stata tagliata.

Per le tabelle che proseguono su pagine successive senza ripetere l'intestazione, lo stadio di normalizzazione "presta" l'header dalla tabella precedente (se la struttura delle colonne combacia), senza fondere i blocchi: ogni chunk resta legato alla sua pagina reale, le citazioni restano corrette.

### Hybrid search con fusione RRF

Il retrieval combina due meccanismi complementari, fusi server-side con Reciprocal Rank Fusion:

- **Denso (OpenAI)** — cattura la vicinanza semantica, robusto su parafrasi e sinonimia
- **Sparso (BM25 italiano)** — pesa i termini per rarità (IDF lato server), efficace su nomi propri, codici, identificatori specifici

La combinazione è particolarmente importante in un corpus monocomunale dove termini come "Pertosa" perdono valore discriminativo (sono ovunque) e dove le entità rare (nomi di consiglieri, indirizzi, codici intervento) hanno alto valore informativo che il solo denso ignora.

### Diversificazione per documento

I documenti amministrativi più lunghi (DUPS, allegati di bilancio, programmi pluriennali) producono molti chunk e tendono a monopolizzare i risultati del retrieval. Una funzione di diversificazione applicata dopo la fusione RRF limita il numero massimo di chunk per documento nei risultati finali, garantendo che il contesto passato al modello rappresenti più fonti diverse anziché molte pagine di un singolo documento.

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
git clone https://github.com/marinoalfonso/open-pertosa.git
cd open-pertosa

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
# Avvia Qdrant (richiede versione 1.10+ per la Query API)
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

Alla prima esecuzione, FastEmbed scarica il modello BM25 italiano (pochi MB, una sola volta).

---

## Deploy su Hetzner

```bash
# Dipendenze di sistema
apt update && apt upgrade -y
apt install -y python3-pip python3-venv nginx
curl -fsSL https://get.docker.com | sh

# Setup progetto
git clone https://github.com/marinoalfonso/open-pertosa.git /opt/pertosa-rag
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

## Monitoring

Il sistema include uno stack di monitoring basato su Prometheus e Grafana.

### Componenti

| Componente | Ruolo |
|---|---|
| Prometheus | Raccolta metriche — scrape ogni 15 secondi, retention 30 giorni |
| Grafana | Visualizzazione dashboard |
| Node Exporter | Metriche di sistema — CPU, RAM, disco |
| prometheus-fastapi-instrumentator | Metriche HTTP esposte da FastAPI su `/metrics` |

### Avvio

```bash
cd monitoring
docker compose up -d
```

### Accesso a Grafana

Grafana non è esposto pubblicamente. Accedi tramite tunnel SSH:

```bash
ssh -L 3000:localhost:3000 root@<server-ip>
```

Poi apri `http://localhost:3000` nel browser.

Credenziali: `admin` / password definita in `monitoring/.env`

---

## Strumenti diagnostici

Due script di sola lettura per ispezionare lo stato dell'indice durante lo sviluppo:

```bash
# Panoramica della collezione: distribuzione per documento, lunghezze, grep
python src/ingestion/inspect_qdrant.py
python src/ingestion/inspect_qdrant.py --grep sindaco

# Dump completo dei chunk con testo integrale
python src/ingestion/dump_chunks.py --doc bilancio --limit 30
```

---

## Limitazioni note

| Problema | Stato |
|---|---|
| Query generiche su contenuti tabellari complessi | Parziale — quando il vocabolario della domanda non si allinea con i termini delle tabelle (es. "interventi 2028" vs i codici CUI/CUP nelle tabelle del programma triennale), il retrieval può recuperare pagine introduttive descrittive invece delle tabelle dati. Mitigato dall'hybrid search; rimane un caso aperto per query molto astratte. |
| Aggregazione temporale | Limite strutturale — il sistema non ha una nozione esplicita di "adesso" o di stato avanzamento di un lavoro. Le risposte su "cosa è in corso" sono inferite dal modello leggendo il linguaggio dei documenti recuperati, con qualità variabile a seconda di quali chunk entrano nel contesto. |
| Sensibilità alla formulazione | Migliorata significativamente con l'hybrid search (BM25 neutralizza il rumore di termini frequenti nel corpus), ma riformulare domande molto astratte in termini più specifici resta utile. |
| Data residency EU (GDPR formale) | Pianificato — migrazione Azure OpenAI EU all'adozione ufficiale. |

---

## Roadmap

- [ ] Costruzione di un golden set strutturato per valutazione oggettiva
- [ ] Migrazione Azure OpenAI EU (denso + LLM)
- [ ] Pipeline di ingestion automatizzata (monitoraggio albo pretorio)
- [ ] Pulizia mirata del corpus (esclusione documenti tecnico-contabili non rilevanti per i cittadini)
- [ ] Valutazione reranking con cross-encoder per query difficili
- [ ] Pannello amministrativo per la gestione dei documenti
- [ ] HTTPS e dominio istituzionale

---

## Licenza

Rilasciato sotto licenza **GNU Affero General Public License v3.0 (AGPL-3.0)**, coerente con la licenza della dipendenza PyMuPDF.

---

## Autore

**Alfonso Marino**
[github.com/marinoalfonso](https://github.com/marinoalfonso)