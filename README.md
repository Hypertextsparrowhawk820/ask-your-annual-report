# 📊 Ask Your Annual Report

> A hybrid AI-powered financial analyst chatbot that answers questions from **Apple, Microsoft & Tesla's 2025 SEC 10-K filings** — and searches the web for everything else.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.2.16-green?logo=chainlink&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.37-red?logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-purple)
![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-orange)
![Tavily](https://img.shields.io/badge/Tavily-Web%20Search-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## What This Project Does

Most RAG demos just answer from documents. This one is a **fully hybrid conversational AI** — it routes every user message to the right intelligence source automatically.

| User asks | Bot does |
|---|---|
| *"What was Apple's revenue in 2025?"* | Searches 10-K filings via FAISS → answers with page citations |
| *"Compare Microsoft and Tesla's AI strategy"* | Retrieves from each company separately → structured comparison |
| *"Who is the Prime Minister of India?"* | Routes to Tavily web search → grounded answer with live sources |
| *"Hello! How are you?"* | Responds conversationally like a friendly AI assistant |
| *"haii "* | Matches the energy — casual, warm response |

**Key result:** Correctly answered 16/20 domain-specific financial questions with accurate page-level citations from SEC filings.

---

## System Architecture

### High-level routing

```
User Question
      │
      ▼
┌─────────────────────────────────┐
│   Intent Router (Llama 3.3 70B) │
└─────────────────────────────────┘
      │
      ├── casual ──────────► Conversational LLM (warm, natural reply)
      │
      ├── financial ───────► FAISS retriever → 10-K chunks → LLM → cited answer
      │
      └── general ─────────► Tavily web search → LLM → sourced answer
```

### Offline indexing pipeline (runs once)

```
SEC 10-K PDFs (Apple, Microsoft, Tesla)
      │
      ▼
PyMuPDF text extractor (page by page)
      │
      ▼
RecursiveCharacterTextSplitter (chunk_size=800, overlap=100)
      │
      ▼
Metadata tagger  →  { company, ticker, year, page, source }
      │
      ▼
HuggingFace Embeddings (all-MiniLM-L6-v2, ~80MB, runs on CPU)
      │
      ▼
FAISS index  →  saved to disk  →  loaded at app startup
```

### Live query pipeline

```
User question
      │
      ▼
Query embedded (same model as index)
      │
      ▼
FAISS similarity search  →  top-5 chunks (optionally filtered by company)
      │
      ▼
Chunks + metadata injected into prompt
      │
      ▼
Llama 3.3 70B (via Groq)  →  cited answer with company + page references
      │
      ▼
Streamlit UI renders answer + collapsible source expander
```

---

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| LLM | Llama 3.3 70B via Groq API | Fast, free tier, state-of-the-art |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Lightweight, runs on CPU, no API key |
| Vector store | FAISS (local, CPU) | Free, fast, no cloud setup needed |
| Orchestration | LangChain 0.2.16 | RetrievalQA + metadata filtering |
| Web search | Tavily API | Clean, developer-friendly, free tier |
| PDF parsing | PyMuPDF (fitz) | Best text extraction for financial PDFs |
| Frontend | Streamlit | Rapid deployment, clean chat UI |
| Data source | SEC EDGAR | Free public financial filings |

---

## Project Structure

```
ask-your-annual-report/
│
├── src/
│   ├── __init__.py
│   ├── ingest.py            # Download/verify 10-K PDFs from SEC EDGAR
│   ├── parse_and_chunk.py   # PDF → clean text chunks with metadata
│   ├── build_index.py       # Embed all chunks → FAISS index on disk
│   ├── rag_chain.py         # LangChain RAG chain with citation prompt
│   ├── retriever.py         # FAISS retriever + multi-company filter
│   └── router.py            # Intent classifier + Tavily web search
│
├── app/
│   ├── __init__.py
│   └── app.py               # Streamlit chat UI (full hybrid chatbot)
│
├── data/
│   └── sec_filings/         # 10-K PDFs live here (gitignored — see setup)
│       ├── AAPL_10K_2025.pdf
│       ├── MSFT_10K_2025.pdf
│       └── TSLA_10K_2025.pdf
│
├── vectorstore/             # FAISS index files (gitignored — regenerate locally)
│   ├── index.faiss
│   └── index.pkl
│
├── .env                     # Your API keys (gitignored)
├── .env.example             # Safe template to commit
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.11 (recommended — best package compatibility for ML)
- Git

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ask-your-annual-report.git
cd ask-your-annual-report
```

### 2. Create a virtual environment

**Windows (PowerShell):**
```powershell
pip install virtualenv
virtualenv venv
venv\Scripts\Activate.ps1
```

**Mac / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get free API keys

You need two free API keys — both take under 2 minutes to get:

| Service | URL | Free tier |
|---|---|---|
| Groq (LLM) | [console.groq.com](https://console.groq.com) | Unlimited (rate limited) |
| Tavily (web search) | [tavily.com](https://tavily.com) | 1,000 searches/month |

### 5. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxxxxxx
```

### 6. Download the 10-K filings

The PDF files are not included in this repo (too large for GitHub). Download them manually from SEC EDGAR:

**For each company:**
1. Go to [SEC EDGAR filing search](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=10-K&dateb=&owner=include&count=5)
2. Enter the ticker → select the most recent 10-K
3. Open the primary document (`.htm` file)
4. `Ctrl + P` → **Save as PDF**
5. Save to `data/sec_filings/` with the exact filename below

| Company | Ticker | Save as |
|---|---|---|
| Apple | AAPL | `AAPL_10K_2025.pdf` |
| Microsoft | MSFT | `MSFT_10K_2025.pdf` |
| Tesla | TSLA | `TSLA_10K_2025.pdf` |

**Direct links:**
- Apple: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=AAPL&type=10-K
- Microsoft: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=MSFT&type=10-K
- Tesla: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=TSLA&type=10-K

### 7. Build the FAISS vector index

```bash
cd src
python build_index.py
```

**What this does:**
- Extracts text from all 3 PDFs (~1,400 chunks total)
- Downloads the embedding model (~80MB, first run only — cached after)
- Embeds all chunks and saves the FAISS index to `vectorstore/`
- Takes 2–3 minutes on first run

Expected output:
```
Processing AAPL_10K_2025.pdf... Created 308 chunks
Processing MSFT_10K_2025.pdf... Created 511 chunks
Processing TSLA_10K_2025.pdf... Created 584 chunks
Total chunks: 1403
FAISS index saved to: .../vectorstore
```

### 8. Launch the app

```bash
cd ..
streamlit run app/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Example Questions

**Financial — answered from 10-K filings with citations:**
```
What are Tesla's main risk factors?
What was Apple's total revenue in 2025?
How does Microsoft describe its AI strategy?
Compare Apple and Microsoft's approach to cloud services
What does Tesla say about competition in the EV market?
Compare Apple and Tesla's supply chain risks
What is Microsoft's employee headcount?
How does Apple describe its approach to privacy?
```

**General — answered from live web search:**
```
Who is the Prime Minister of India?
How does retrieval augmented generation work?
What is the latest news in artificial intelligence?
What is the current USD to EUR exchange rate?
Who won the last FIFA World Cup?
```

**Casual — handled conversationally:**
```
Hello! / Hi / Hey / Haii
How are you?
What can you do?
Who are you?
Thanks!
```

---

## Key Features

### Smart intent routing
Every message is classified by Llama 3.3 70B before processing:
- `casual` → warm, conversational LLM response
- `financial` → RAG pipeline with 10-K filings
- `general` → Tavily live web search

### Page-level citations
Every financial answer includes exact citations: company name, page number, and source file. Users always know exactly where the information came from.

### Multi-company comparison mode
Enable "Comparison mode" in the sidebar to retrieve context from each selected company separately and get a structured side-by-side answer.

### Live web search fallback
General knowledge questions are answered using Tavily web search — so the bot never says "I don't know" for out-of-domain questions.

### Conversational personality
Handles informal messages (greetings, thanks, casual chat) naturally — not robotically redirecting every message to the documents.

### Zero hallucination design for financials
Financial answers are strictly grounded in filing context. The LLM is explicitly instructed to say "I don't have enough information" rather than guess.

### Free infrastructure
Entirely free to run — Groq free tier + Tavily free tier + local FAISS + local embeddings.

---

## Evaluation Results

Manually evaluated on 20 domain-specific questions across all three companies:

| Category | Questions | Correct | Score |
|---|---|---|---|
| Revenue & financial figures | 5 | 5 | 100% |
| Risk factors | 5 | 4 | 80% |
| Strategy & AI investments | 5 | 4 | 80% |
| Cross-company comparisons | 5 | 3 | 60% |
| **Overall** | **20** | **16** | **80%** |

**Why cross-company comparisons score lower:** When the relevant context for two companies appears in different sections of different documents, retrieval sometimes misses one side. This is improved by enabling "Comparison mode" which retrieves from each company separately.

---

## Known Limitations

| Limitation | Root cause | Workaround |
|---|---|---|
| Financial tables parsed as raw text | PDF text extraction flattens grids | Accept minor formatting issues |
| Cross-year comparisons need both PDFs | Only one year per company indexed | Add more filing years |
| Tavily 1,000 req/month on free tier | API rate limit | Upgrade or cache results |
| FAISS runs on CPU | No GPU inference | Fast enough for demo purposes |
| No memory across sessions | Streamlit stateless by default | Chat history within session only |

---

## Roadmap / Future Improvements

- [ ] Add more companies — Amazon, Google, Meta 10-K filings
- [ ] Structured table extraction for precise financial data
- [ ] Deploy on Streamlit Community Cloud (live shareable URL)
- [ ] RAGAS automated evaluation pipeline
- [ ] Support uploading custom PDFs at runtime
- [ ] Chart generation from extracted financial figures
- [ ] Add conversation memory across sessions
- [ ] Docker containerization for easy deployment

---

## Troubleshooting

### `faiss-cpu` install fails
```bash
pip install faiss-cpu --only-binary=:all:
```

### PyMuPDF install fails on Python 3.13
```bash
pip install pymupdf==1.25.5
```

### `langchain_core.pydantic_v1` error
Your LangChain versions are mismatched. Run:
```bash
pip uninstall langchain langchain-core langchain-community langchain-groq -y
pip install langchain==0.2.16 langchain-core==0.2.40 langchain-community==0.2.16 langchain-groq==0.1.9
```

### Groq model decommissioned error
Update the model name in `src/rag_chain.py` and `app/app.py`:
```python
model="llama-3.3-70b-versatile"   # current as of 2025
```
Check latest models at [console.groq.com/docs/models](https://console.groq.com/docs/models)

### SEC EDGAR returns 403 error
The `sec-edgar-downloader` library requires a valid user-agent. Use version 4.0.0:
```bash
pip install sec-edgar-downloader==4.0.0
```
Or download PDFs manually from SEC EDGAR (recommended for clean results).

---

## Requirements

Full `requirements.txt`:

```
sec-edgar-downloader==4.0.0
pymupdf==1.25.5
langchain==0.2.16
langchain-core==0.2.40
langchain-community==0.2.16
langchain-groq==0.1.9
langchain-huggingface==0.0.3
sentence-transformers==3.0.1
faiss-cpu==1.9.0.post1
tavily-python
streamlit==1.37.1
python-dotenv==1.0.1
pandas==2.2.2
tqdm==4.66.5
```

---

## How to Add More Companies

1. Download the company's 10-K PDF from SEC EDGAR
2. Save to `data/sec_filings/TICKER_10K_YEAR.pdf` (e.g. `AMZN_10K_2025.pdf`)
3. Add the ticker to `COMPANY_MAP` in `src/parse_and_chunk.py`:
   ```python
   COMPANY_MAP = {
       "AAPL": "Apple",
       "MSFT": "Microsoft",
       "TSLA": "Tesla",
       "AMZN": "Amazon"   # ← add here
   }
   ```
4. Add to `COMPANIES` list in `app/app.py`:
   ```python
   COMPANIES = ["Apple", "Microsoft", "Tesla", "Amazon"]
   ```
5. Rebuild the index:
   ```bash
   cd src && python build_index.py
   ```

---

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/add-amazon`)
3. Commit your changes (`git commit -m 'Add Amazon 10-K support'`)
4. Push to the branch (`git push origin feature/add-amazon`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License — free to use, modify, and distribute.

---

## Acknowledgements

- [LangChain](https://langchain.com) — RAG orchestration framework
- [Groq](https://groq.com) — blazing fast free LLM inference
- [Tavily](https://tavily.com) — web search API for developers
- [SEC EDGAR](https://www.sec.gov/edgar) — free public financial filings
- [Hugging Face](https://huggingface.co) — sentence-transformers embedding model
- [Meta AI](https://ai.meta.com) — Llama 3.3 open source LLM

---

## Author

Built as part of a data science portfolio project — Tier 4 GenAI/LLM project in the ML project roadmap.

If you found this useful, please star the repo!
