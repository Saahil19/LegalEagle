# 🦅 LegalEagle — Complete Interview Preparation Guide

> This document covers every concept, design decision, architecture detail, and interview question for the LegalEagle project. Work through it section by section.

---

## 📋 TABLE OF CONTENTS

### PART A — Project Fundamentals
- [A1. What is LegalEagle? (Project + Goal + Problem Statement)](#a1)
- [A2. Full Technology Stack — Every Library & Why](#a2)

### PART B — Data Layer
- [B1. The CUAD Dataset — What it is, how it is structured](#b1)
- [B2. QA → NER Conversion (BIO Tagging) — Why and How](#b2)

### PART C — ML Core (BERT NER)
- [C1. BERT Architecture Deep Dive](#c1)
- [C2. Fine-Tuning for Token Classification](#c2)
- [C3. Sliding Window Inference — Why & How](#c3)
- [C4. Evaluation Metrics — seqeval, F1, Precision, Recall](#c4)

### PART D — Vector Search & RAG
- [D1. What is an Embedding? (all-MiniLM-L6-v2 deep dive)](#d1)
- [D2. ChromaDB — Storage, indexing, cosine similarity](#d2)
- [D3. Qdrant — How it differs, Docker setup, collection structure](#d3)
- [D4. RAG Pipeline — Full flow: chunk → embed → store → retrieve → generate](#d4)
- [D5. Text Chunking — RecursiveCharacterTextSplitter explained](#d5)

### PART E — GenAI Concepts
- [E1. flan-t5-base — Architecture, why chosen, prompt engineering](#e1)
- [E2. LangChain Tools & Agents](#e2)
- [E3. LangGraph — StateGraph, nodes, edges, conditional routing](#e3)
- [E4. Server-Sent Events (SSE) Streaming](#e4)

### PART F — Risk Scoring System
- [F1. How Risk Score is Calculated (1–10 scale)](#f1)
- [F2. If YOU give a new contract — How it will be judged (all metrics)](#f2)

### PART G — Backend Architecture
- [G1. FastAPI — All endpoints, request/response flow](#g1)
- [G2. SQLite + SQLAlchemy ORM — Tables, columns, why SQLite](#g2)
- [G3. Celery + Redis — Async task queue architecture](#g3)
- [G4. Lazy Model Loading — Why and how](#g4)

### PART H — Frontend Architecture
- [H1. Next.js App Router — Pages, routing, components](#h1)
- [H2. Frontend → Backend Request Flow (complete HTTP lifecycle)](#h2)
- [H3. EventSource / SSE on the frontend (Chat.tsx)](#h3)

### PART I — Model Optimization
- [I1. ONNX Export — What is ONNX, why export](#i1)
- [I2. INT8 Quantization — FP32 vs INT8, dynamic quantization](#i2)
- [I3. Benchmarks — Size, speed, accuracy tradeoffs](#i3)

### PART J — Complete System Architecture
- [J1. End-to-End Workflow Diagram](#j1)
- [J2. Data flow: file upload → NER → RAG → scoring → DB → frontend](#j2)

### PART K — Interview Questions
- [K1. Beginner-Level Questions](#k1)
- [K2. Intermediate-Level Questions](#k2)
- [K3. Advanced / Deep-Dive Questions](#k3)
- [K4. Trick / Gotcha Questions](#k4)

---

> Sections are being filled progressively below.

---

<a name="a1"></a>
## A1. What is LegalEagle? — Project, Goal & Problem Statement

### 🔴 The Real-World Problem

Legal contracts are dense, multi-page documents full of clauses that can cost companies millions if missed or misunderstood. Consider:

- A **non-compete clause** locking an employee out of their entire industry for 5 years worldwide.
- An **indemnification clause** that is entirely one-sided — the smaller party must cover ALL legal claims but receives no mutual protection.
- A **termination clause** allowing the other party to walk away with zero notice.

**The problem:** Manual contract review by a lawyer costs **$300–$1000/hour**. A single contract can take 5–20 hours. Startups, freelancers, and small businesses simply cannot afford this — so they sign without reading or understanding.

Even large legal teams miss clauses when reviewing hundreds of contracts under time pressure.

---

### 🎯 The Goal

**LegalEagle** is an end-to-end AI system that automates the first pass of legal contract review. It:

1. **Accepts** any raw contract (`.txt` or `.pdf`)
2. **Extracts** key legal entities and clause types using a fine-tuned BERT NER model
3. **Scores** each clause for risk (1–10) using RAG-augmented LLM reasoning
4. **Generates** a structured Markdown risk report with per-clause explanations
5. **Answers** free-form natural language questions about the contract via a chat interface

The system does NOT replace lawyers — it acts as an intelligent first-pass auditor that flags what a lawyer should focus on, saving 80–90% of review time.

---

### 🏗️ What Problem is Being Solved at Each Layer

| Layer | Problem Solved |
|-------|----------------|
| **BERT NER** | "Which parts of this contract are Termination clauses, Indemnification clauses, etc.?" |
| **RAG (Qdrant)** | "Is this clause normal compared to similar contracts in the real world?" |
| **LLM (flan-t5)** | "Given the clause text and market comparisons, how risky is this on a 1–10 scale?" |
| **FastAPI** | "How do we expose all of this AI as clean HTTP endpoints any frontend can call?" |
| **SQLite** | "How do we persist results so the user can come back and view them later?" |
| **Next.js** | "How does the user interact with all of this without writing any code?" |
| **SSE Streaming** | "How do we make the chat feel real-time, like ChatGPT?" |
| **ONNX INT8** | "How do we make BERT fast enough to run on a CPU server without a GPU?" |

---

### 📌 Interview One-Liner

> *"LegalEagle is a full-stack AI contract analysis system that uses a fine-tuned BERT NER model to extract legal clauses, a Qdrant vector database for market comparison via RAG, and a flan-t5 LLM to generate 1–10 risk scores, all served through a FastAPI backend with real-time SSE streaming chat, a Next.js frontend, and SQLite persistence."*

---

<a name="a2"></a>
## A2. Full Technology Stack — Every Library & Why

### 🧠 AI / ML Layer

| Library | Version | What it does in LegalEagle | Why this, not alternatives |
|---------|---------|---------------------------|---------------------------|
| **`transformers`** (HuggingFace) | latest | Loads `bert-base-uncased`, `AutoTokenizer`, `AutoModelForTokenClassification`; also wraps `flan-t5-base` via `pipeline()` | Industry standard for transformer models; supports both PyTorch and ONNX backends |
| **`torch` (PyTorch)** | latest | Tensor operations, `torch.no_grad()` for inference, `argmax` on logits | BERT is a PyTorch model; needed for the forward pass |
| **`sentence-transformers`** / **`langchain_huggingface.HuggingFaceEmbeddings`** | latest | Encodes text chunks into 384-dim vectors using `all-MiniLM-L6-v2` | Free, no API key, runs locally, 22M params (fast), Apache 2.0 license |
| **`seqeval`** | latest | Computes Precision, Recall, F1 for NER labels at the entity level | Standard NER evaluation library; understands BIO tag schemes correctly |
| **`optimum`** (HuggingFace Optimum) | latest | `ORTModelForTokenClassification` — exports BERT to ONNX and loads ONNX models with the same HF API | Bridges HuggingFace Transformers ↔ ONNX Runtime seamlessly |
| **`onnxruntime`** | latest | Runs the exported ONNX model on CPU; `quantize_dynamic()` compresses FP32 → INT8 | Hardware-agnostic runtime; 2.5× faster than PyTorch on CPU |
| **`datasets`** (HuggingFace) | latest | Loads the CUAD dataset from disk in Arrow format for efficient training | Native HuggingFace format; supports streaming large datasets |

---

### 🔗 LangChain / Orchestration Layer

| Library | What it does | Why chosen |
|---------|-------------|-----------|
| **`langchain-core`** | Base classes: `Document`, `BaseRetriever`, `PromptTemplate` | Foundation for all LangChain abstractions |
| **`langchain-community`** | `PyPDFLoader`, `TextLoader`, `Docx2txtLoader` | Unified interface to load any file format into `Document` objects |
| **`langchain-chroma`** | `Chroma.from_documents()` — wraps ChromaDB as a LangChain vector store | Easy local vector store for Notebook 2; no external service needed |
| **`langchain_huggingface`** | `HuggingFaceEmbeddings`, `HuggingFacePipeline` | Wraps local models behind LangChain's embedding/LLM interface |
| **`langchain` `@tool` decorator** | Converts a Python function into an agent-callable tool with auto schema | Agents can discover and call tools without hard-coding |
| **`langgraph`** | `StateGraph`, `TypedDict` state, nodes, edges, `add_conditional_edges` | Stateful multi-agent pipelines with conditional routing; cannot be done with plain LangChain chains |

---

### 🗃️ Vector Databases

| DB | Used in | Storage | Why |
|----|---------|---------|-----|
| **ChromaDB** | Notebook 2 | `data/chroma_db/` (SQLite-backed on disk) | Simple local setup, zero config, good for prototyping |
| **Qdrant** | Notebook 3, 4, 5, backend `tasks.py` | `data/qdrant_storage/` (Docker volume) or `:memory:` | Production-grade: server-side metadata filtering, REST API, web dashboard at `:6333/dashboard`, distributed cluster support |

---

### 🌐 Backend Layer

| Library | Role | Key Detail |
|---------|------|-----------|
| **`fastapi`** | Web framework — defines all HTTP routes | Async, auto-generates OpenAPI docs at `/docs`; uses Pydantic for schema validation |
| **`uvicorn`** | ASGI server that runs FastAPI | Handles async I/O; required because FastAPI is async |
| **`pydantic`** | Request/response body validation | `AnalyzeRequest(job_id: str)` — FastAPI reads body and validates type automatically |
| **`sqlalchemy`** | ORM — maps Python classes to SQLite tables | `AnalysisJob`, `QARecord` classes map to DB tables; handles sessions, queries |
| **`sqlite3`** (stdlib) | The actual database engine | Zero-config, single `.db` file, swappable with PostgreSQL via `DATABASE_URL` env var |
| **`celery`** | Async task queue — wraps `analyze_contract_core()` as a background task | In production: returns `202 Accepted` instantly; processes heavy AI work in background |
| **`redis`** | Celery's message broker and result backend | Broker: `redis://localhost:6379/0`; Results: `redis://localhost:6379/1` |
| **`python-multipart`** | Parses `multipart/form-data` uploads in FastAPI | Required for `UploadFile` to work |
| **`shutil`** (stdlib) | `shutil.copyfileobj()` streams uploaded file to disk | Memory-efficient: doesn't load entire file into RAM |
| **`uuid`** (stdlib) | `uuid.uuid4()` generates unique job IDs | Guarantees no collision between concurrent uploads |

---

### 🖥️ Frontend Layer

| Tech | Role | Key Detail |
|------|------|-----------|
| **Next.js 14** (App Router) | React framework with file-based routing | `app/page.tsx` = home, `app/report/[id]/page.tsx` = dynamic report page |
| **TypeScript** | Type-safe JavaScript | Catches errors at compile time; all components typed |
| **`EventSource` API** (browser built-in) | Connects to the SSE stream at `GET /ask/stream` | Native browser API; no library needed; auto-reconnects |
| **`fetch` API** (browser built-in) | `POST /upload`, `POST /analyze`, `GET /report/{id}` | Standard HTTP calls; used in `page.tsx` and `report/[id]/page.tsx` |
| **CSS Variables + Vanilla CSS** | Design system: colors, glass effect, animations | `globals.css` defines `--accent`, `--bg`, `--surface`, `--red-glow` etc. |
| **`localStorage`** | Auth state persistence across page refreshes | Stores `isAuthenticated` flag; checked on every page load |

---

### 🐳 Infrastructure

| Tool | Role |
|------|------|
| **Docker** | Runs Qdrant as `docker run -d --name qdrant-legal -p 6333:6333 qdrant/qdrant` |
| **Redis** | Celery broker; must be running locally for async task queue |
| **Jupyter Notebooks** | Development & demonstration environment for all 7 pipeline stages |

---

### Why NOT other alternatives?

| Choice Made | What we could have used | Why we chose ours |
|-------------|------------------------|------------------|
| `flan-t5-base` (local LLM) | GPT-4, Gemini API | No API key, no cost, runs offline, deterministic |
| `all-MiniLM-L6-v2` | OpenAI `text-embedding-ada-002` | Free, local, 384-dim is sufficient for legal clause similarity |
| Qdrant | Pinecone, Weaviate | Local Docker, no billing, production-grade filtering |
| SQLite | MongoDB, PostgreSQL | Zero-config for demo; SQLAlchemy makes it swappable |
| Next.js | React (CRA), Vue | App Router has built-in SSR, dynamic routes, TypeScript first-class |
| FastAPI | Flask, Django | Native async, auto-docs, Pydantic validation built-in |

---

<a name="b1"></a>
## B1. The CUAD Dataset — What it is, How it is Structured

### 📖 What is CUAD?

**CUAD** stands for **Contract Understanding Atticus Dataset**, created by **The Atticus Project** — a non-profit organization of legal professionals. It was released publicly on HuggingFace at `theatticusproject/cuad`.

- **Scale:** 510 commercial legal contracts
- **Annotations:** 13,000+ expert-annotated clause spans across **41 legal clause categories**
- **Annotators:** Trained law students and practicing attorneys
- **License:** CC BY 4.0 (free to use, even commercially)
- **Purpose:** Benchmarking NLP models on legal contract understanding

### 📦 What Types of Contracts are in CUAD?

CUAD contains real-world commercial contracts across many industries:

| Contract Type | Example |
|--------------|---------|
| Distributor Agreements | Electric City Corp distributor agreement (1999) |
| Software Licensing | WhiteSmoke Inc. promotion agreement |
| Consulting Agreements | Various service contracts |
| Non-Disclosure Agreements | Confidentiality-heavy contracts |
| Employment Agreements | Non-compete heavy contracts |
| Pharmaceutical Licensing | Pacira Pharmaceuticals licensing |

Each contract is a real document — not synthetic — sourced from public SEC filings.

---

### 🗂️ Native Format: SQuAD-style Question-Answering (QA)

CUAD is natively structured as a **Question-Answering (QA)** dataset following the **SQuAD 2.0 format**. This means every annotation is stored as:

```json
{
  "title": "LIMEENERGYCO_09_09_1999-DISTRIBUTOR_AGREEMENT",
  "paragraphs": [
    {
      "context": "This Agreement is entered into as of the 7th day of September, 1999,
                  by and between Electric City Corp., an Illinois corporation
                  ('Company') and John Doe ('Distributor')...",
      "qas": [
        {
          "question": "Highlight the parts of this contract related to 'Parties'",
          "id": "LIMEENERGYCO__Parties",
          "answers": [
            {
              "text": "Electric City Corp",
              "answer_start": 74
            },
            {
              "text": "John Doe",
              "answer_start": 120
            }
          ]
        },
        {
          "question": "Highlight the parts related to 'Governing Law'",
          "id": "LIMEENERGYCO__GoverningLaw",
          "answers": [
            {
              "text": "laws of the State of Illinois",
              "answer_start": 4312
            }
          ]
        }
      ]
    }
  ]
}
```

**Key fields:**
- `context` — the raw contract text (can be 30,000+ characters)
- `question` — one of 41 legal clause questions like "Highlight parts related to Termination"
- `answers[].text` — the exact substring in `context` that answers the question
- `answers[].answer_start` — character offset (index) where the answer begins in `context`

There are **41 unique questions**, one per clause category. Each contract can have answers for multiple categories.

---

### 🪟 The Windows MAX_PATH Problem

**Problem:** The raw CUAD dataset contains 500+ PDF files with extremely long filenames (e.g., `LIMEENERGYCO_09_09_1999-EX-10.12-DISTRIBUTOR AGREEMENT_FINAL_SIGNED.pdf`). Windows has a hard limit of **260 characters** for file paths. Trying to download these PDFs using the standard `load_dataset("theatticusproject/cuad")` command crashes mid-download on Windows.

**Solution used in LegalEagle:**
```python
# Instead of downloading all PDFs:
dataset = load_dataset(
    "json",
    data_files="CUAD_v1/CUAD_v1.json"  # only the JSON annotations
)
```

By specifying `data_files` to only fetch the structured JSON, we completely bypass downloading any PDF — and the JSON filenames are short enough to work on Windows.

**Why this matters in an interview:** Shows awareness of OS-level constraints and practical problem-solving beyond just running tutorial code.

---

### 📊 How CUAD is Loaded in the Project

After downloading, the dataset is in HuggingFace **Arrow format** (`data/cuad/`):
```
data/cuad/
├── train/
│   └── data-00000-of-00001.arrow   ← binary columnar format
├── validation/
│   └── data-00000-of-00001.arrow
└── dataset_info.json
```

Accessing it:
```python
from datasets import load_from_disk

dataset = load_from_disk("../data/cuad")
# dataset["train"][0] → one QA example
# Keys: 'id', 'title', 'context', 'question', 'answers'
```

**Arrow format** is a columnar binary format (Apache Arrow) — much faster to read than JSON because it's memory-mapped and zero-copy.

---

### 🎯 41 Clause Categories in CUAD (the ones used in LegalEagle)

LegalEagle uses a **subset of 8 categories** (selected based on legal importance):

| Entity Label | CUAD Question | Importance |
|---|---|---|
| `Parties` | "Who are the parties?" | Identifies who signed |
| `Agreement_Date` | "When was this signed?" | Effective date tracking |
| `Governing_Law` | "What jurisdiction governs?" | Determines legal recourse location |
| `Termination` | "What are termination conditions?" | When/how contract can end |
| `Indemnification` | "Who must indemnify whom?" | Financial liability exposure |
| `Confidentiality` | "What is confidential?" | Data/IP secrecy obligations |
| `IP_Ownership` | "Who owns IP created?" | Intellectual property rights |
| `Non_Compete` | "Is there a non-compete clause?" | Career/business restrictions |

These 8 were chosen because they represent the **highest legal risk** and are most commonly contested in contract disputes.

---

<a name="b2"></a>
## B2. QA → NER Conversion (BIO Tagging) — Why and How

### ❓ Why Convert at All?

The CUAD dataset is QA format: given a question + context, find the answer span. There are two ways to use this:

**Option 1 — Train a QA model (extractive):**
- Use `bert-base-uncased` with a QA head (`AutoModelForQuestionAnswering`)
- For each of 8 clause types, ask the question, get start/end positions
- **Problem:** You need 8 separate forward passes per contract (one per clause). Slow. Also, the QA model only answers one question at a time.

**Option 2 — Convert to NER and train a token classifier:**
- Merge all 8 clause types into a single label set
- Tag every token once → single forward pass extracts ALL clause types simultaneously
- **Chosen approach:** More efficient, cleaner output, single model

**Why NER wins:**
- **One pass** extracts all 8 entity types simultaneously
- NER models are smaller and faster than QA models
- Output is structured entity spans, not start/end indices
- Easier to post-process into a JSON report

---

### 🏷️ What is BIO Tagging?

**BIO** stands for **B**egin – **I**nside – **O**utside. It is the standard labeling scheme for sequence labeling (NER) tasks.

Every token in the text gets exactly one of these tag types:
- **`B-{LABEL}`** — This token **B**egins a new entity of type `LABEL`
- **`I-{LABEL}`** — This token is **I**nside (continuation of) the current entity
- **`O`** — This token is **O**utside (not part of any entity)

### 📝 Step-by-Step BIO Example

**Original contract text:**
```
This contract is governed by the laws of New York and shall terminate
upon thirty days written notice.
```

**Known annotations from CUAD:**
- `Governing_Law` → "laws of New York" (character offsets 33–48)
- `Termination` → "terminate upon thirty days written notice" (offsets 53–93)

**After BIO tagging (token level):**

| Token | Tag |
|-------|-----|
| This | O |
| contract | O |
| is | O |
| governed | O |
| by | O |
| the | O |
| laws | B-Governing_Law |
| of | I-Governing_Law |
| New | I-Governing_Law |
| York | I-Governing_Law |
| and | O |
| shall | O |
| terminate | B-Termination |
| upon | I-Termination |
| thirty | I-Termination |
| days | I-Termination |
| written | I-Termination |
| notice | I-Termination |
| . | O |

**Rule:** Only the very first token of an entity gets `B-`. All subsequent tokens of the same entity get `I-`. Any token not in any entity span gets `O`.

---

### ⚙️ How the Conversion Code Works (Step by Step)

**Inputs from CUAD:**
- `context` — full contract text string
- `answers[].answer_start` — character index where entity begins
- `answers[].text` — the entity string itself (so `answer_end = answer_start + len(text)`)

**Step 1 — Tokenize with offset mapping**
```python
encoding = tokenizer(
    context,
    max_length=256,
    truncation=True,
    stride=128,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,   # ← CRITICAL
    padding="max_length"
)
# offset_mapping[i] = (char_start, char_end) for token i
# e.g., token 6 ("laws") → (33, 37)
```

`return_offsets_mapping=True` is the key parameter. It tells the tokenizer to return the **character-level start and end position** of every token in the original string. Without this, we can't align CUAD's character offsets to BERT's subword tokens.

**Step 2 — Build a character-level label array**
```python
char_labels = ["O"] * len(context)

for answer in qas["answers"]:
    start = answer["answer_start"]
    end   = start + len(answer["text"])
    entity_type = clause_type  # e.g., "Termination"

    char_labels[start] = f"B-{entity_type}"
    for i in range(start + 1, end):
        char_labels[i] = f"I-{entity_type}"
```

**Step 3 — Map character labels → token labels**
```python
token_labels = []
for (char_start, char_end) in offset_mapping:
    if char_start == 0 and char_end == 0:
        token_labels.append(-100)  # [CLS], [SEP], [PAD] → ignored in loss
    else:
        token_labels.append(label2id[char_labels[char_start]])
```

We take the label at the **start character** of each token. Because BERT uses **WordPiece** tokenization (e.g., "termination" → ["term", "##ination"]), the first sub-token gets `B-Termination`, all subsequent sub-tokens get `I-Termination`.

**Step 4 — Result**
```python
# Final label sequence (as integer IDs):
# [CLS]=−100, "laws"=B-Governing_Law, "of"=I-Governing_Law, ...
# These are fed directly to the BERT training loop as targets
```

---

### 🔢 The Full Label Set (17 classes)

```python
LABELS = [
    "O",
    "B-Parties",        "I-Parties",
    "B-Agreement_Date", "I-Agreement_Date",
    "B-Governing_Law",  "I-Governing_Law",
    "B-Termination",    "I-Termination",
    "B-Indemnification","I-Indemnification",
    "B-Confidentiality","I-Confidentiality",
    "B-IP_Ownership",   "I-IP_Ownership",
    "B-Non_Compete",    "I-Non_Compete",
]
# 1 O class + 8 entity types × 2 (B + I) = 17 total labels
```

`ID2LABEL = {0: "O", 1: "B-Parties", 2: "I-Parties", ...}` is stored in the model's `config.json` so inference can decode integer predictions back to human-readable labels.

---

### ⚠️ The `-100` Trick (Critical for Training)

BERT adds special tokens: `[CLS]` at start, `[SEP]` at end, `[PAD]` for padding. These tokens have **no real linguistic meaning** and should not contribute to the NER loss.

PyTorch's `CrossEntropyLoss` has a special parameter: `ignore_index=-100`. Any token with label `-100` is **completely ignored** during loss calculation.

```python
# In training:
loss_fn = CrossEntropyLoss(ignore_index=-100)
# [CLS] token has label=-100 → its prediction is not penalized
# [PAD] tokens have label=-100 → no gradient flows from padding
```

This is why the BIO conversion code sets `token_labels.append(-100)` for special tokens — without this, the model would try to classify `[CLS]` as a legal entity, which is meaningless and would corrupt training.

---

### 📏 The Chunking Problem (MAX_LEN=256)

Legal contracts are 30,000–100,000 characters long. BERT has a hard limit of **512 tokens**. Even 512 tokens = ~350 words, which covers maybe 2–3 paragraphs of a contract.

**Solution: Chunk with stride (overlap)**
```python
encoding = tokenizer(
    text,
    max_length=256,       # each chunk: 256 tokens
    stride=128,            # consecutive chunks overlap by 128 tokens
    return_overflowing_tokens=True,  # produces multiple chunks
)
```

**Why stride/overlap?** If a clause spans a chunk boundary (e.g., a `Termination` clause starts at token 240 and ends at token 270), it would be cut in half without overlap. With `stride=128`, the next chunk starts 128 tokens before the end of the previous chunk — so the boundary clause appears fully in at least one chunk.

This creates multiple 256-token windows from a single contract. During training, the model learns from ALL windows. During inference, predictions from overlapping windows are merged (last-write wins for overlapping tokens).

---

### 🎯 Why NER over QA — Final Summary

| Dimension | QA Approach | NER (BIO) Approach |
|-----------|------------|-------------------|
| Forward passes per contract | 8 (one per clause type) | **1** |
| Output format | start/end integer indices | Entity spans as strings |
| Model architecture | QA head (2 output neurons) | Token classifier (17 neurons) |
| Training data needed | Original QA format | Converted BIO labels |
| Post-processing | Extract substring by indices | Merge B+I tokens |
| Scalability | Add new clause = new question | Add new clause = new label class |
| Complexity | Lower (reuse existing QA models) | Moderate (requires BIO conversion) |
| **Winner for LegalEagle** | | ✅ **NER — single pass, structured output** |

---

<a name="c1"></a>
## C1. BERT Architecture Deep Dive

### 🧠 What is BERT?

**BERT** = **B**idirectional **E**ncoder **R**epresentations from **T**ransformers. Released by Google in 2018, it fundamentally changed NLP by enabling models to understand context from **both left and right** simultaneously.

Before BERT, models like LSTM read text left-to-right (or right-to-left) — they were **unidirectional**. BERT reads the entire sentence at once — **bidirectional** — so it understands that "bank" in "river bank" means something completely different from "bank" in "bank account."

---

### 📐 BERT's Core Building Block: The Transformer Encoder

BERT is a stack of **Transformer Encoder layers**. It uses ONLY the encoder part of the original Transformer (Vaswani et al., 2017). There is no decoder.

**`bert-base-uncased` architecture numbers:**

| Property | Value |
|----------|-------|
| Encoder layers (L) | 12 |
| Hidden size (H) | 768 |
| Attention heads (A) | 12 |
| Feed-forward size | 3072 (4 × H) |
| Total parameters | ~110 million |
| Max input tokens | 512 |
| Vocabulary size | 30,522 tokens |

**`uncased`** means all text is lowercased before tokenization. "TechCorp" and "techcorp" are identical to the model. This is fine for legal NER since entity meaning doesn't depend on case.

---

### 🔤 WordPiece Tokenization

BERT uses **WordPiece** — a subword tokenization algorithm. Instead of splitting on spaces, it splits words into frequent subword pieces:

```
"indemnification" → ["ind", "##emn", "##ification"]
"termination"     → ["term", "##ination"]
"TechCorp"        → ["tech", "##corp"]   (uncased → lowercased first)
"New"             → ["new"]              (common word, stays whole)
```

The `##` prefix means "this piece continues the previous token (no space before it)."

**Why subword?**
- Handles rare/unseen legal terms (e.g., "indemnitor") by splitting into known pieces
- Keeps vocabulary size manageable (30,522 tokens covers almost everything)
- Never produces `[UNK]` (unknown) for any English word

**Special tokens added automatically:**
```
[CLS]  ← prepended to every input; its final hidden state = sentence representation
[SEP]  ← appended at the end of each sequence
[PAD]  ← added to make all sequences the same length in a batch
```

---

### ⚙️ How One Transformer Encoder Layer Works

Each of BERT's 12 layers applies the same operations:

```
Input tokens (embeddings)
        │
        ▼
┌─────────────────────────┐
│  Multi-Head Self-       │  ← Each token attends to ALL other tokens
│  Attention (12 heads)   │
└────────────┬────────────┘
             │ + Residual connection
             ▼
       Layer Normalization
             │
             ▼
┌─────────────────────────┐
│  Feed-Forward Network   │  ← Two linear layers: 768→3072→768
│  (position-wise)        │
└────────────┬────────────┘
             │ + Residual connection
             ▼
       Layer Normalization
             │
             ▼
   Output (same shape as input: seq_len × 768)
```

**Self-Attention formula:**
```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V
```
- **Q** (Query), **K** (Key), **V** (Value) are linear projections of the input
- `√d_k` (= √64 per head) is a scaling factor preventing vanishing gradients
- Output: each token gets a weighted sum of all other tokens' values — this is how "context" flows

**12 heads** means 12 parallel attention computations, each learning different relationship patterns (e.g., one head may learn subject-verb agreement, another learns coreference).

---

### 📦 Input Representation (3 Embeddings Summed)

Every token is represented as the **sum of three embeddings**:

```
Final token embedding = Token Embedding + Position Embedding + Segment Embedding
```

| Embedding | Purpose | Size |
|-----------|---------|------|
| **Token** | Maps token ID → vector (learned lookup table) | 768 |
| **Position** | Encodes token position (0, 1, 2, …, 511) — learned, not sinusoidal | 768 |
| **Segment** | `A` or `B` — which sentence this token belongs to (for sentence-pair tasks) | 768 |

For NER (single sequence): all tokens get Segment A embedding. Position embeddings are crucial — they tell the model where in the sentence each token sits.

---

### 🎯 Why BERT for Legal NER (not GPT, not T5)?

| Property | BERT | GPT | T5 |
|----------|------|-----|----|
| Architecture | Encoder only | Decoder only | Encoder-Decoder |
| Directionality | Bidirectional ✅ | Left-to-right | Bidirectional encoder |
| Best for | Classification, NER, QA | Text generation | Seq2seq tasks |
| Token-level output | Yes ✅ — one vector per token | No | Possible but cumbersome |
| Fine-tuning cost | Low ✅ | High | Medium |

**For NER**, you need one prediction **per input token** — BERT's encoder naturally produces one 768-dim vector per token. Adding a linear classification head on top is trivial. GPT cannot do this because it's causal (each token only sees past tokens).

---

<a name="c2"></a>
## C2. Fine-Tuning for Token Classification

### 🔧 What is Fine-Tuning?

BERT is pre-trained on **BooksCorpus + Wikipedia** (~3.3 billion words) using two tasks:
1. **Masked Language Modeling (MLM):** 15% of tokens are masked; BERT predicts them
2. **Next Sentence Prediction (NSP):** Given two sentences, predict if B follows A

After pre-training, BERT has rich language understanding but no domain-specific knowledge. **Fine-tuning** adds a task-specific head and trains the whole model on labeled data for the target task.

---

### 🏗️ Model Architecture for Token Classification

```
Input: [CLS] "laws" "of" "New" "York" [SEP]
         ↓     ↓     ↓    ↓     ↓      ↓
      BERT (12 Transformer encoder layers)
         ↓     ↓     ↓    ↓     ↓      ↓
      Hidden states: each token → 768-dim vector
         ↓     ↓     ↓    ↓     ↓      ↓
      Linear Layer (768 → 17)   ← classification head
         ↓     ↓     ↓    ↓     ↓      ↓
      Logits: 17 scores per token
         ↓     ↓     ↓    ↓     ↓      ↓
      argmax → label IDs: [O, B-GL, I-GL, I-GL, I-GL, O]
```

**`AutoModelForTokenClassification`** does exactly this: takes pre-trained BERT weights + adds a `Linear(768, num_labels=17)` on top.

---

### 📋 The HuggingFace Trainer API

```python
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer
)

model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=17,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

training_args = TrainingArguments(
    output_dir="./models/bert-ner-cuad",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,          # standard for BERT fine-tuning
    weight_decay=0.01,           # L2 regularization
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    processing_class=tokenizer,   # (was 'tokenizer=' in older versions)
    compute_metrics=compute_metrics,
)
trainer.train()
```

**Key hyperparameter decisions:**

| Param | Value | Why |
|-------|-------|-----|
| `learning_rate` | `2e-5` | Standard for BERT; too high (1e-3) causes catastrophic forgetting, too low (1e-7) is too slow |
| `num_train_epochs` | `3` | BERT fine-tuning converges fast; beyond 4 epochs often overfits |
| `batch_size` | `16` | Balances GPU memory vs. gradient stability |
| `weight_decay` | `0.01` | L2 regularisation prevents weights from growing too large |

---

### 📉 Loss Function: Cross-Entropy per Token

For each token, the model produces 17 logits. The loss is:

```
Loss = CrossEntropy(logits, true_label_id)
     = -log( softmax(logits)[true_label_id] )
```

This loss is computed for **every non-padded token** in every sequence and averaged over the batch. Tokens with `label=-100` (special tokens) are excluded via `ignore_index=-100`.

---

### 🔄 The `tokenizer` → `processing_class` Deprecation

In older HuggingFace versions:
```python
Trainer(tokenizer=tokenizer, ...)  # ← deprecated warning in 4.38+
```
In newer versions:
```python
Trainer(processing_class=tokenizer, ...)  # ← correct current API
```
This is a detail that shows you actually ran the code and dealt with real API changes — interviewers appreciate this.

---

### 💾 Saving the Fine-Tuned Model

```python
trainer.save_model("./models/bert-ner-cuad-final")
tokenizer.save_pretrained("./models/bert-ner-cuad-final")
```

This saves:
```
models/bert-ner-cuad-final/
├── config.json          ← num_labels=17, id2label, label2id
├── model.safetensors    ← all fine-tuned weights (~436 MB)
├── tokenizer.json
├── tokenizer_config.json
├── vocab.txt            ← 30,522 WordPiece vocabulary
└── special_tokens_map.json
```

The `config.json` stores `id2label` so inference code can decode predictions back to entity names without re-defining the label map.

---

<a name="c3"></a>
## C3. Sliding Window Inference — Why & How

### 🪟 The Core Problem

BERT can process at most **512 tokens** per forward pass. A typical legal contract has:
- 30,000 to 100,000 characters
- Which tokenizes to ~5,000–15,000 BERT tokens

Naively truncating to 512 tokens means you read **less than 10%** of the contract and miss most entities.

---

### ✅ Solution: Sliding Window with Stride

```python
enc = _NER_TOKENIZER(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=256,              # window size: 256 tokens per chunk
    stride=128,                  # overlap: next window starts 128 tokens back
    return_overflowing_tokens=True,   # creates multiple chunks automatically
    return_offsets_mapping=True,      # maps each token → original char position
    padding="max_length"
)
```

**What this produces for a 1000-token contract:**

```
Window 1: tokens   0 → 255   (chars 0    → ~1800)
Window 2: tokens 128 → 383   (chars ~900 → ~2700)   ← 128 overlap with W1
Window 3: tokens 256 → 511   (chars ~1800 → ~3600)  ← 128 overlap with W2
Window 4: tokens 384 → 639   (chars ~2700 → ~4500)  ← 128 overlap with W3
... and so on
```

Each window is a full 256-token BERT input. They are processed **independently** by BERT, then predictions are merged.

---

### ⚙️ Inference Loop (from `tasks.py _run_ner()`)

```python
def _run_ner(text: str) -> dict:
    enc = _NER_TOKENIZER(
        text, return_tensors="pt", truncation=True,
        max_length=256, stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )
    offsets = enc.pop("offset_mapping")          # shape: [num_windows, 256, 2]
    enc.pop("overflow_to_sample_mapping", None)  # not needed for inference

    entities = {}

    with torch.no_grad():
        for i in range(enc["input_ids"].shape[0]):   # for each window
            chunk = {k: v[i:i+1] for k, v in enc.items()}  # one window
            logits = _NER_MODEL(**chunk).logits          # shape: [1, 256, 17]
            preds = torch.argmax(logits, dim=-1)[0].tolist()  # [256]

            cur_label, cur_start = None, None
            for pred_id, (start, end) in zip(preds, offsets[i].tolist()):
                if start == 0 and end == 0:    # special token [CLS]/[SEP]/[PAD]
                    continue
                label = ID2LABEL.get(pred_id, "O")

                if label.startswith("B-"):     # new entity starts
                    if cur_label and cur_start is not None:
                        span = text[cur_start:end].strip()
                        if span:
                            entities.setdefault(cur_label, []).append(span)
                    cur_label, cur_start = label[2:], start   # e.g. "Termination", 340

                elif label.startswith("I-") and cur_label == label[2:]:
                    pass  # continue accumulating same entity

                else:                          # "O" or different entity type
                    if cur_label and cur_start is not None:
                        span = text[cur_start:start].strip()
                        if span:
                            entities.setdefault(cur_label, []).append(span)
                    cur_label, cur_start = None, None

    # Deduplicate (overlapping windows can produce duplicate spans)
    return {k: list(dict.fromkeys(v)) for k, v in entities.items()}
```

---

### 🔑 Key Details to Explain in Interview

**1. `offset_mapping` is the bridge:**
Each `(start, end)` pair in `offset_mapping[i]` gives the character positions in the **original string** for token `i`. This lets us reconstruct entity spans as original text substrings — without knowing anything about tokenization boundaries.

**2. `torch.no_grad()` block:**
Disables gradient computation during inference. Without it, PyTorch builds a computation graph for every forward pass (wasting memory and time). `no_grad()` makes inference ~2× faster and uses much less RAM.

**3. `argmax` on logits:**
```python
preds = torch.argmax(logits, dim=-1)
# logits shape: [batch=1, seq_len=256, num_labels=17]
# argmax over dim=-1 → picks the highest-scoring label for each token
# output shape: [1, 256] → 256 predicted label IDs
```

**4. Deduplication with `dict.fromkeys()`:**
Overlapping windows will produce the same entity span twice. `list(dict.fromkeys(v))` removes duplicates while **preserving insertion order** (unlike `list(set(v))` which is unordered). This matters for presenting entities in the order they appear in the contract.

**5. Why 256 tokens (not 512)?**
The 512-token limit is BERT's maximum. Using 256 with 128-stride gives denser coverage (more overlaps), better entity boundary detection, and allows larger batch sizes in GPU memory. It's a deliberate trade-off: more windows but each is more accurately processed.

---

### 📊 Sliding Window Coverage Diagram

```
Contract (1000 tokens total):
|──────────────────────────────────────────────────────────|

Window 1 (0-255):
|████████████████████████████████|

Window 2 (128-383):               ← 128 overlap with W1
                |████████████████████████████████|

Window 3 (256-511):               ← 128 overlap with W2
                                |████████████████████████████████|

Window 4 (384-639):               ← 128 overlap with W3
                                          |████████████████████████████████|
```

A clause at the boundary of Window 1/2 (tokens 230–260) is fully covered by both windows → guaranteed to be detected.

---

<a name="c4"></a>
## C4. Evaluation Metrics — seqeval, F1, Precision, Recall

### ❌ Why Standard Accuracy Fails for NER

Imagine a contract with 200 tokens. Typically:
- 180 tokens are labeled `O` (plain text)
- 20 tokens are labeled with entity types (B-/I- tags)

A model that **predicts `O` for every single token** achieves:
```
Accuracy = 180/200 = 90%   ← looks great, but the model found ZERO entities!
```

This is called the **class imbalance problem**. For NER, we need metrics that measure how well the model finds the rare entity tokens.

---

### 📐 The Three Core Metrics

#### Precision
> "Of all the entity spans the model predicted, what fraction were actually correct?"

```
Precision = True Positives / (True Positives + False Positives)
           = correctly found entities / all predicted entities
```

**Example:**
- Model predicted 10 Termination spans
- 7 were actually correct (match gold annotation)
- 3 were false positives (model hallucinated them)
- `Precision = 7/10 = 0.70`

#### Recall
> "Of all the actual entity spans in the text, what fraction did the model find?"

```
Recall = True Positives / (True Positives + False Negatives)
       = correctly found entities / all actual entities in text
```

**Example:**
- There are 9 actual Termination spans in the contract
- Model found 7 of them
- Missed 2 (false negatives)
- `Recall = 7/9 = 0.78`

#### F1 Score
> "The harmonic mean of Precision and Recall — balances both."

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = 2 × (0.70 × 0.78) / (0.70 + 0.78)
   = 2 × 0.546 / 1.48
   = 0.738
```

**Why harmonic mean (not arithmetic average)?**
The harmonic mean penalizes extreme imbalances. If Precision=1.0 and Recall=0.0:
- Arithmetic mean: (1.0 + 0.0)/2 = 0.50 (falsely optimistic)
- Harmonic mean: 2×(1.0×0.0)/(1.0+0.0) = 0.0 (correctly shows failure)

---

### 🧮 Entity-Level vs Token-Level Evaluation

This is a critical distinction. **`seqeval` evaluates at the entity span level**, not the token level.

**Token-level (wrong for NER):**
- Checks if each token's label is correct individually
- `["B-Termination", "I-Termination"]` → 2 token predictions checked

**Entity-level (correct, used by seqeval):**
- An entity prediction is a **complete span**: `(start_token, end_token, entity_type)`
- The prediction is correct ONLY if the **entire span** matches exactly
- Partial matches count as zero

**Example:**
- Gold: `Termination` spans tokens 45–50 (6 tokens)
- Model predicts: `Termination` spans tokens 45–49 (5 tokens — missed last token)
- Token-level score: 5/6 = 83% correct
- Entity-level score: **0%** — wrong because the span boundary is wrong

```python
from seqeval.metrics import classification_report, f1_score

# predictions and labels are lists of lists (one per sentence)
predictions = [["O", "B-Termination", "I-Termination", "O"]]
labels      = [["O", "B-Termination", "I-Termination", "O"]]

report = classification_report(labels, predictions, output_dict=True)
# Returns per-entity F1, precision, recall:
# {
#   "Termination": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 1},
#   "micro avg":   {"f1-score": 1.0, ...},
#   "macro avg":   {"f1-score": 1.0, ...}
# }
```

---

### 📊 Per-Entity F1 Scores — What to Expect

Based on the CUAD dataset characteristics and typical BERT fine-tuning results:

| Entity | Expected F1 | Why |
|--------|------------|-----|
| `Parties` | ~0.85–0.92 | Short spans, distinctive patterns ("between X and Y") |
| `Agreement_Date` | ~0.88–0.95 | Very short, consistent format ("as of March 15, 2023") |
| `Governing_Law` | ~0.80–0.90 | Short, predictable phrasing ("laws of the State of X") |
| `Confidentiality` | ~0.65–0.80 | Medium-length, more variable phrasing |
| `Termination` | ~0.55–0.72 | Can be long multi-sentence clauses |
| `Indemnification` | ~0.50–0.68 | Complex, variable phrasing |
| `IP_Ownership` | ~0.48–0.65 | Long, highly variable across contracts |
| `Non_Compete` | ~0.40–0.60 | Very long clauses, difficult to bound exactly |

**Key insight:** Short, formulaic entities score highest. Long, complex multi-sentence clauses score lowest — because a model with 256-token windows may not capture the full entity span, causing entity-level misses even when most tokens are correct.

---

### 🗺️ Confusion Matrix (Used in LegalEagle)

```python
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Flatten all token predictions and labels
flat_preds  = [p for seq in all_preds  for p in seq]
flat_labels = [l for seq in all_labels for l in seq]

cm = confusion_matrix(flat_labels, flat_preds, labels=LABELS)
# Plotted as a heatmap: rows=true labels, cols=predicted labels
# Diagonal = correct predictions
# Off-diagonal = confusions (e.g., B-Agreement_Date predicted as B-Parties)
```

**Common confusions to expect:**
- `Agreement_Date` confused with `Parties` — both are short, appear in preamble
- `Confidentiality` confused with `IP_Ownership` — both deal with proprietary information
- `I-X` tokens sometimes predicted as `B-X` — model restarts an entity mid-span

---

### 🎯 `compute_metrics` Function (passed to Trainer)

```python
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    true_labels, true_preds = [], []
    for pred_seq, label_seq in zip(predictions, labels):
        seq_labels, seq_preds = [], []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:       # skip special tokens
                continue
            seq_labels.append(ID2LABEL[l])
            seq_preds.append(ID2LABEL[p])
        true_labels.append(seq_labels)
        true_preds.append(seq_preds)

    return {
        "f1":        f1_score(true_labels, true_preds),
        "precision": precision_score(true_labels, true_preds),
        "recall":    recall_score(true_labels, true_preds),
    }
```

This function is called by the `Trainer` after every epoch to log validation metrics. The `-100` skip is critical — special tokens have no label and must be excluded before passing to `seqeval`.

---

<a name="d1"></a>
## D1. What is an Embedding? (all-MiniLM-L6-v2 Deep Dive)

### 🔢 What is a Text Embedding?

An **embedding** is a fixed-size numerical vector that encodes the **semantic meaning** of a piece of text. Two texts with similar meaning produce vectors that are **geometrically close** in the vector space, regardless of the exact words used.

```
"Either party may terminate upon 30 days notice."
         ↓   (sentence encoder)
[0.12, -0.45, 0.78, 0.03, -0.21, ..., 0.56]   ← 384 numbers

"This contract can be ended by either side with one month notice."
         ↓   (same encoder)
[0.14, -0.43, 0.76, 0.05, -0.19, ..., 0.54]   ← 384 numbers, very close!

"The royalty payment is due quarterly."
         ↓
[-0.32, 0.67, -0.12, 0.88, 0.41, ..., -0.23]  ← far away from the first two
```

This is fundamentally different from keyword search (TF-IDF, BM25) — embeddings capture **meaning**, not word overlap.

---

### 🏗️ The `all-MiniLM-L6-v2` Model

| Property | Value |
|----------|-------|
| Architecture | MiniLM (distilled transformer) |
| Layers | 6 |
| Hidden size | 384 |
| Parameters | ~22 million |
| Output dimension | **384 floats** per input text |
| Max input | 256 tokens (WordPiece) |
| Training objective | Sentence-level semantic similarity (contrastive learning) |
| Speed | ~14,000 sentences/second on GPU |
| License | Apache 2.0 (free, commercial use allowed) |
| API key needed? | ❌ None — runs 100% locally |

**Why 22M params, not 110M (BERT)?**
`all-MiniLM-L6-v2` was distilled from a larger model using **knowledge distillation** — a teacher-student training where the small model learns to mimic the large model's output. Result: 5× smaller, ~3× faster, with 95%+ of the larger model's semantic accuracy.

---

### 🧠 How It Produces Embeddings

**Step 1 — Tokenize input:** The text is tokenized with WordPiece (same as BERT).

**Step 2 — Forward pass:** The 6 Transformer encoder layers process the tokens, producing a 384-dim hidden state for each token.

**Step 3 — Mean Pooling:** All token hidden states are averaged into a single 384-dim vector.
```python
# Simplified mean pooling:
token_embeddings = model(input_ids, attention_mask)  # [seq_len, 384]
sentence_embedding = token_embeddings.mean(dim=0)    # [384] ← final embedding
```
Mean pooling is preferred over using the `[CLS]` token (as BERT does) because it captures the contribution of all tokens, not just the special first token.

**Step 4 — L2 Normalization:**
```python
embedding = embedding / embedding.norm()  # unit vector on a 384-dim sphere
```
After normalization, all vectors have length 1.0. This makes **cosine similarity** equivalent to **dot product** — much faster to compute.

---

### 📐 Cosine Similarity — The Distance Metric

```
cosine_similarity(A, B) = (A · B) / (|A| × |B|)
```

After L2 normalization, `|A| = |B| = 1`, so:
```
cosine_similarity(A, B) = A · B   (just dot product!)
```

| Score | Meaning |
|-------|---------|
| **1.0** | Identical meaning |
| **0.8–0.9** | Very similar (paraphrase) |
| **0.5–0.7** | Related topic |
| **0.0–0.3** | Unrelated |
| **< 0** | Contradictory (rare in practice) |

**Real example from LegalEagle:**
```
Query:   "What is the notice period for termination?"
Chunk A: "Either party may terminate upon ninety (90) days written notice."
Chunk B: "The distributor shall pay royalties on a quarterly basis."

cosine_similarity(query, A) → 0.87  ✅ Highly relevant
cosine_similarity(query, B) → 0.18  ❌ Unrelated
```

---

### 🔧 How it's Used in LegalEagle

```python
from langchain_huggingface import HuggingFaceEmbeddings

_EMBEDDER = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},  # L2 normalize → dot product = cosine
)

# Embed a query string → 384-dim vector
query_vector = _EMBEDDER.embed_query("What is the governing law?")

# Embed a list of documents → list of 384-dim vectors
doc_vectors = _EMBEDDER.embed_documents(["clause text 1", "clause text 2"])
```

Used in TWO places:
1. **Indexing time:** Embed contract chunks → store in Qdrant/ChromaDB
2. **Query time:** Embed the question → search for nearest neighbours in the DB

---

<a name="d2"></a>
## D2. ChromaDB — Storage, Indexing, Cosine Similarity

### 📦 What is ChromaDB?

ChromaDB is a **local, embedded vector database** — it runs as a Python library (no separate server needed). It stores embeddings + their associated metadata and text, and provides fast approximate nearest-neighbour (ANN) search.

Used in **Notebook 2** (`legal_eagle_rag_pipeline.ipynb`) as the first vector store prototype.

---

### 🗃️ How ChromaDB Stores Data

ChromaDB persists data to `data/chroma_db/` using **SQLite as its backend**:

```
data/chroma_db/
├── chroma.sqlite3        ← metadata, collection config, document text
└── <uuid>/
    └── data_level0.bin   ← HNSW graph index (the actual vector index)
```

Each stored item (called a **document**) consists of:
```
{
  "id":        "chunk_42",
  "embedding": [0.12, -0.45, 0.78, ...],   # 384 floats
  "document":  "Either party may terminate upon 90 days...",  # raw text
  "metadata":  {"source_file": "LIMEENERGYCO.txt", "chunk_index": 3}
}
```

---

### ⚡ HNSW Index (How Search is Fast)

ChromaDB uses **HNSW (Hierarchical Navigable Small World)** — a graph-based ANN algorithm.

- Vectors are connected in a layered graph where nearby vectors point to each other
- Query: start at a random entry point → greedily hop to neighbours closer to the query → return top-k
- Time complexity: **O(log N)** vs. brute-force O(N) — logarithmic in the number of stored vectors
- At 10,000 chunks: brute-force takes 10,000 dot products; HNSW takes ~150

The tradeoff: HNSW is **approximate** (can miss the true nearest neighbour rarely). For legal RAG this is acceptable — near-perfect recall is more important than exact top-1.

---

### 🔧 LangChain Integration

```python
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# BUILD: embed all chunks and store (one-time, slow)
vectorstore = Chroma.from_documents(
    documents=chunks,                       # list of Document objects
    embedding=embedding_model,
    collection_name="cuad_contracts",
    persist_directory="../data/chroma_db",  # persist to disk
)

# LOAD (subsequent runs — fast, no re-embedding):
vectorstore = Chroma(
    collection_name="cuad_contracts",
    embedding_function=embedding_model,
    persist_directory="../data/chroma_db",
)

# QUERY:
results = vectorstore.similarity_search_with_score(
    "What is the notice period for termination?",
    k=4   # top 4 most similar chunks
)
# Returns: [(Document, distance), (Document, distance), ...]
```

**Note on distance vs similarity:**
ChromaDB returns a **distance** (lower = more similar). To get similarity: `similarity = 1 - distance`. This is why the `search_contracts()` utility function in Notebook 2 does `round(1 - score, 4)`.

---

### ❗ Limitations of ChromaDB (Why Qdrant Replaced It)

| Limitation | Detail |
|-----------|--------|
| No metadata pre-filtering | ChromaDB filters metadata **after** retrieval — all vectors are scored, then filtered |
| No web UI | No way to visually browse the stored vectors |
| Single process | Not built for multi-process server deployment |
| No REST API | Cannot be queried from another service |
| Limited scalability | Struggles past ~1M vectors on a single machine |

---

<a name="d3"></a>
## D3. Qdrant — How it Differs, Docker Setup, Collection Structure

### 🚀 What is Qdrant?

**Qdrant** is a production-grade vector database built in Rust, designed to run as a standalone server. It exposes a REST API and gRPC API, has a web dashboard, supports distributed clusters, and critically: performs **server-side metadata pre-filtering** — filtering happens before scoring, not after.

Used from **Notebook 3 onwards** and in the production `backend/tasks.py`.

---

### 🐳 Running Qdrant with Docker

```bash
docker run -d \
  --name qdrant-legal \
  -p 6333:6333 \                          # REST API + Web UI
  -p 6334:6334 \                          # gRPC API
  -v ./data/qdrant_storage:/qdrant/storage \  # persist data to local disk
  qdrant/qdrant
```

- **Port 6333:** REST API — `http://localhost:6333/`; Web dashboard at `http://localhost:6333/dashboard`
- **Port 6334:** gRPC — faster binary protocol for high-throughput production
- **Volume mount:** `./data/qdrant_storage` → `/qdrant/storage` in the container. Data survives container restarts.

**Qdrant fallback in `tasks.py`:**
```python
try:
    check = subprocess.run(
        ["docker", "ps", "--filter", "name=qdrant-legal", "--format", "{{.Names}}"],
        capture_output=True, text=True, timeout=5
    )
    if "qdrant-legal" in check.stdout:
        _QDRANT = QdrantClient(host="localhost", port=6333)  # real Docker
    else:
        _QDRANT = QdrantClient(":memory:")   # in-memory fallback
except Exception:
    _QDRANT = QdrantClient(":memory:")
```
This makes the backend resilient — it works without Docker by falling back to in-memory Qdrant.

---

### 📋 Collection Structure — Points, Vectors, Payloads

A **collection** in Qdrant is equivalent to a table in SQL. Each record is a **Point**:

```python
from qdrant_client.models import PointStruct

PointStruct(
    id      = 42,                           # integer or UUID
    vector  = [0.12, -0.45, 0.78, ...],    # 384-float embedding
    payload = {                              # arbitrary JSON metadata
        "text":          "Either party may terminate upon 90 days notice...",
        "source_file":   "LIMEENERGYCO_1999.txt",
        "contract_type": "Distributor",
        "jurisdiction":  "US",
        "upload_date":   "2024-01-15",
        "chunk_index":   3,
    }
)
```

**Creating a collection:**
```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient(host="localhost", port=6333)

client.create_collection(
    collection_name="contracts",
    vectors_config=VectorParams(
        size=384,              # must match embedding model output dimension
        distance=Distance.COSINE,   # similarity metric
    ),
)
```

**Uploading points:**
```python
client.upsert(
    collection_name="contracts",
    points=[
        PointStruct(id=i, vector=embedding, payload={...})
        for i, (embedding, metadata) in enumerate(zip(embeddings, metadatas))
    ]
)
```

---

### 🔍 Server-Side Metadata Filtering — Qdrant's Killer Feature

In ChromaDB: search ALL vectors → return top-k → then filter by metadata (post-filter).
In Qdrant: filter by metadata **first** → search only qualifying vectors → return top-k.

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

results = client.query_points(
    collection_name="contracts",
    query=query_embedding,           # 384-float query vector
    query_filter=Filter(must=[
        FieldCondition(
            key="contract_type",
            match=MatchValue(value="Distributor")  # only search Distributor contracts
        ),
        FieldCondition(
            key="jurisdiction",
            match=MatchValue(value="US")
        ),
    ]),
    limit=5,                         # top 5 results
    with_payload=True,               # include text + metadata in response
)

# results.points → list of ScoredPoint objects
for point in results.points:
    print(point.score)                    # cosine similarity score
    print(point.payload["text"])          # the matched text chunk
    print(point.payload["source_file"])   # which contract it came from
```

**Why this matters:**
- If you have 50,000 contract chunks from 500 contracts, and you want only Distributor ones (say 8,000 chunks), Qdrant only scores those 8,000 — not all 50,000.
- ChromaDB would score all 50,000, then discard 42,000. Much slower.

---

### 🆚 ChromaDB vs Qdrant Summary

| Feature | ChromaDB | Qdrant |
|---------|---------|--------|
| Deployment | Embedded library | Docker server (or cloud) |
| Metadata filtering | Post-retrieval (slow) | **Server-side pre-filter (fast)** |
| Web dashboard | ❌ None | ✅ `localhost:6333/dashboard` |
| REST API | ❌ None | ✅ Full REST + gRPC |
| Persistence | SQLite file | Docker volume |
| Production readiness | Prototype | ✅ Production-grade |
| Language | Python | Rust (very fast) |
| Distributed clusters | ❌ | ✅ |
| LangChain integration | ✅ | ✅ |
| Setup complexity | Zero | Medium (Docker) |

---

<a name="d4"></a>
## D4. RAG Pipeline — Full Flow: Chunk → Embed → Store → Retrieve → Generate

### 🔄 What is RAG?

**RAG = Retrieval-Augmented Generation.** It is an architecture that separates *knowledge storage* (vector DB) from *generation* (LLM), combining both at query time.

**The core insight:** Instead of encoding all knowledge into model weights (which is expensive and stale), store knowledge in a searchable database and retrieve relevant context at query time.

```
WITHOUT RAG:
  Question → LLM → Answer
  (LLM must "remember" everything from training)

WITH RAG:
  Question → [Search DB for relevant context] → LLM(question + context) → Answer
  (LLM only needs to read and reason, not remember)
```

---

### 📊 The Two Phases of RAG

| Phase | When | What happens |
|-------|------|-------------|
| **Offline (Indexing)** | Once, when contracts are ingested | Load → Chunk → Embed → Store in vector DB |
| **Online (Query)** | Every time a question is asked | Embed question → Search DB → Retrieve top-k chunks → Feed to LLM → Generate answer |

---

### 🏗️ Full Pipeline — Step by Step

#### OFFLINE: Indexing Phase

```
Raw Contract (.txt / .pdf)
        │
        ▼ Step 1: LOAD
LangChain Document Loader
(PyPDFLoader / TextLoader)
        │
        ▼ Returns: Document(page_content="...", metadata={source: ...})
        │
        ▼ Step 2: CHUNK
RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        │
        ▼ Returns: [Chunk1, Chunk2, Chunk3, ...] (each ≤512 chars)
        │
        ▼ Step 3: EMBED
all-MiniLM-L6-v2.encode(chunk_text)
        │
        ▼ Returns: [384-float vector per chunk]
        │
        ▼ Step 4: STORE
Qdrant.upsert(vector + payload{text, source_file, contract_type})
        │
        ▼ Persisted to data/qdrant_storage/ ← done once
```

#### ONLINE: Query Phase

```
User Question: "Is this indemnification clause one-sided?"
        │
        ▼ Step 5: EMBED QUESTION
all-MiniLM-L6-v2.encode(question) → 384-float query vector
        │
        ▼ Step 6: RETRIEVE
Qdrant.query_points(query_vector, filter={contract_type: "Consulting"}, limit=3)
        │
        ▼ Returns: top 3 most similar chunks with scores
        │
        ▼ Step 7: FORMAT CONTEXT
context = "\n\n".join([point.payload["text"] for point in results.points])
        │
        ▼ Step 8: BUILD PROMPT
prompt = f"""You are a legal analyst.
CLAUSE: {clause_text}
SIMILAR MARKET CLAUSES: {context}
Score this clause 1-10 for risk. Reply: SCORE: <n> REASONING: <sentence>"""
        │
        ▼ Step 9: GENERATE
flan-t5-base.generate(prompt) → "SCORE: 7 REASONING: One-sided indemnification..."
        │
        ▼ Step 10: PARSE
score = re.search(r"SCORE:\s*(\d+)", response)  → 7
reason = re.search(r"REASONING:\s*(.+)", response) → "One-sided..."
```

---

### 🔧 How RAG is Used in `tasks.py` (`_rag_search` + `_score_clause`)

```python
def _rag_search(query: str, k: int = 2) -> list:
    """Step 5+6: embed query and search Qdrant"""
    qvec = _EMBEDDER.embed_query(query)        # 384-float vector
    res = _QDRANT.query_points(
        collection_name="contracts",
        query=qvec,
        limit=k,
        with_payload=True
    )
    return [{
        "source": p.payload.get("source_file", "?")[:40],
        "type":   p.payload.get("contract_type", "?"),
        "text":   p.payload.get("text", "")[:200]
    } for p in res.points]


def _score_clause(clause_type: str, texts: list) -> dict:
    """Steps 7–10: retrieve similar clauses, build prompt, score with LLM"""
    clause_text = "; ".join(texts[:2])[:300]
    similar = _rag_search(clause_text)          # Step 6: retrieve similar clauses

    # Step 7: format context
    ctx = "\n".join(f"- [{r['type']}] {r['text'][:120]}" for r in similar)

    # Step 8: build prompt
    prompt = (
        f"You are a legal risk analyst. Score this clause 1-10.\n"
        f"CLAUSE TYPE: {clause_type}\nTEXT: {clause_text}\n"
        f"SIMILAR MARKET CLAUSES:\n{ctx}\n"
        f"1=low risk, 10=high risk. Reply: SCORE: <n> REASONING: <sentence>"
    )

    # Step 9: generate
    try:
        resp = _LLM.invoke(prompt)
        m_s = re.search(r"SCORE:\s*(\d+)", resp, re.I)
        m_r = re.search(r"REASONING:\s*(.+)", resp, re.I | re.S)
        score = min(10, max(1, int(m_s.group(1)))) if m_s else RISK_BASELINE.get(clause_type, 5)
        reason = m_r.group(1).strip()[:200] if m_r else "Baseline heuristic."
    except Exception:
        score = RISK_BASELINE.get(clause_type, 5)
        reason = "Baseline heuristic score applied."

    return {"score": score, "reasoning": reason, "text": clause_text[:150],
            "similar_clauses": similar}
```

---

### 🛡️ Fallback: Risk Baseline

If the LLM fails to parse a score (garbled output), the system falls back to `RISK_BASELINE`:

```python
RISK_BASELINE = {
    "Termination":     6,
    "Indemnification": 7,
    "Non_Compete":     8,
    "IP_Ownership":    7,
    "Confidentiality": 5,
    "Governing_Law":   4,
    "Parties":         2,
    "Agreement_Date":  1,
}
```

These are expert-defined heuristic scores based on how inherently risky each clause type is — `Non_Compete` is always high risk regardless of exact wording; `Agreement_Date` is almost never a risk.

---

### 💡 Why RAG over Pure Fine-Tuning for Risk Scoring?

| Aspect | Fine-tune BERT/T5 for scoring | RAG + LLM |
|--------|------------------------------|-----------|
| New contracts | Requires retraining | **Just add to vector DB** |
| Explainability | Black box score | Score + reasoning + source citations |
| Context | Only trained data | Live retrieval from real market contracts |
| Setup cost | High (labeled score data) | Low (just ingest contracts) |
| Score quality | Limited by training data | Improves as more contracts are added |

---

<a name="d5"></a>
## D5. Text Chunking — RecursiveCharacterTextSplitter Explained

### ❓ Why Chunk at All?

Two hard constraints:
1. **Embedding model limit:** `all-MiniLM-L6-v2` max input = **256 tokens** (~170 words)
2. **Meaning concentration:** A 30,000-character contract embedded as one vector loses all local structure — the embedding represents the "average" of the whole document, which is useless for clause-level retrieval

Chunking splits the contract into pieces small enough to embed meaningfully. Each chunk should ideally contain one coherent idea or clause.

---

### 🔪 Why `RecursiveCharacterTextSplitter`?

LangChain provides multiple splitters. `RecursiveCharacterTextSplitter` is the best for legal text because it tries separators in order of preference:

```
Priority order of separators tried:
1. "\n\n"  ← double newline (paragraph break)  — BEST: keeps logical sections intact
2. "\n"    ← single newline (line break)
3. ". "    ← sentence boundary
4. " "     ← word boundary
5. ""      ← character split (last resort — never ideal)
```

It tries to split at `\n\n` first. If the resulting chunk is still too large, it recursively tries `\n`, then `. `, and so on. This is "recursive" in the sense that it applies itself recursively to oversized chunks using the next separator level.

**Why not a simple fixed-size splitter?**
A naive splitter cuts every 512 characters regardless of content — it will cut mid-sentence, mid-clause, or even mid-word. The recursive splitter respects natural text boundaries, keeping clauses semantically intact.

---

### ⚙️ Parameters Used

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,    # max characters per chunk
    chunk_overlap=50,  # shared characters between adjacent chunks
)

chunks = splitter.split_documents(documents)
```

**`chunk_size=512`:** Each chunk is at most 512 characters (~85 words). Well within the 256-token limit of MiniLM.

**`chunk_overlap=50`:** The last 50 characters of chunk N become the first 50 characters of chunk N+1.

---

### 🎯 Why Overlap? (Critical Concept)

Consider a Termination clause that spans character positions 490–560 in a contract:

```
WITHOUT OVERLAP (chunk_size=512):
  Chunk 1: chars   0–511  → contains "...Either party may termin"  (cut mid-clause!)
  Chunk 2: chars 512–1023 → contains "ate upon 90 days notice..."  (cut mid-clause!)

  Searching for "termination notice period":
  ❌ Neither chunk has the complete clause → retrieval FAILS
```

```
WITH OVERLAP (chunk_size=512, overlap=50):
  Chunk 1: chars   0–511  → "...Either party may termin"
  Chunk 2: chars 462–973  → "...Either party may terminate upon 90 days notice..."
                             ↑ 50-char overlap means clause appears FULLY in Chunk 2

  Searching for "termination notice period":
  ✅ Chunk 2 has the complete clause → retrieval SUCCEEDS
```

The overlap ensures that **no clause ever falls entirely in a crack between chunks**.

---

### 📏 Concrete Example (from LegalEagle contracts)

**Original text (700 characters):**
```
ARTICLE 9. TERMINATION.
9.1 This Agreement shall remain in force for an initial term of three (3) years
commencing on the Effective Date and shall automatically renew for successive
one-year terms unless either party provides written notice of non-renewal at
least ninety (90) days prior to the expiration of the then-current term.

9.2 Either party may terminate this Agreement upon material breach by the
other party, provided that the non-breaching party gives thirty (30) days
written notice of such breach...
```

**After splitting (chunk_size=512, overlap=50):**

| Chunk | Characters | Content |
|-------|-----------|---------|
| Chunk 1 | 0–511 | Article 9 header + 9.1 complete |
| Chunk 2 | 462–973 | Last 50 chars of 9.1 + all of 9.2 |

Both chunks are independently embedded and stored. A query for "termination breach notice" will retrieve Chunk 2 with high similarity because it contains the complete breach-termination language.

---

### 🆚 Chunking Strategy Comparison

| Strategy | How it splits | Problem |
|---------|--------------|---------|
| **Fixed-size** (`CharacterTextSplitter`) | Every N characters | Cuts mid-sentence, mid-clause |
| **Sentence** (`NLTKTextSplitter`) | At sentence boundaries | Legal sentences can be 300+ words long |
| **Token** (`TokenTextSplitter`) | Every N tokens | Token count hard to predict; may still cut clauses |
| **Recursive** ✅ | Tries `\n\n` → `\n` → `.` → ` ` → `""` | Best for structured legal documents with paragraphs |
| **Semantic** (advanced) | Splits where embedding similarity drops | Most accurate; computationally expensive |

**Why recursive wins for LegalEagle:**
Legal contracts are written with clear paragraph breaks (`\n\n`) separating articles and clauses. The recursive splitter respects these natural boundaries, producing chunks that correspond to complete legal provisions.

---

<a name="e1"></a>
## E1. flan-t5-base — Architecture, Why Chosen, Prompt Engineering

### 🧠 What is flan-t5?

**flan-t5** = **F**ine-tuned **LAN**guage Net — a T5 model instruction-tuned by Google on 1800+ NLP tasks. T5 itself (Text-To-Text Transfer Transformer) treats every NLP problem as a text-in → text-out task.

**`flan-t5-base`** is the smallest practical size:

| Property | Value |
|----------|-------|
| Architecture | T5 (Encoder-Decoder Transformer) |
| Parameters | 250 million |
| Training | Instruction-tuned on 1800+ task prompts |
| Input | Any text prompt (natural language instruction) |
| Output | Generated text (up to `max_new_tokens`) |
| License | Apache 2.0 (free, local, no API key) |
| GPU required? | No — runs on CPU (slow but functional) |

---

### ⚙️ T5 Architecture: Encoder-Decoder

Unlike BERT (encoder only) or GPT (decoder only), T5 uses **both**:

```
Prompt text
    │
    ▼
Encoder (12 layers) → encodes prompt into hidden states
    │
    ▼
Decoder (12 layers) → auto-regressively generates output tokens
    │
    ▼
Generated text: "SCORE: 7 REASONING: One-sided clause..."
```

- **Encoder:** Reads and understands the full prompt (bidirectional attention)
- **Decoder:** Generates tokens one by one, attending to encoder output + previously generated tokens
- Each generated token is fed back as input to generate the next → until `max_new_tokens` or EOS

This makes T5 ideal for **text generation tasks** like scoring with explanation — something BERT (encoder-only) cannot do natively.

---

### 🔧 How flan-t5 is Loaded in LegalEagle

```python
from transformers import pipeline as hf_pipeline
from langchain_huggingface import HuggingFacePipeline

# Step 1: create a HuggingFace text-generation pipeline
gen = hf_pipeline(
    "text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256,   # max tokens to generate in response
    do_sample=False,      # greedy decoding (deterministic, no randomness)
)

# Step 2: wrap it as a LangChain LLM interface
_LLM = HuggingFacePipeline(pipeline=gen)

# Step 3: call it like any LangChain LLM
response = _LLM.invoke("Your prompt here...")
# Returns: a plain Python string
```

**`do_sample=False`** means **greedy decoding** — at each step, pick the single highest-probability next token. This is deterministic: same prompt always gives same output. For risk scoring, determinism is desirable — you want consistent scores, not random variation.

**`max_new_tokens=256`** caps the output length. Legal risk reasoning is short (one sentence), so 256 is more than enough.

---

### ✍️ Prompt Engineering in LegalEagle

Prompt engineering is the practice of structuring the input text to guide the LLM toward the desired output format. flan-t5 is instruction-tuned — it follows explicit instructions in the prompt.

**The risk scoring prompt:**
```
"You are a legal risk analyst. Score this clause 1-10.
CLAUSE TYPE: Termination
TEXT: terminate immediately without written notice
SIMILAR MARKET CLAUSES:
- [Distributor] Either party may terminate upon ninety (90) days notice...
- [Licensing] This Agreement may be terminated upon thirty (30) days notice...
1=low risk, 10=high risk. Reply: SCORE: <n> REASONING: <sentence>"
```

**Why each part is there:**

| Prompt Element | Purpose |
|---------------|---------|
| `"You are a legal risk analyst."` | Role assignment — grounds the model in the domain |
| `"Score this clause 1-10."` | Explicit instruction — T5 is instruction-tuned to follow commands |
| `CLAUSE TYPE: Termination` | Tells the model what legal concept to reason about |
| `TEXT: <extracted span>` | The actual clause text to evaluate |
| `SIMILAR MARKET CLAUSES:` | RAG context — market comparison anchors the score |
| `1=low risk, 10=high risk.` | Defines the scale explicitly |
| `Reply: SCORE: <n> REASONING: <sentence>` | Forces structured output format for regex parsing |

**Parsing the output with regex:**
```python
resp = _LLM.invoke(prompt)
# resp = "SCORE: 8 REASONING: Immediate termination with no notice is unusual."

m_s = re.search(r"SCORE:\s*(\d+)", resp, re.I)
m_r = re.search(r"REASONING:\s*(.+)", resp, re.I | re.S)

score  = min(10, max(1, int(m_s.group(1)))) if m_s else RISK_BASELINE[clause_type]
reason = m_r.group(1).strip()[:200]          if m_r else "Baseline heuristic."
```

`min(10, max(1, ...))` clamps the score to [1,10] in case flan-t5 outputs `11` or `0`.

---

### 🆚 Why flan-t5-base NOT GPT-4 / Gemini / Claude?

| Criterion | flan-t5-base | GPT-4 / Gemini API |
|-----------|-------------|-------------------|
| API key required | ❌ No | ✅ Yes |
| Cost per request | Free | $0.01–$0.06 per call |
| Privacy | 100% local | Data sent to cloud |
| Determinism | ✅ Greedy decode | Variable (temperature) |
| Output quality | Moderate (250M params) | Excellent (trillion params) |
| Offline capable | ✅ Yes | ❌ No |
| Latency | ~2-5s on CPU | ~0.5s API |

For a demo/portfolio project: **local + free + no API key** is the right choice. In production, swapping to Gemini is one line change (just change the `_LLM` object).

---

<a name="e2"></a>
## E2. LangChain Tools & Agents

### 🤖 What is an Agent?

A standard LLM call is: `prompt → response`. That's one fixed step.

An **Agent** is a **loop** that dynamically decides which tools to call, in what order, based on intermediate results:

```
Task given to Agent
        │
        ▼
   [Think: what do I need to do?]
        │
        ├── Call Tool A → observe output
        │
        ├── Call Tool B with output of A → observe output
        │
        └── [Is task complete?] YES → return final answer
                               NO  → call another tool
```

The agent loop runs until the task is complete or a max iteration limit is hit.

---

### 🔧 LangChain `@tool` Decorator

Any Python function can become an agent-callable tool with `@tool`:

```python
from langchain_core.tools import tool
import json

@tool
def extract_legal_entities(contract_text: str) -> str:
    """Extract legal entities (Parties, Dates, Governing Law,
    Termination, Indemnification, Confidentiality, IP Ownership,
    Non-Compete) from a contract. Returns JSON string of entities."""
    entities = run_ner(contract_text)   # calls BERT NER
    return json.dumps(entities, indent=2)
```

The `@tool` decorator automatically creates:
- `extract_legal_entities.name` → `"extract_legal_entities"` (used to call it)
- `extract_legal_entities.description` → the docstring (used by agent to decide when to call it)
- `extract_legal_entities.args_schema` → JSON schema from the type hint `contract_text: str`

The **description** is critical — the agent (LLM) reads descriptions to decide which tool fits the current need.

---

### 🛠️ Two Agents in Notebook 4

#### Extractor Agent

```python
tools = [extract_legal_entities]

# The agent is told: "Extract all legal entities from this contract"
# It calls: extract_legal_entities(contract_text=<full text>)
# BERT NER runs → returns JSON string of entities
# Agent returns the result
```

Input to agent: `raw contract text (string)`
Output from agent: `JSON string → {"Parties": [...], "Termination": [...]}`

#### Risk Scorer Agent

```python
@tool
def score_clause_risk(clause_type: str, clause_text: str) -> str:
    """Score a legal clause 1-10 for risk using market comparison.
    Uses RAG to find similar clauses, then LLM to score.
    Returns JSON with score and reasoning."""
    result = _score_clause(clause_type, [clause_text])
    return json.dumps(result)

tools = [score_clause_risk]

# The agent iterates over each entity type from Extractor Agent output
# For each: calls score_clause_risk(clause_type="Termination", clause_text="...")
# Returns JSON: {"score": 8, "reasoning": "...", "similar_clauses": [...]}
```

---

### 🔄 Agent vs Direct Function Call — Why Use Agents?

In the **production backend** (`tasks.py`), the pipeline calls functions directly:
```python
entities = _run_ner(text)
risk_scores = {ct: _score_clause(ct, spans) for ct, spans in entities.items()}
```

In **Notebook 4**, the same logic is expressed as agents:
```python
extractor_agent.invoke({"input": contract_text})
risk_scorer_agent.invoke({"input": entities_json})
```

**Why agents in notebooks?**
- Demonstrates the concept of autonomous tool-calling
- Agent can retry on failure, choose different tools, or ask for clarification
- In a real multi-tool system (10+ tools), agents decide ordering — humans don't hardcode it
- Portfolio demonstration of LangChain agent patterns

**Why direct calls in production?**
- Predictable behavior (no LLM-decided routing)
- Faster (no agent reasoning loop overhead)
- Easier to debug

---

### 📋 LCEL (LangChain Expression Language) — Modern LangChain

In LangChain 1.3+, the old `RetrievalQA` chain was removed. The modern pattern is **LCEL** — pipe-based composition:

```python
from langchain_core.prompts import PromptTemplate

# Old (deprecated):
chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
answer = chain.run(question)

# New (LCEL / manual):
docs = retriever.invoke(question)                         # Step 1: retrieve
context = "\n\n".join(d.page_content for d in docs)      # Step 2: format
prompt_text = prompt.format(context=context, question=question)  # Step 3: build
answer = llm.invoke(prompt_text)                          # Step 4: generate
```

**Why manual invocation?**
- Full transparency: you can inspect `docs` before passing to LLM
- Citations work reliably — you know exactly which chunks were used
- No magic happening inside a black-box chain

---

### 🎯 Tool Schema — How the Agent Knows What to Call

When an agent sees a task, it reads tool descriptions and schemas to decide:

```
Agent receives: "Score the Termination clause: 'terminate immediately without notice'"

Available tools:
  1. extract_legal_entities(contract_text: str)
     → "Extract legal entities from a contract"
  2. score_clause_risk(clause_type: str, clause_text: str)
     → "Score a legal clause 1-10 for risk"

Agent reasons: "I need to score a clause, not extract. Use tool 2."
Agent calls: score_clause_risk(
    clause_type="Termination",
    clause_text="terminate immediately without notice"
)
```

This decision-making is what separates an agent from a hardcoded pipeline.

---

<a name="e3"></a>
## E3. LangGraph — StateGraph, Nodes, Edges, Conditional Routing

### 🗺️ What is LangGraph?

**LangGraph** is built on top of LangChain and models multi-agent workflows as a **directed graph**. Each node is an agent/function, each edge defines what runs next.

Key difference from plain LangChain chains:
- **Chains:** linear, fixed sequence (A → B → C)
- **LangGraph:** graph with loops, conditions, shared memory (A → B → if X then C else D → E)

Used in **Notebook 5** (`legal_eagle_langgraph.ipynb`).

---

### 📦 The Shared State — `AgentState`

All nodes share a single typed dictionary that evolves as the graph runs:

```python
from typing import TypedDict, List, Dict

class AgentState(TypedDict):
    contract_text:      str    # raw contract text (set at START)
    contract_name:      str    # file name
    entities:           dict   # set by Extractor node
    comparator_results: dict   # set by Comparator node
    risk_scores:        dict   # set by Risk Scorer node
    report:             str    # set by Report Generator node
    needs_human_review: bool   # set by Risk Scorer (True if any score > 7)
    human_review_flags: list   # list of high-risk clause names
```

**Why `TypedDict`?**
- Provides static type hints (IDE autocomplete, mypy checks)
- LangGraph validates state keys at runtime
- All nodes READ from and WRITE to this single state object — no manual passing of variables

Each node receives the full state and returns a **partial update** (only the keys it changed):

```python
def extractor_node(state: AgentState) -> dict:
    text = state["contract_text"]
    entities = run_ner(text)
    return {"entities": entities}   # ← only updates this key; rest of state unchanged
```

---

### 🔵 The 5 Nodes

```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(AgentState)

# Node 1: Extractor
def extractor_node(state):
    entities = run_ner(state["contract_text"])   # BERT NER
    return {"entities": entities}

# Node 2: Comparator
def comparator_node(state):
    results = {}
    for clause_type, spans in state["entities"].items():
        similar = _rag_search("; ".join(spans[:2]))   # Qdrant search
        results[clause_type] = similar
    return {"comparator_results": results}

# Node 3: Risk Scorer
def risk_scorer_node(state):
    scores, flags = {}, []
    for clause_type, spans in state["entities"].items():
        similar = state["comparator_results"].get(clause_type, [])
        result = score_with_llm(clause_type, spans, similar)  # flan-t5
        scores[clause_type] = result
        if result["score"] > 7:
            flags.append(clause_type)
    return {
        "risk_scores":        scores,
        "needs_human_review": len(flags) > 0,
        "human_review_flags": flags
    }

# Node 4: Human Review (only runs if score > 7)
def human_review_node(state):
    flags = state["human_review_flags"]
    # In production: send email, create JIRA ticket, pause for approval
    # In demo: add a warning to state
    print(f"⚠️ ATTORNEY REVIEW REQUIRED for: {flags}")
    return {}  # no state change, just side effect

# Node 5: Report Generator
def report_generator_node(state):
    report = _build_report(
        state["contract_name"],
        state["entities"],
        state["risk_scores"],
        state["human_review_flags"]
    )
    return {"report": report}

# Register all nodes
workflow.add_node("extractor",        extractor_node)
workflow.add_node("comparator",       comparator_node)
workflow.add_node("risk_scorer",      risk_scorer_node)
workflow.add_node("human_review",     human_review_node)
workflow.add_node("report_generator", report_generator_node)
```

---

### ➡️ Edges — Standard and Conditional

```python
# Standard edges (always run next in sequence)
workflow.set_entry_point("extractor")          # START → extractor
workflow.add_edge("extractor",  "comparator")  # always: extractor → comparator
workflow.add_edge("comparator", "risk_scorer") # always: comparator → risk_scorer

# Conditional edge (dynamic routing based on state)
def route_after_scoring(state: AgentState) -> str:
    if state.get("needs_human_review"):
        return "human_review"      # go to human review first
    return "report_generator"      # skip straight to report

workflow.add_conditional_edges(
    "risk_scorer",           # source node
    route_after_scoring,     # function that returns the next node name
    {
        "human_review":     "human_review",      # mapping: return value → node
        "report_generator": "report_generator",
    }
)

# After human review, always go to report
workflow.add_edge("human_review",     "report_generator")
workflow.add_edge("report_generator", END)   # terminal

# Compile the graph
app = workflow.compile()
```

---

### 🔄 Execution Flow Visualized

```
START
  │
  ▼
[extractor_node]
  Input:  state["contract_text"]
  Output: state["entities"] = {"Termination": [...], "IP_Ownership": [...]}
  │
  ▼
[comparator_node]
  Input:  state["entities"]
  Output: state["comparator_results"] = {"Termination": [similar_1, similar_2]}
  │
  ▼
[risk_scorer_node]
  Input:  state["entities"] + state["comparator_results"]
  Output: state["risk_scores"], state["needs_human_review"]=True, state["human_review_flags"]=["IP_Ownership"]
  │
  ├─── needs_human_review=True ──▶ [human_review_node]
  │                                    │
  │                                    ▼
  │                              (flags attorney)
  │                                    │
  └─── needs_human_review=False ─┐     │
                                  ▼     ▼
                           [report_generator_node]
                              Input:  full state
                              Output: state["report"] = "# Legal Contract..."
                                  │
                                  ▼
                                 END
```

---

### 🆚 LangGraph vs LangChain Agents vs Direct Calls

| | Direct Function Calls | LangChain Agents | LangGraph |
|--|--|--|--|
| Routing | Hardcoded | LLM decides | Graph + condition functions |
| State sharing | Manual passing | Agent scratchpad | Typed `AgentState` dict |
| Conditional flow | `if/else` in code | LLM reasoning | `add_conditional_edges` |
| Visibility | Easy to debug | Hard (LLM decides) | Graph visualization |
| Human-in-loop | Manual | Difficult | Built-in (interrupt nodes) |
| Best for | Simple pipelines | Dynamic tool selection | Complex multi-agent workflows |

---

<a name="e4"></a>
## E4. Server-Sent Events (SSE) Streaming

### 📡 What is SSE?

**Server-Sent Events (SSE)** is a web standard where a server can push a stream of text events to a browser client over a single long-lived HTTP connection.

```
Client                           Server
  │                                │
  │  GET /ask/stream?...           │
  │ ─────────────────────────────▶ │
  │                                │  (connection stays open)
  │ ◀─── data: {"token":"Found "}  │  event 1 (after 0ms)
  │ ◀─── data: {"token":"the "}    │  event 2 (after 40ms)
  │ ◀─── data: {"token":"clause"}  │  event 3 (after 80ms)
  │ ◀─── ...                       │
  │ ◀─── data: [DONE]              │  final sentinel
  │                                │  (server closes)
```

Each event is a text line formatted as:
```
data: <payload>\n\n
```
The double newline (`\n\n`) is the SSE event delimiter — the browser's `EventSource` uses this to know one event has ended.

---

### 🆚 SSE vs WebSockets vs Polling

| | SSE | WebSocket | HTTP Polling |
|--|--|--|--|
| Direction | Server → Client (one-way) | Bidirectional | Client asks repeatedly |
| Protocol | HTTP/1.1 | ws:// (separate protocol) | HTTP |
| Browser API | `EventSource` (built-in) | `WebSocket` (built-in) | `setInterval + fetch` |
| Reconnect | Auto-reconnect built-in | Manual | N/A |
| Overhead | Low | Very low | High (repeated connections) |
| Use case | Streaming text, notifications | Chat, gaming, collaborative | Simple polling |
| CORS | Standard HTTP CORS | Separate WS handshake | Standard CORS |

**Why SSE for LegalEagle chat?**
- The chat is one-way: server pushes answer tokens, client only sends questions (via separate POST/GET)
- No need for bidirectional WebSocket — SSE is simpler
- `EventSource` is built into every browser — zero library needed on frontend
- FastAPI's `StreamingResponse` makes SSE trivial to implement

---

### ⚙️ Backend: FastAPI SSE Implementation

```python
import asyncio
import json
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator

@app.get("/ask/stream")
async def ask_stream(job_id: str = Query(...), question: str = Query(...),
                     db: Session = Depends(get_db)):

    # 1. Load pre-computed results from SQLite (NO model re-run)
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    entities  = json.loads(job.entities_json or "{}")
    risk_data = json.loads(job.risk_json or "{}")
    report    = job.report_md or ""

    # 2. Build the answer (same 3-stage logic as /ask)
    q_lower = question.lower()
    # ... (entity match, risk match, report search)
    full_answer = "\n\n".join(parts) if parts else "Not found."

    # 3. Store in qa_records
    qa = QARecord(job_id=job_id, question=question, answer=full_answer)
    db.add(qa); db.commit()

    # 4. Define the async generator that streams word by word
    async def event_generator() -> AsyncGenerator[str, None]:
        words = full_answer.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield f"data: {json.dumps({'token': chunk})}\n\n"  # SSE format
            await asyncio.sleep(0.04)   # 40ms delay = ~25 words/second
        yield "data: [DONE]\n\n"        # sentinel event

    # 5. Return StreamingResponse with SSE content-type
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",      # don't buffer/cache the stream
            "X-Accel-Buffering": "no",         # disable nginx buffering (production)
        }
    )
```

**Key details:**
- `async def event_generator()` — must be `async` so `await asyncio.sleep()` works without blocking the server
- `AsyncGenerator[str, None]` — type hint: yields strings, never returns a value
- `await asyncio.sleep(0.04)` — **non-blocking** 40ms pause. During this sleep, FastAPI can handle other requests. This is why `async def` matters — synchronous `time.sleep(0.04)` would block the entire server thread.
- `"text/event-stream"` — the MIME type that tells the browser this is an SSE stream
- `"Cache-Control": "no-cache"` — browsers/proxies must not cache SSE responses
- `"X-Accel-Buffering": "no"` — tells nginx (reverse proxy) not to buffer the response; without this, nginx waits for the full response before forwarding, breaking streaming

---

### 🖥️ Frontend: EventSource in Chat.tsx

```typescript
const onSend = async (e?: React.FormEvent) => {
    const userQ = input;
    setInput('');
    setMessages(prev => [...prev, { role: 'user', text: userQ }]);
    setLoading(true);

    // Add empty AI message placeholder
    setMessages(prev => [...prev, { role: 'ai', text: '' }]);

    // Open SSE connection
    const url = `http://localhost:8000/ask/stream?job_id=${encodeURIComponent(jobId)}&question=${encodeURIComponent(userQ)}`;
    const eventSource = new EventSource(url);

    eventSource.onmessage = (event) => {
        if (event.data === '[DONE]') {
            eventSource.close();   // close connection
            setLoading(false);
            return;
        }
        try {
            const data = JSON.parse(event.data);   // { token: "Found " }
            setMessages(prev => {
                const newArr = [...prev];
                newArr[newArr.length - 1].text += data.token;  // append token
                return newArr;
            });
        } catch (err) {}
    };

    eventSource.onerror = () => {
        eventSource.close();
        setLoading(false);
    };
};
```

**Key details:**

**`new EventSource(url)`** — opens a persistent GET connection. The browser automatically:
- Keeps the connection alive
- Parses lines starting with `data:` as events
- Fires `onmessage` for each complete event (delimited by `\n\n`)
- Auto-reconnects if the connection drops (with `[DONE]` sentinel we close manually)

**State update pattern:**
```typescript
newArr[newArr.length - 1].text += data.token;
```
The last message in the array is the AI's reply (added as empty string initially). Each token is **appended** to it, never replacing — this produces the typing effect.

**`encodeURIComponent(question)`** — URL-encodes the question for safe inclusion in the query string. "What is the governing law?" becomes `What%20is%20the%20governing%20law%3F`.

---

### 🔁 Full SSE Request Lifecycle

```
1. User types question → clicks Send
2. Chat.tsx: setMessages adds user bubble + empty AI bubble
3. Chat.tsx: new EventSource("GET /ask/stream?job_id=...&question=...")
4. Browser: sends HTTP GET with "Accept: text/event-stream" header
5. FastAPI: receives request, loads SQLite data, builds full_answer string
6. FastAPI: starts async event_generator()
7. Server sends: "data: {"token":"Found "}\n\n"   (40ms)
8. Server sends: "data: {"token":"the "}\n\n"     (40ms)
9. ...
10. Server sends: "data: [DONE]\n\n"
11. Chat.tsx: eventSource.close() → setLoading(false)
12. Connection closed. qa_records row already written in step 5.
```

**Total latency:** The first token arrives as soon as FastAPI builds the answer (typically < 1s since no model is re-run). Then tokens stream at 40ms intervals giving ~25 words/second.

---

<a name="f1"></a>
## F1. How Risk Score is Calculated (1–10 Scale)

### 🎯 The Score is NOT a Single Step — It's a Pipeline

The final risk score for each clause comes from **4 combined steps**:

```
Step 1: BERT NER extracts clause text span
         ↓
Step 2: MiniLM embeds the span → Qdrant finds 2 similar market clauses
         ↓
Step 3: flan-t5 receives (clause text + similar clauses) → generates "SCORE: X REASONING: ..."
         ↓
Step 4: If flan-t5 fails → fall back to RISK_BASELINE[clause_type]
```

---

### 📏 The 1–10 Scale Defined

| Score | Badge | Meaning | Action |
|-------|-------|---------|--------|
| 1–3 | 🟢 LOW | Standard, balanced clause. Both parties protected equally. | No action needed |
| 4–6 | 🟡 MEDIUM | Unusual terms or jurisdiction. Worth reviewing carefully. | Review before signing |
| 7–10 | 🔴 HIGH | One-sided, no protections, extreme scope. Potentially harmful. | Seek legal advice. Do NOT sign without attorney |

### 🔢 Overall Score Calculation

```python
# After scoring all clause types:
risk_scores = {
    "Parties":         {"score": 2, ...},
    "Agreement_Date":  {"score": 1, ...},
    "Termination":     {"score": 8, ...},
    "Indemnification": {"score": 7, ...},
    "IP_Ownership":    {"score": 7, ...},
}

avg = sum(d["score"] for d in risk_scores.values()) / len(risk_scores)
# avg = (2+1+8+7+7) / 5 = 5.0

job.overall_score = round(avg, 2)  # stored as Float in SQLite
job.needs_human_review = 1 if any(d["score"] > 7 for d in risk_scores.values()) else 0
```

The overall score is a **simple arithmetic mean** of all extracted clause scores. `needs_human_review` is `True` if **any single clause** exceeds 7 — not the average.

### 🚩 The Flags List

```python
flags = [ct for ct, d in risk_scores.items() if d["score"] > 7]
# flags = ["Termination"]  (score=8, which is > 7)
```

These flagged clause types appear in the report with ⚠️, trigger the `needs_human_review=1` DB field, and show the red "Attorney Review Required" banner in the frontend.

---

### 📊 Risk Baseline (Fallback Scores)

```python
RISK_BASELINE = {
    "Termination":     6,
    "Indemnification": 7,
    "Non_Compete":     8,
    "IP_Ownership":    7,
    "Confidentiality": 5,
    "Governing_Law":   4,
    "Parties":         2,
    "Agreement_Date":  1,
}
```

These are **expert-defined heuristic scores** applied when:
1. flan-t5 generates unparseable output (no `SCORE: X` found by regex)
2. An exception occurs during LLM scoring

The values reflect inherent clause riskiness independent of wording — `Non_Compete` baseline is 8 because even a "standard" non-compete is inherently restrictive. `Agreement_Date` baseline is 1 because dates are factual, not risky.

---

<a name="f2"></a>
## F2. If YOU Give a New Sample Contract — How it is Judged

### 🔄 The Exact Journey of Your Contract

Suppose you upload `my_contract.txt`:

**Step 1 — File accepted:**
- Extension checked: must be `.txt` or `.pdf`
- UUID generated: `job_id = "f4a2-9c31-..."`
- Saved to `data/uploads/f4a2-9c31-....txt`

**Step 2 — Text extraction:**
```python
text = Path(file_path).read_text(encoding="utf-8", errors="ignore")[:5000]
```
Only first **5000 characters** are processed. If your contract is 50,000 chars, only the first 10% is analyzed. This is a deliberate speed/resource tradeoff for the demo.

**Step 3 — BERT NER on your contract:**
The model was trained on CUAD contracts. It will look for the 8 entity types it was trained on. If your contract uses unusual phrasing (e.g., "cessation" instead of "termination"), the model may miss it — this is a known limitation of supervised NER.

**Step 4 — NER fallback (if nothing found):**
```python
if not entities:
    text_lower = text.lower()
    if "terminat" in text_lower:
        entities["Termination"] = ["termination clause present"]
    if "indemnif" in text_lower:
        entities["Indemnification"] = ["indemnification clause present"]
    # ... keyword matching as fallback
```
Simple keyword detection ensures the system always returns *something*.

**Step 5 — Each entity is RAG-scored:**
For every extracted clause, MiniLM embeds it → Qdrant finds the 2 most similar market clauses from the 10 indexed CUAD contracts → flan-t5 generates a score comparing your clause to those market examples.

**Step 6 — Score interpretation:**

```
YOUR CONTRACT SCORE DEPENDS ON:

1. WHAT your clause says (the actual NER-extracted text)
2. HOW it compares to market (Qdrant RAG context)
3. WHAT flan-t5 infers about the comparison

Examples:
- "terminate upon 90 days notice" → similar to CUAD standard → score ~4 (MEDIUM)
- "terminate immediately without notice" → far from market standard → score ~8 (HIGH)
- "governing law: Delaware" → standard jurisdiction → score ~3 (LOW)
- "governing law: Cayman Islands" → offshore, unusual → score ~6 (MEDIUM)
```

### 📐 What Metrics the Judge Uses

| Metric | How Calculated | What it means for your contract |
|--------|---------------|--------------------------------|
| **Per-clause score (1-10)** | flan-t5 output, clamped to [1,10], fallback to RISK_BASELINE | How risky each specific clause is |
| **Overall score** | Arithmetic mean of all clause scores | Single-number summary of the contract |
| **needs_human_review** | True if ANY clause > 7 | Whether an attorney must review before signing |
| **flags list** | All clauses with score > 7 | Specific danger areas to focus on |
| **RAG similarity score** | Cosine similarity of clause embedding vs market chunks | How "market standard" your clause is |
| **Reasoning string** | flan-t5-generated explanation | Why the score was assigned |

### 🎯 How to Make a Contract Score HIGH (Red) vs LOW (Green)

| For HIGH risk 🔴 | For LOW risk 🟢 |
|------------------|----------------|
| "terminate immediately without notice" | "terminate upon 90 days written notice" |
| "indemnify from ANY and ALL claims" (one-sided) | "mutual indemnification" |
| "non-compete for 10 years worldwide" | "non-compete for 1 year in [specific city]" |
| "all IP belongs to Company including prior work" | "IP created during this engagement belongs to Company" |
| "governing law: Cayman Islands" | "governing law: California" |

---

<a name="g1"></a>
## G1. FastAPI — All Endpoints, Request/Response Flow

### 🌐 What is FastAPI?

**FastAPI** is a modern Python web framework built on:
- **Starlette** (ASGI web toolkit) — handles routing, middleware, request/response
- **Pydantic** (data validation) — validates request bodies automatically
- **Uvicorn** (ASGI server) — runs the async event loop

It auto-generates **OpenAPI** (Swagger) docs at `/docs` and ReDoc at `/redoc` — no extra work needed.

---

### 📋 All Endpoints — Complete Reference

#### `POST /upload`
```
Input:  multipart/form-data  →  file (binary)
Output: JSON
  {
    "job_id":   "a3f9-...",
    "filename": "contract.txt",
    "path":     "data/uploads/a3f9-....txt",
    "message":  "File uploaded. Call POST /analyze..."
  }

Errors:
  400 → file extension not in {.txt, .pdf}
```

**Code path:** `main.py → upload()` → validates suffix → `uuid.uuid4()` → `shutil.copyfileobj()` → returns dict

---

#### `POST /analyze`
```
Input:  JSON body  →  { "job_id": "a3f9-..." }
Output: JSON
  {
    "job_id":        "a3f9-...",
    "status":        "done",
    "overall_score": 6.4,
    "message":       "Analysis complete! Call GET /report/..."
  }

Errors:
  404 → no file found for that job_id
  500 → AI pipeline threw an exception
```

**Code path:** `main.py → analyze()` → finds file → creates `AnalysisJob` DB row → calls `analyze_contract_core()` in `tasks.py` → refreshes DB row → returns score

**Note:** Runs **synchronously** in demo mode. In production, `run_analysis.delay(job_id, ...)` would queue to Celery and return `202 Accepted` immediately.

---

#### `GET /report/{job_id}`
```
Input:  path parameter  →  job_id in URL
Output: JSON
  {
    "job_id":             "a3f9-...",
    "contract_name":      "consulting_agreement",
    "status":             "done",
    "overall_score":      6.4,
    "needs_human_review": true,
    "created_at":         "2024-01-15 10:23:45",
    "entities":           { "Parties": [...], "Termination": [...] },
    "risk_scores":        { "Termination": {"score": 8, "reasoning": "..."} },
    "report_markdown":    "# Legal Contract Risk Analysis Report\n..."
  }

Errors:
  404 → no job found with that id
```

**Code path:** `main.py → get_report()` → `db.query(AnalysisJob).filter(id=job_id).first()` → `json.loads(entities_json)` → returns full dict

---

#### `POST /ask`
```
Input:  JSON body  →  { "job_id": "a3f9-...", "question": "What is the notice period?" }
Output: JSON
  {
    "job_id":   "a3f9-...",
    "question": "What is the notice period?",
    "answer":   "Extracted entities: terminate immediately without written notice\nTermination: score 8/10 — ..."
  }

Errors:
  404 → job not found
  400 → analysis not complete yet (status != "done")
```

**3-stage retrieval (no LLM call):**
1. Entity keyword match → finds entity spans containing question keywords
2. Risk score keyword match → finds clause scores matching question keywords
3. Report line search → finds relevant lines in the Markdown report

---

#### `GET /ask/stream`
```
Input:  query params  →  ?job_id=a3f9-...&question=What+is+the+governing+law%3F
Output: text/event-stream (SSE)
  data: {"token": "Governing "}
  data: {"token": "Law: "}
  ...
  data: [DONE]

Errors:
  404 → job not found
  400 → analysis not complete
```

Same 3-stage retrieval as `/ask`, but streams the answer word-by-word at 40ms intervals.

---

#### `GET /health`
```
Output: JSON  →  { "status": "ok", "timestamp": "2024-01-15T10:23:45" }
```
Used by the notebook startup code to poll until the server is ready after `subprocess.Popen(uvicorn...)`.

---

### ⚙️ FastAPI Internals Used

**Dependency Injection — `Depends(get_db)`:**
```python
def get_db():
    db = SessionLocal()
    try:
        yield db       # ← yield makes it a generator/context manager
    finally:
        db.close()     # ← always closes, even if exception occurs

@app.post("/analyze")
def analyze(req: AnalyzeRequest, db: Session = Depends(get_db)):
    # db is injected automatically — no need to open/close manually
```

FastAPI's `Depends` system calls `get_db()` for every request, yields the `db` session, and ensures `db.close()` runs after the response — even if an exception is raised. This prevents connection leaks.

**Pydantic Schema Validation:**
```python
class AnalyzeRequest(BaseModel):
    job_id: str

class AskRequest(BaseModel):
    job_id: str
    question: str
```
If the client sends `{"job_id": 123}` (integer instead of string), FastAPI automatically returns `422 Unprocessable Entity` with a detailed error — no manual validation code needed.

**CORS Middleware:**
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```
Without CORS, the browser blocks requests from `localhost:3000` (Next.js) to `localhost:8000` (FastAPI) — same-origin policy restriction. The middleware adds `Access-Control-Allow-Origin: http://localhost:3000` to every response header, telling the browser these cross-origin requests are allowed.

---

<a name="g2"></a>
## G2. SQLite + SQLAlchemy ORM — Tables, Columns, Why SQLite

### 🗃️ The Two Tables

#### `analysis_jobs`

| Column | SQLAlchemy Type | Python Type | Description |
|--------|----------------|-------------|-------------|
| `id` | `String` (PK) | `str` | UUID from `uuid.uuid4()` |
| `contract_name` | `String` | `str` | Filename stem (no extension) |
| `status` | `String` | `str` | `pending` → `running` → `done` \| `failed` |
| `celery_task_id` | `String` (nullable) | `str` | Celery task ID (async mode only) |
| `created_at` | `DateTime` | `datetime` | Auto-set on creation |
| `updated_at` | `DateTime` | `datetime` | Auto-updated on change |
| `overall_score` | `Float` (nullable) | `float` | Average of all clause scores |
| `report_md` | `Text` (nullable) | `str` | Full Markdown report string |
| `entities_json` | `Text` (nullable) | `str` | JSON string: `{"Parties": [...]}` |
| `risk_json` | `Text` (nullable) | `str` | JSON string: `{"Termination": {"score": 8}}` |
| `needs_human_review` | `Integer` | `int` | `0` or `1` (SQLite has no Boolean) |

#### `qa_records`

| Column | SQLAlchemy Type | Description |
|--------|----------------|-------------|
| `id` | `Integer` (PK, autoincrement) | Row number |
| `job_id` | `String` (indexed) | Foreign key → `analysis_jobs.id` |
| `question` | `Text` | User's question string |
| `answer` | `Text` | Retrieved answer string |
| `created_at` | `DateTime` | When the question was asked |

---

### ⚙️ How SQLAlchemy ORM Works

**ORM = Object-Relational Mapper.** It maps Python class instances to database table rows. You interact with Python objects instead of writing raw SQL.

```python
# Creating a row (no SQL written):
job = AnalysisJob(
    id=req.job_id,
    contract_name=contract_name,
    status="pending"
)
db.add(job)   # stage for insertion
db.commit()   # write to disk

# Reading a row:
job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
# SQLAlchemy generates: SELECT * FROM analysis_jobs WHERE id = ? LIMIT 1

# Updating:
job.status = "done"
job.overall_score = 6.4
db.commit()   # UPDATE analysis_jobs SET status=?, overall_score=? WHERE id=?

# The ORM tracks changes ("dirty" objects) automatically
```

**Session lifecycle:**
```python
SessionLocal = sessionmaker(bind=engine)

def get_db():
    db = SessionLocal()   # open connection
    try:
        yield db
    finally:
        db.close()        # always close (return to pool)
```

`db.expire_all()` → marks all loaded objects as stale (forces re-read from DB on next access). Used in `/analyze` after `analyze_contract_core()` updates the DB from within `tasks.py` — the session in `main.py` needs to re-read the fresh data.

---

### 📁 Why SQLite (not PostgreSQL / MongoDB)?

**SQLite advantages for this project:**
- **Zero config:** `legaleagle.db` is a single file — no server, no install, no connection string
- **Portable:** The entire database is one file you can copy/move
- **`check_same_thread=False`:** SQLite's default mode only allows one thread at a time. FastAPI uses multiple threads. This flag disables that restriction for the demo.

```python
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
```

**Production upgrade path:** Change one env var:
```bash
DATABASE_URL=postgresql://user:pass@localhost/legaleagle
```
The SQLAlchemy ORM code requires **zero changes** — it generates the correct SQL dialect automatically.

---

<a name="g3"></a>
## G3. Celery + Redis — Async Task Queue Architecture

### ⚡ The Problem Celery Solves

`POST /analyze` currently runs the full AI pipeline **synchronously** — the HTTP request blocks for 30-60 seconds while BERT + Qdrant + flan-t5 run. The browser times out. Users cannot use the UI during analysis.

**With Celery:**
```
Client                FastAPI           Redis            Celery Worker
  │                     │                │                    │
  │  POST /analyze       │                │                    │
  │ ──────────────────▶  │                │                    │
  │                      │  queue task    │                    │
  │                      │ ──────────────▶│                    │
  │  202 Accepted         │                │  consume task      │
  │ ◀──────────────────── │                │ ──────────────────▶│
  │                                        │                    │  (runs AI)
  │  poll GET /report/{id}                 │                    │
  │ ──────────────────────────────────────────────────────────  │
  │  {"status": "running"}                                      │
  │ ◀──────────────────────────────────────────────────────────  │
  │  (after 45s)                                                │
  │  {"status": "done", "overall_score": 6.4}                   │
```

---

### ⚙️ Celery Configuration (`celery_app.py`)

```python
from celery import Celery

celery_app = Celery(
    "legaleagle",
    broker="redis://localhost:6379/0",   # Redis DB 0: task queue
    backend="redis://localhost:6379/1",  # Redis DB 1: result storage
)

celery_app.conf.update(
    task_serializer="json",             # tasks serialized as JSON (not pickle)
    result_serializer="json",
    accept_content=["json"],
    result_expires=86400,               # results stored for 1 day
    worker_prefetch_multiplier=1,       # worker fetches 1 task at a time
    task_acks_late=True,                # task acknowledged AFTER completion, not before
)
```

**`task_acks_late=True`** — if the worker crashes mid-task, the task goes back to the queue (not lost). Without this, a crash between "acknowledged" and "completed" means the task is silently dropped.

**`worker_prefetch_multiplier=1`** — each worker takes only 1 task at a time. Since AI inference is CPU-heavy, this prevents one worker from hoarding all tasks while others sit idle.

---

### 🔧 The Celery Task Wrapper

```python
@celery_app.task(name="tasks.run_analysis")
def run_analysis(job_id: str, file_path: str, contract_name: str):
    """Celery-wrapped version of analyze_contract_core."""
    return analyze_contract_core(job_id, file_path, contract_name)
```

The actual AI logic is in `analyze_contract_core()` — a plain Python function. The Celery decorator just wraps it for async dispatch:

```python
# Sync (demo mode — used currently):
analyze_contract_core(job_id, file_path, contract_name)

# Async (production mode — Celery):
run_analysis.delay(job_id, file_path, contract_name)
# Returns immediately with a task ID; result stored in Redis after completion
```

**`run_analysis.delay()`** is Celery's shorthand for `run_analysis.apply_async()`. It serializes the arguments to JSON, pushes to Redis queue, and returns immediately.

---

### 📦 Redis Roles in This System

Redis is used for **two purposes**:

| Role | Redis DB | Data stored |
|------|----------|------------|
| **Broker (task queue)** | DB 0 (`/0`) | Serialized task payloads waiting to be picked up by workers |
| **Result backend** | DB 1 (`/1`) | Task results keyed by task ID, expired after 24 hours |

When a worker completes a task, results are also written to **SQLite** by `analyze_contract_core()` — so `/report/{id}` reads from SQLite, not from Redis. Redis results are only used by `run_analysis.AsyncResult(task_id)` if you want to check Celery-level task status.

---

<a name="g4"></a>
## G4. Lazy Model Loading — Why and How

### 🐢 The Problem: Loading 4 Models Takes 30+ Seconds

```python
_NER_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH)         # ~5s
_NER_MODEL     = AutoModelForTokenClassification.from_pretrained() # ~10s
_EMBEDDER      = HuggingFaceEmbeddings("all-MiniLM-L6-v2")         # ~5s
_LLM           = HuggingFacePipeline(pipeline=hf_pipeline(...))    # ~15s
_QDRANT        = QdrantClient(host="localhost", port=6333)          # <1s
```

If you load all 4 models at server startup (`@app.on_event("startup")`), the server takes 35+ seconds to start — and fails health checks during that window.

---

### ✅ Solution: Lazy Loading on First Request

```python
# Module-level globals (initially None)
_MODELS_LOADED  = False
_NER_TOKENIZER  = None
_NER_MODEL      = None
_EMBEDDER       = None
_LLM            = None
_QDRANT         = None

def _load_models():
    global _MODELS_LOADED, _NER_TOKENIZER, _NER_MODEL, _EMBEDDER, _LLM, _QDRANT
    if _MODELS_LOADED:
        return        # ← guard: only load once, skip on subsequent calls
    
    # Load everything (runs once, ~35 seconds)
    _NER_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH)
    _NER_MODEL     = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    _NER_MODEL.eval()        # switch to inference mode (disables dropout)
    _EMBEDDER      = HuggingFaceEmbeddings("all-MiniLM-L6-v2")
    gen            = hf_pipeline("text-generation", model="google/flan-t5-base")
    _LLM           = HuggingFacePipeline(pipeline=gen)
    _QDRANT        = QdrantClient(...)   # with Docker fallback
    
    _MODELS_LOADED = True    # set flag so this block never runs again
```

Called at the start of `analyze_contract_core()`:
```python
def analyze_contract_core(job_id, file_path, contract_name):
    _load_models()   # ← loads on first /analyze call; skips on all subsequent calls
    # ... rest of pipeline
```

**Benefits:**
- Server starts instantly (< 1 second)
- Health check `/health` responds immediately
- First `/analyze` request is slow (35s model load + analysis time)
- All subsequent `/analyze` requests skip the load (models already in memory)

**Why `_NER_MODEL.eval()`?**
PyTorch models have two modes:
- **Training mode (default):** Dropout layers randomly zero out neurons (for regularization)
- **Inference mode (`eval()`):** Dropout is disabled — every forward pass is deterministic

Forgetting `.eval()` causes slightly different NER predictions on each run (non-deterministic) — a subtle but real bug.

