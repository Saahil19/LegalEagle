# 🗺️ LegalEagle — Complete Data Flow Map
> What data goes FROM where TO where, and in what format.

---

## ⚡ QUICK ANSWER: The Two Separate Uses of "Embeddings"

There are **two completely different embedding operations** happening in this project. People confuse them because both use neural networks to produce vectors — but they are for different purposes:

| | BERT NER (Part C) | MiniLM Sentence Embeddings (Part D) |
|--|--|--|
| **Model** | `bert-ner-cuad-final` (fine-tuned BERT) | `all-MiniLM-L6-v2` |
| **Input** | Contract text tokens | Contract text chunks |
| **Output** | **17 class labels per token** (BIO tags) | **384-dim float vector per chunk** |
| **Purpose** | Find WHERE entities are in the text | Find WHICH chunks are similar to a query |
| **Stored where?** | NOT stored — used immediately | Stored in Qdrant/ChromaDB |
| **Used for** | Extracting clause text spans | Semantic search / RAG |

They never interact. BERT NER does not produce vectors that go into Qdrant. MiniLM does not produce labels.

---

## 📋 FULL PIPELINE: Input → Output at Every Step

### STEP 1 — User Uploads a File

```
User (browser)
   │
   │  HTTP POST /upload
   │  multipart/form-data body:  file = "consulting_agreement.txt" (binary bytes)
   ▼
FastAPI /upload endpoint (main.py)
   │
   │  Validates: suffix must be in {".txt", ".pdf"}
   │  Generates: job_id = uuid.uuid4()  →  e.g. "a3f9-bc12-..."
   │  Writes:    data/uploads/a3f9-bc12-....txt   (raw file saved to disk)
   │
   │  Returns JSON:
   ▼
{
  "job_id": "a3f9-bc12-...",
  "filename": "consulting_agreement.txt",
  "path": "data/uploads/a3f9-bc12-....txt",
  "message": "File uploaded. Call POST /analyze..."
}
   │
   ▼
Frontend stores job_id in memory, then immediately calls /analyze
```

---

### STEP 2 — /analyze Triggers the AI Pipeline

```
Frontend
   │
   │  HTTP POST /analyze
   │  JSON body: { "job_id": "a3f9-bc12-..." }
   ▼
FastAPI /analyze endpoint (main.py)
   │
   │  Finds:  data/uploads/a3f9-bc12-....txt
   │  Creates DB row: AnalysisJob(id="a3f9...", status="pending")
   │  Calls:  analyze_contract_core(job_id, file_path, contract_name)
   │
   ▼
tasks.py → analyze_contract_core()
   │
   │  1. Reads file:  text = Path(file_path).read_text()[:5000]
   │                  ↑ RAW STRING, first 5000 chars
   │
   │  2. Calls _load_models() → loads BERT, MiniLM, flan-t5, Qdrant client
   │
   │  3. Calls _run_ner(text)
   │  4. Calls _score_clause() for each entity type found
   │  5. Calls _build_report()
   │  6. Saves everything to SQLite
```

---

### STEP 3 — BERT NER: What Goes In, What Comes Out

```
INPUT:
   text = "This Agreement is made between TechCorp Inc. and John Doe as of March 15,
           2023. Either party may terminate immediately without written notice.
           All work product shall be the exclusive property of Company..."
           (raw string, up to 5000 characters)

   ↓ fed to:

_NER_TOKENIZER(
    text,
    max_length=256,
    stride=128,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,    ← CRITICAL: gives char positions per token
    padding="max_length"
)

   ↓ produces:

enc = {
    "input_ids":       tensor([[101, 2023, 3820, 2003, ...], [101, ...]]),
                       # shape: [num_windows, 256]
                       # integers (token IDs from vocab)

    "attention_mask":  tensor([[1, 1, 1, ..., 0, 0], [...]]),
                       # shape: [num_windows, 256]
                       # 1=real token, 0=padding

    "token_type_ids":  tensor([[0, 0, 0, ..., 0], [...]]),
                       # shape: [num_windows, 256]
                       # all 0 for single-sentence input
}

offsets = tensor([[[0,0],[0,4],[5,9],[10,12],...], [...]])
          # shape: [num_windows, 256, 2]
          # (char_start, char_end) for every token in original string
          # (0,0) = special token [CLS]/[SEP]/[PAD]

   ↓ for each window:

_NER_MODEL(input_ids, attention_mask, token_type_ids)
   ↓ produces:

logits = tensor([[[0.12, -0.45, 0.78, ..., 0.03],   ← 17 scores for token 0
                  [-0.32, 0.67, 0.11, ..., -0.21],   ← 17 scores for token 1
                  ...]])                               ← one row per token
# shape: [1, 256, 17]

   ↓ argmax over dim=-1:

preds = [0, 0, 0, 1, 2, 2, 2, 0, 8, 9, 0, ...]
        # 256 integers, each is a label ID
        # 0=O, 1=B-Parties, 2=I-Parties, ... 8=B-Termination, etc.

   ↓ decode using offsets + ID2LABEL:

For each (pred_id, (char_start, char_end)):
  - pred_id=1 (B-Parties), char_start=34 → entity starts at char 34
  - pred_id=2 (I-Parties), char_start=42 → entity continues
  - pred_id=0 (O), char_start=51        → entity ended at char 50
  → span = text[34:51] = "TechCorp Inc."
  → entities["Parties"].append("TechCorp Inc.")

OUTPUT of _run_ner():
{
    "Parties":        ["TechCorp Inc.", "John Doe"],
    "Agreement_Date": ["March 15, 2023"],
    "Termination":    ["terminate immediately without written notice"],
    "IP_Ownership":   ["exclusive property of Company"]
}
# A plain Python dict: {entity_type: [list of text spans]}
# These are SUBSTRINGS of the original contract text
```

---

### STEP 4 — MiniLM Embeddings for RAG: What Goes In, What Comes Out

```
This happens INSIDE _score_clause(), called once per entity type.

INPUT to _rag_search():
   query = "terminate immediately without written notice"
           (a string — the extracted NER entity text)

   ↓ fed to:

_EMBEDDER.embed_query(query)
   which calls:
   all-MiniLM-L6-v2.encode(query)

   ↓ produces:

qvec = [0.12, -0.45, 0.78, 0.03, -0.21, 0.56, ...]
       # 384 floats — the SEMANTIC MEANING of the clause as a vector

   ↓ fed to:

_QDRANT.query_points(
    collection_name="contracts",
    query=qvec,          ← the 384-float query vector
    limit=2,             ← return top 2 most similar stored chunks
    with_payload=True
)

   ↓ Qdrant computes cosine_similarity(qvec, stored_vector) for every stored point
   ↓ returns top 2 highest similarity points

OUTPUT of _rag_search():
[
    {
        "source": "LIMEENERGYCO_1999.txt",
        "type":   "Distributor",
        "text":   "Either party may terminate this Agreement, with or without cause,
                   upon ninety (90) days prior written notice..."
    },
    {
        "source": "WHITESMOKE_INC_2011.txt",
        "type":   "Licensing",
        "text":   "This Agreement may be terminated by either party upon thirty (30)
                   days written notice..."
    }
]
# A list of dicts — similar market clauses retrieved from CUAD contracts
```

---

### STEP 5 — flan-t5: What Goes In, What Comes Out

```
INPUT to _LLM.invoke():
   The complete prompt string built from:
     - clause_type  = "Termination"
     - clause_text  = "terminate immediately without written notice"
     - similar[0]   = the first retrieved CUAD chunk
     - similar[1]   = the second retrieved CUAD chunk

   prompt = """You are a legal risk analyst. Score this clause 1-10.
   CLAUSE TYPE: Termination
   TEXT: terminate immediately without written notice
   SIMILAR MARKET CLAUSES:
   - [Distributor] Either party may terminate upon ninety (90) days notice...
   - [Licensing] This Agreement may be terminated upon thirty (30) days notice...
   1=low risk, 10=high risk. Reply: SCORE: <n> REASONING: <sentence>"""

   ↓ fed to:

flan-t5-base text-generation pipeline
(max_new_tokens=256, do_sample=False)

   ↓ produces:

resp = "SCORE: 8 REASONING: Immediate termination with no notice period
        is highly unusual compared to standard 30-90 day market practice."

   ↓ parsed with regex:

score  = 8
reason = "Immediate termination with no notice period is highly unusual..."

OUTPUT of _score_clause():
{
    "score":          8,
    "reasoning":      "Immediate termination with no notice period...",
    "text":           "terminate immediately without written notice",
    "similar_clauses": [{"source": "LIMEENERGYCO...", "text": "..."},
                        {"source": "WHITESMOKE...",   "text": "..."}]
}
```

---

### STEP 6 — Report Builder: What Goes In, What Comes Out

```
INPUT to _build_report():
   contract_name = "consulting_agreement"
   entities      = {"Parties": [...], "Termination": [...], "IP_Ownership": [...]}
   risk_scores   = {
       "Parties":     {"score": 2, "reasoning": "...", "text": "..."},
       "Termination": {"score": 8, "reasoning": "...", "text": "..."},
       "IP_Ownership":{"score": 7, "reasoning": "...", "text": "..."}
   }
   flags = ["Termination", "IP_Ownership"]  ← clauses with score > 7

OUTPUT: a Markdown string like:
"# Legal Contract Risk Analysis Report
 **Contract:** consulting_agreement
 **Overall Risk Score:** 5.7/10  🟡 MEDIUM
 ...
 ### Termination ⚠️
 **Score:** [████████░░] 8/10  🔴 HIGH
 ..."
```

---

### STEP 7 — SQLite Save: What is Stored Where

```
AnalysisJob row updated in analysis_jobs table:

  id             = "a3f9-bc12-..."         (UUID, primary key)
  contract_name  = "consulting_agreement"
  status         = "done"
  overall_score  = 5.67                    (average of all clause scores)
  report_md      = "# Legal Contract..."   (full Markdown string, TEXT column)
  entities_json  = '{"Parties": [...], "Termination": [...]}'   (JSON string)
  risk_json      = '{"Parties": {"score":2,...}, "Termination": {"score":8,...}}'
  needs_human_review = 1                   (because score > 7 found)
```

---

### STEP 8 — GET /report: What Goes Out to Frontend

```
Frontend
   │
   │  HTTP GET /report/a3f9-bc12-...
   ▼
FastAPI /report endpoint
   │
   │  db.query(AnalysisJob).filter(id="a3f9...").first()
   │  json.loads(job.entities_json)  → Python dict
   │  json.loads(job.risk_json)      → Python dict
   │
   │  Returns JSON:
   ▼
{
    "job_id":             "a3f9-bc12-...",
    "contract_name":      "consulting_agreement",
    "status":             "done",
    "overall_score":      5.67,
    "needs_human_review": true,
    "entities": {
        "Parties":         ["TechCorp Inc.", "John Doe"],
        "Termination":     ["terminate immediately without written notice"],
        "IP_Ownership":    ["exclusive property of Company"]
    },
    "risk_scores": {
        "Parties":     {"score": 2, "reasoning": "...", "text": "..."},
        "Termination": {"score": 8, "reasoning": "...", "text": "..."},
        "IP_Ownership":{"score": 7, "reasoning": "...", "text": "..."}
    },
    "report_markdown": "# Legal Contract Risk Analysis Report\n..."
}
   │
   ▼
Frontend /report/[id]/page.tsx
   Renders risk cards, entity grid, score badge from this JSON
```

---

### STEP 9 — Chat (SSE): What Goes In, What Comes Out

```
User types: "What is the termination clause?"
   │
   │  EventSource connects to:
   │  GET /ask/stream?job_id=a3f9-bc12-...&question=What+is+the+termination+clause%3F
   ▼
FastAPI /ask/stream endpoint
   │
   │  Loads from SQLite:
   │    entities  = {"Parties": [...], "Termination": [...]}
   │    risk_data = {"Termination": {"score":8, "reasoning":"...", "text":"..."}}
   │    report    = "# Legal Contract..."
   │
   │  Stage 1 — Entity keyword match:
   │    q_lower = "what is the termination clause?"
   │    keyword "termination" found in entity type "Termination"
   │    → direct_hits = ["terminate immediately without written notice"]
   │
   │  Stage 2 — Risk score keyword match:
   │    keyword "termination" found in risk_data key "Termination"
   │    → risk_answer = "Termination: score 8/10 — Immediate termination..."
   │
   │  Stage 3 — Report line search:
   │    search_words = ["what", "termination", "clause"] (words len > 3)
   │    Scan report lines containing these words
   │    → relevant_lines = ["### Termination ⚠️", "Score: 8/10 HIGH"]
   │
   │  Assembles final answer string
   │  Splits into words: ["Found", "the", "following", "entities:", ...]
   │
   │  Streams word by word as SSE:
   ▼
data: {"token": "Found "}
data: {"token": "the "}
data: {"token": "following "}
...every 40ms...
data: [DONE]
   │
   ▼
Chat.tsx EventSource.onmessage handler
   Appends each token to the last message in state
   → User sees text appearing word by word (ChatGPT effect)
```

---

## 🗺️ FULL SYSTEM MAP (One Diagram)

```
User Browser (Next.js)
│
│ POST /upload (multipart)
│──────────────────────────────────────────────────────▶ FastAPI
│                                                          │
│                                                          │ Save file to disk
│                                                          │ data/uploads/<uuid>.txt
│                                                          │
│ POST /analyze (JSON: job_id)                            │
│──────────────────────────────────────────────────────▶  │
│                                                          │
│                                                     tasks.py: analyze_contract_core()
│                                                          │
│                                              ┌───────────▼────────────┐
│                                              │  READ FILE             │
│                                              │  text = file[:5000]    │
│                                              │  (raw string)          │
│                                              └───────────┬────────────┘
│                                                          │ text (string)
│                                              ┌───────────▼────────────┐
│                                              │  BERT NER              │
│                                              │  Input: text string    │
│                                              │  Tokenize → Windows    │
│                                              │  Forward pass → logits │
│                                              │  argmax → label IDs    │
│                                              │  decode via offsets    │
│                                              │  Output: {type:[spans]}│
│                                              └───────────┬────────────┘
│                                                          │ entities dict
│                                              ┌───────────▼────────────┐
│                                              │  For each entity type: │
│                                              │                        │
│                                              │  1. MiniLM embeds the  │
│                                              │     extracted text →   │
│                                              │     384-float vector   │
│                                              │                        │
│                                              │  2. Qdrant ANN search  │
│                                              │     Input: 384-float   │
│                                              │     Output: top-2 CUAD │
│                                              │     chunks             │
│                                              │                        │
│                                              │  3. Build prompt with  │
│                                              │     clause + chunks    │
│                                              │                        │
│                                              │  4. flan-t5 generates  │
│                                              │     "SCORE: X REASON:" │
│                                              │                        │
│                                              │  Output: {score, text} │
│                                              └───────────┬────────────┘
│                                                          │ risk_scores dict
│                                              ┌───────────▼────────────┐
│                                              │  Build Markdown report │
│                                              │  Save to SQLite        │
│                                              │  status="done"         │
│                                              └───────────┬────────────┘
│                                                          │
│ GET /report/<id>                                         │
│◀──────────────────────────────────────────────────────── │
│ ← JSON: entities, risk_scores, report_md, overall_score  │
│                                                          │
│ Render report page                                       │
│                                                          │
│ GET /ask/stream?job_id=...&question=...                  │
│──────────────────────────────────────────────────────▶  │
│                                                          │ Load pre-computed
│                                                          │ entities + risk_scores
│                                                          │ from SQLite
│                                                          │ (NO model re-run)
│                                                          │
│◀ ── SSE stream: word-by-word answer ─────────────────── │
│ data: {"token":"Found "}\n\n                             │
│ data: {"token":"entities"}\n\n                           │
│ ...                                                      │
│ data: [DONE]\n\n                                         │
```

---

## 🔑 Key Clarification Points

### Q: Does BERT NER produce embeddings that go into Qdrant?
**No.** BERT NER produces **label predictions** (which token is which entity type). The output of BERT NER is a dict of `{entity_type: [text_spans]}`. These spans are strings, not vectors. Nothing from BERT NER enters Qdrant.

### Q: What goes into Qdrant?
Two things at two different times:
1. **Offline (setup):** Chunks of the 10 CUAD sample contracts → embedded by MiniLM → stored as 384-float vectors with metadata in Qdrant's `contracts` collection.
2. **Online (per analysis):** The extracted NER entity text span (e.g., "terminate immediately") → embedded by MiniLM → used as a query vector to search Qdrant → returns similar market clauses.

### Q: What does flan-t5 receive?
A plain **text string** (the prompt). Not vectors, not tokenized tensors. LangChain's `HuggingFacePipeline` handles tokenization internally before feeding to flan-t5.

### Q: What is stored in SQLite?
The final processed outputs only:
- `entities_json` — JSON string of the BERT NER result dict
- `risk_json` — JSON string of all clause scores + reasoning from flan-t5
- `report_md` — The full Markdown report string
- `overall_score` — A single float (average)

**Raw text is NOT stored in SQLite.** Only the file on disk and the processed JSON.

### Q: What does the frontend receive from the backend?
JSON only. The frontend never touches the raw contract text, the model, or the vector DB. It only reads from the SQLite-backed REST API response.

### Q: What is stored in Qdrant?
Chunks of the **10 pre-indexed CUAD sample contracts** — not the user's uploaded contract. When a user uploads a new contract, it goes through BERT NER → risk scoring, but its chunks are NOT added to Qdrant. Qdrant is the static "market reference" database.
