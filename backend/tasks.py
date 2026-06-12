# backend/tasks.py
"""
Analysis logic — can be called directly (sync) or via Celery (async).
Core function _analyze_contract() is a plain Python function.
Celery task run_analysis() wraps it for async queue usage.
"""
import json, re, warnings
from pathlib import Path
from datetime import datetime
warnings.filterwarnings("ignore")

from celery_app import celery_app
from database import SessionLocal, AnalysisJob

# ── Lazy-loaded globals (initialised once per process) ────────────────────────
_MODELS_LOADED = False
_NER_TOKENIZER = None
_NER_MODEL     = None
_EMBEDDER      = None
_LLM           = None
_QDRANT        = None

LABELS = [
    "O",
    "B-Parties", "I-Parties",
    "B-Agreement_Date", "I-Agreement_Date",
    "B-Governing_Law", "I-Governing_Law",
    "B-Termination", "I-Termination",
    "B-Indemnification", "I-Indemnification",
    "B-Confidentiality", "I-Confidentiality",
    "B-IP_Ownership", "I-IP_Ownership",
    "B-Non_Compete", "I-Non_Compete",
]
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

RISK_BASELINE = {
    "Termination": 6, "Indemnification": 7, "Non_Compete": 8,
    "IP_Ownership": 7, "Confidentiality": 5, "Governing_Law": 4,
    "Parties": 2, "Agreement_Date": 1,
}


def _load_models():
    global _MODELS_LOADED, _NER_TOKENIZER, _NER_MODEL, _EMBEDDER, _LLM, _QDRANT
    if _MODELS_LOADED:
        return
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline as hf_pipeline
    from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
    from qdrant_client import QdrantClient

    # Resolve model path relative to this file (tasks.py is in backend/)
    # so the model is one level up: LegalEagle/models/bert-ner-cuad-final
    MODEL_PATH = (Path(__file__).parent.parent / "models" / "bert-ner-cuad-final").resolve()
    model_path_str = str(MODEL_PATH)

    _NER_TOKENIZER = AutoTokenizer.from_pretrained(model_path_str)
    _NER_MODEL = AutoModelForTokenClassification.from_pretrained(
        model_path_str, ignore_mismatched_sizes=True
    )
    _NER_MODEL.eval()

    _EMBEDDER = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )

    gen = hf_pipeline("text-generation", model="google/flan-t5-base",
                      max_new_tokens=256, do_sample=False)
    _LLM = HuggingFacePipeline(pipeline=gen)

    try:
        import subprocess
        check = subprocess.run(
            ["docker", "ps", "--filter", "name=qdrant-legal", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=5
        )
        if "qdrant-legal" in check.stdout:
            _QDRANT = QdrantClient(host="localhost", port=6333)
        else:
            _QDRANT = QdrantClient(":memory:")
    except Exception:
        _QDRANT = QdrantClient(":memory:")

    _MODELS_LOADED = True


def _run_ner(text: str) -> dict:
    import torch
    enc = _NER_TOKENIZER(
        text, return_tensors="pt", truncation=True,
        max_length=256, stride=128,
        return_overflowing_tokens=True, return_offsets_mapping=True,
        padding="max_length"
    )
    offsets = enc.pop("offset_mapping")
    enc.pop("overflow_to_sample_mapping", None)
    entities = {}
    with torch.no_grad():
        for i in range(enc["input_ids"].shape[0]):
            chunk = {k: v[i:i+1] for k, v in enc.items()}
            preds = torch.argmax(_NER_MODEL(**chunk).logits, dim=-1)[0].tolist()
            cur_label, cur_start = None, None
            for pred_id, (start, end) in zip(preds, offsets[i].tolist()):
                if start == 0 and end == 0:
                    continue
                label = ID2LABEL.get(pred_id, "O")
                if label.startswith("B-"):
                    if cur_label and cur_start is not None:
                        span = text[cur_start:end].strip()
                        if span:
                            entities.setdefault(cur_label, []).append(span)
                    cur_label, cur_start = label[2:], start
                elif label.startswith("I-") and cur_label == label[2:]:
                    pass
                else:
                    if cur_label and cur_start is not None:
                        span = text[cur_start:start].strip()
                        if span:
                            entities.setdefault(cur_label, []).append(span)
                    cur_label, cur_start = None, None
    return {k: list(dict.fromkeys(v)) for k, v in entities.items()}


def _rag_search(query: str, k: int = 2) -> list:
    try:
        qvec = _EMBEDDER.embed_query(query)
        res = _QDRANT.query_points(
            collection_name="contracts", query=qvec, limit=k, with_payload=True
        )
        return [{"source": p.payload.get("source_file", "?")[:40],
                 "type": p.payload.get("contract_type", "?"),
                 "text": p.payload.get("text", "")[:200]} for p in res.points]
    except Exception as e:
        return [{"source": "N/A", "type": "N/A", "text": str(e)}]


def _score_clause(clause_type: str, texts: list) -> dict:
    clause_text = "; ".join(texts[:2])[:300]
    similar = _rag_search(clause_text)
    ctx = "\n".join(f"- [{r['type']}] {r['text'][:120]}" for r in similar)
    prompt = (
        f"You are a legal risk analyst. Score this clause 1-10.\n"
        f"CLAUSE TYPE: {clause_type}\nTEXT: {clause_text}\n"
        f"SIMILAR MARKET CLAUSES:\n{ctx}\n"
        f"1=low risk, 10=high risk. Reply: SCORE: <n> REASONING: <sentence>"
    )
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


def _build_report(contract_name, entities, risk_scores, flags) -> str:
    scores = list(risk_scores.values())
    avg = sum(d["score"] for d in scores) / len(scores) if scores else 0

    def badge(s):
        return "🟢 LOW" if s <= 3 else ("🟡 MEDIUM" if s <= 7 else "🔴 HIGH")

    lines = [
        "# Legal Contract Risk Analysis Report",
        f"**Contract:** {contract_name}",
        f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Overall Risk Score:** {avg:.1f}/10  {badge(avg)}", "",
        "---", "## Summary",
        f"Analyzed **{len(entities)} clause types**.",
        f"**{len(flags)} HIGH RISK** clause(s) detected.",
    ]
    if flags:
        lines += ["", "> ⚠️ **ATTORNEY REVIEW REQUIRED**",
                  "> High-risk clauses flagged:"]
        for c in flags:
            lines.append(f"> - **{c}**: {risk_scores[c]['score']}/10")

    lines += ["", "---", "## Clause Analysis"]
    for clause, data in sorted(risk_scores.items(), key=lambda x: -x[1]["score"]):
        bar = "█" * data["score"] + "░" * (10 - data["score"])
        hr = " ⚠️" if clause in flags else ""
        lines += [
            f"### {clause}{hr}",
            f"**Score:** `[{bar}]` {data['score']}/10  {badge(data['score'])}",
            f"**Text:** _{data['text']}_",
            f"**Reasoning:** {data['reasoning']}", ""
        ]

    lines += ["---", "## Recommendations"]
    for clause, data in sorted(risk_scores.items(), key=lambda x: -x[1]["score"]):
        s = data["score"]
        if s >= 7:
            lines.append(f"- 🔴 **{clause}** ({s}/10): Seek legal advice before signing.")
        elif s >= 4:
            lines.append(f"- 🟡 **{clause}** ({s}/10): Review carefully.")
        else:
            lines.append(f"- 🟢 **{clause}** ({s}/10): Standard clause.")

    return "\n".join(lines)


# ── CORE FUNCTION (plain Python — no Celery needed) ───────────────────────────
def analyze_contract_core(job_id: str, file_path: str, contract_name: str) -> dict:
    """
    Run the full analysis pipeline. Updates the DB row directly.
    Returns a summary dict.
    """
    db = SessionLocal()
    try:
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if job:
            job.status = "running"
            db.commit()

        _load_models()

        text = Path(file_path).read_text(encoding="utf-8", errors="ignore")[:5000]

        # NER
        entities = _run_ner(text)
        if not entities:
            # Fallback: extract based on keywords from raw text
            entities = {}
            text_lower = text.lower()
            if "terminat" in text_lower:
                entities["Termination"] = ["termination clause present"]
            if "indemnif" in text_lower:
                entities["Indemnification"] = ["indemnification clause present"]
            if "non-compete" in text_lower or "noncompete" in text_lower:
                entities["Non_Compete"] = ["non-compete clause present"]
            if "governing law" in text_lower:
                entities["Governing_Law"] = ["governing law clause present"]
            if "confidential" in text_lower:
                entities["Confidentiality"] = ["confidentiality clause present"]
            if not entities:
                entities = {"General": ["contract analyzed"]}

        # Score each clause
        risk_scores = {ct: _score_clause(ct, spans) for ct, spans in entities.items()}
        flags = [ct for ct, d in risk_scores.items() if d["score"] > 7]
        avg = sum(d["score"] for d in risk_scores.values()) / len(risk_scores)

        # Build report
        report_md = _build_report(contract_name, entities, risk_scores, flags)

        # Save to DB
        if job:
            job.status = "done"
            job.overall_score = round(avg, 2)
            job.report_md = report_md
            job.entities_json = json.dumps(entities)
            job.risk_json = json.dumps(risk_scores)
            job.needs_human_review = 1 if flags else 0
            db.commit()

        return {"job_id": job_id, "status": "done", "overall_score": round(avg, 2)}

    except Exception as exc:
        if job:
            job.status = "failed"
            job.report_md = f"Error: {str(exc)}"
            db.commit()
        raise
    finally:
        db.close()


# ── CELERY TASK WRAPPER ───────────────────────────────────────────────────────
@celery_app.task(name="tasks.run_analysis")
def run_analysis(job_id: str, file_path: str, contract_name: str):
    """Celery-wrapped version of analyze_contract_core."""
    return analyze_contract_core(job_id, file_path, contract_name)
