# backend/main.py
"""
LegalEagle FastAPI Backend
Endpoints:
  POST /upload          — upload a contract file
  POST /analyze         — run full AI analysis (sync)
  GET  /report/{id}     — fetch report from SQLite
  POST /ask             — RAG question-answering on a contract
  GET  /ask/stream      — streaming SSE chat
"""
import json, re, uuid, shutil, asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, AsyncGenerator

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import init_db, get_db, AnalysisJob, QARecord

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="LegalEagle API", version="1.0.0")

import os

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.on_event("startup")
def startup():
    init_db()

# ── Celery import (lazy — only needed for /analyze) ────────────────────────────
def _get_celery_task():
    from tasks import run_analysis
    return run_analysis

# ── Schemas ────────────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    job_id: str

class AskRequest(BaseModel):
    job_id: str
    question: str

# ══════════════════════════════════════════════════════════════════════════════
# POST /upload
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/upload", summary="Upload a contract file (.txt or .pdf)")
async def upload(file: UploadFile = File(...)):
    allowed = {".txt", ".pdf"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Only {allowed} files accepted.")

    job_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{job_id}{suffix}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {
        "job_id": job_id,
        "filename": file.filename,
        "path": str(dest),
        "message": "File uploaded. Call POST /analyze with this job_id to start analysis.",
    }


# ══════════════════════════════════════════════════════════════════════════════
# POST /analyze
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/analyze", summary="Enqueue async analysis for an uploaded contract")
def analyze(req: AnalyzeRequest, db: Session = Depends(get_db)):
    # Find the uploaded file
    matches = list(UPLOAD_DIR.glob(f"{req.job_id}.*"))
    if not matches:
        raise HTTPException(404, f"No upload found for job_id={req.job_id}")
    file_path = str(matches[0])
    contract_name = matches[0].stem

    # Create DB record
    job = AnalysisJob(
        id=req.job_id,
        contract_name=contract_name,
        status="pending",
    )
    db.add(job)
    db.commit()

    # Always run synchronously (reliable for demo).
    # Celery/Redis queuing is the production pattern — same code, just async.
    job.status = "running"
    db.commit()

    try:
        from tasks import analyze_contract_core
        analyze_contract_core(req.job_id, file_path, contract_name)
    except Exception as e:
        job.status = "failed"
        db.commit()
        raise HTTPException(500, f"Analysis failed: {str(e)}")

    db.expire_all()
    db.refresh(job)
    return {
        "job_id": req.job_id,
        "status": job.status,
        "overall_score": job.overall_score,
        "message": "Analysis complete! Call GET /report/{job_id} for full results.",
    }


# ══════════════════════════════════════════════════════════════════════════════
# GET /report/{id}
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/report/{job_id}", summary="Fetch analysis report from SQLite")
def get_report(job_id: str, db: Session = Depends(get_db)):
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(404, f"No job found with id={job_id}")

    response = {
        "job_id": job_id,
        "contract_name": job.contract_name,
        "status": job.status,
        "overall_score": job.overall_score,
        "needs_human_review": bool(job.needs_human_review),
        "created_at": str(job.created_at),
    }
    if job.status == "done":
        response["entities"] = json.loads(job.entities_json or "{}")
        response["risk_scores"] = json.loads(job.risk_json or "{}")
        response["report_markdown"] = job.report_md
    return response


# ══════════════════════════════════════════════════════════════════════════════
# POST /ask
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/ask", summary="Ask a free-form question about an analysed contract (RAG)")
def ask(req: AskRequest, db: Session = Depends(get_db)):
    job = db.query(AnalysisJob).filter(AnalysisJob.id == req.job_id).first()
    if not job:
        raise HTTPException(404, f"No job found with id={req.job_id}")

    DONE_STATUSES = {"done", "running_sync"}
    if job.status not in DONE_STATUSES:
        raise HTTPException(
            400,
            f"Analysis not complete yet (status={job.status}). "
            f"Wait for status to be 'done' then retry."
        )

    entities   = json.loads(job.entities_json or "{}")
    risk_data  = json.loads(job.risk_json or "{}")
    report     = job.report_md or ""
    q_lower    = req.question.lower()

    # ── 1. Direct entity match ────────────────────────────────────────────────
    direct_hits = []
    for etype, spans in entities.items():
        keywords = etype.lower().replace("_", " ").split()
        if any(kw in q_lower for kw in keywords):
            direct_hits.extend(spans[:3])

    # ── 2. Risk score match ───────────────────────────────────────────────────
    risk_answer = ""
    for clause, data in risk_data.items():
        keywords = clause.lower().replace("_", " ").split()
        if any(kw in q_lower for kw in keywords):
            risk_answer = (
                f"{clause}: score {data['score']}/10 — {data.get('reasoning','')[:150]}"
                f" | Extracted: {data.get('text','')[:100]}"
            )
            break

    # ── 3. Search raw report lines ────────────────────────────────────────────
    search_words = [w for w in q_lower.split() if len(w) > 3]
    relevant_lines = [
        line.strip() for line in report.split("\n")
        if line.strip() and any(w in line.lower() for w in search_words)
        and not line.startswith("#") and not line.startswith("|")
    ]

    # ── Build final answer ────────────────────────────────────────────────────
    parts = []
    if direct_hits:
        parts.append(f"Extracted entities: {'; '.join(dict.fromkeys(direct_hits))[:200]}")
    if risk_answer:
        parts.append(risk_answer)
    if relevant_lines:
        parts.append("From report: " + " | ".join(relevant_lines[:2])[:200])

    answer = "\n".join(parts) if parts else "Not found in this contract's analysis."

    # ── Store Q&A ─────────────────────────────────────────────────────────────
    qa = QARecord(job_id=req.job_id, question=req.question, answer=answer)
    db.add(qa)
    db.commit()

    return {"job_id": req.job_id, "question": req.question, "answer": answer}


# ══════════════════════════════════════════════════════════════════════════════
# GET /ask/stream  (Server-Sent Events — streaming chat)
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/ask/stream", summary="Streaming chat about a contract (SSE)")
async def ask_stream(job_id: str = Query(...), question: str = Query(...),
                     db: Session = Depends(get_db)):
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(404, f"No job found with id={job_id}")

    DONE_STATUSES = {"done", "running_sync"}
    if job.status not in DONE_STATUSES:
        raise HTTPException(400, f"Analysis not complete (status={job.status})")

    entities  = json.loads(job.entities_json or "{}")
    risk_data = json.loads(job.risk_json or "{}")
    report    = job.report_md or ""
    q_lower   = question.lower()

    # Build answer parts (same logic as /ask)
    direct_hits = []
    for etype, spans in entities.items():
        keywords = etype.lower().replace("_", " ").split()
        if any(kw in q_lower for kw in keywords):
            direct_hits.extend(spans[:3])

    risk_answer = ""
    for clause, data in risk_data.items():
        keywords = clause.lower().replace("_", " ").split()
        if any(kw in q_lower for kw in keywords):
            reason = data.get('reasoning','').replace('<sentence>', 'Standard clause.')
            risk_answer = (
                f"**{clause.replace('_', ' ')}**\n"
                f"Risk Score: {data['score']}/10\n"
                f"Reasoning: {reason[:150]}\n"
                f"Extracted Text: \"{data.get('text','')[:100]}...\""
            )
            break

    search_words = [w for w in q_lower.split() if len(w) > 3]
    relevant_lines = [
        line.strip() for line in report.split("\n")
        if line.strip() and any(w in line.lower() for w in search_words)
        and not line.startswith("#") and not line.startswith("|")
    ]

    parts = []
    if direct_hits:
        parts.append(f"Found the following entities: {', '.join(dict.fromkeys(direct_hits))[:200]}")
    if risk_answer:
        parts.append(risk_answer)
    if relevant_lines:
        parts.append("Additional context from report:\n- " + "\n- ".join(relevant_lines[:2])[:200])
    
    full_answer = "\n\n".join(parts) if parts else "I couldn't find specific information regarding that in the analyzed contract."

    # Store
    qa = QARecord(job_id=job_id, question=question, answer=full_answer)
    db.add(qa)
    db.commit()

    # Stream word by word with SSE
    async def event_generator() -> AsyncGenerator[str, None]:
        words = full_answer.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield f"data: {json.dumps({'token': chunk})}\n\n"
            await asyncio.sleep(0.04)   # 40ms between words = natural typing speed
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
