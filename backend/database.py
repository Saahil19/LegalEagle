# backend/database.py
import sqlite3
from sqlalchemy import create_engine, Column, String, Integer, Float, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/legaleagle.db")

# If using SQLite, we need check_same_thread=False. For PostgreSQL, we don't.
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    id            = Column(String, primary_key=True, index=True)
    contract_name = Column(String)
    status        = Column(String, default="pending")   # pending | running | done | failed
    celery_task_id = Column(String, nullable=True)
    created_at    = Column(DateTime, default=datetime.utcnow)
    updated_at    = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    overall_score = Column(Float, nullable=True)
    report_md     = Column(Text, nullable=True)         # full Markdown report
    entities_json = Column(Text, nullable=True)         # JSON string
    risk_json     = Column(Text, nullable=True)         # JSON string
    needs_human_review = Column(Integer, default=0)     # 0 or 1


class QARecord(Base):
    __tablename__ = "qa_records"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    job_id     = Column(String, index=True)
    question   = Column(Text)
    answer     = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
