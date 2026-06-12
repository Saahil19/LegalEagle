# backend/celery_app.py
from celery import Celery

celery_app = Celery(
    "legaleagle",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=86400,          # 1 day
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)
