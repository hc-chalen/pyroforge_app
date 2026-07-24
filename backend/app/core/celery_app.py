from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "pyroforge",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=settings.celery_result_expires,
    task_track_started=True,
    # Tasks run PyTorch/ESM-2, which is not fork-safe on macOS: the default
    # "prefork" pool aborts child workers with signal 6 (SIGABRT) the moment a
    # forked process touches torch. Default to the non-forking "solo" pool (the
    # same pool docker-compose passes via --pool=solo) so the worker is safe no
    # matter how it is launched. Override with `celery ... --pool=<name>` if you
    # need concurrency on a fork-safe platform.
    worker_pool="solo",
)

celery_app.autodiscover_tasks(["app.tasks"])
