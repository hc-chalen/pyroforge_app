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
    # The pool is single-slot, so one runaway job blocks every other user
    # indefinitely — hence the limits below. NOTE: Celery implements these in the
    # *pool*, and only prefork implements them; the solo pool discards the
    # timeouts the worker hands it (see app.core.limits for the exact code path).
    # They are declared here because `app.core.limits` reads them back and
    # enforces the soft limit itself. Without that module they are inert.
    task_soft_time_limit=settings.task_soft_time_limit,
    task_time_limit=settings.task_time_limit,
)

celery_app.autodiscover_tasks(["app.tasks"])

# Registers the task_prerun/task_postrun receivers that make task_soft_time_limit
# real under the solo pool. Imported for the side effect; must come after the
# config above so the receivers read the final settings.
from app.core import limits  # noqa: E402,F401
