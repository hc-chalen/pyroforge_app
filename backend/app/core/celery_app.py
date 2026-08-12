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
    # ------------------------------------------------------------------
    # Queue routing.
    #
    # Task durations span two orders of magnitude — a prediction is ~0.6 s once
    # ESM-2 is resident, a site-saturation scan is ~150 s on a 66-mer. On one
    # single-slot worker every short job queues behind whatever long job is
    # running (head-of-line blocking): with a class running the UAT together,
    # one classmate's optimization stalls everyone else's Tm prediction. Three
    # queues, one worker each, so the classes cannot block each other:
    #
    #   fast   predict_sequence, registry_search   ~1 s, ESM-2 resident
    #   heavy  optimize_sequence, train_models     minutes, CPU/GPU-bound
    #   io     fold_structure                      waits on the ESMFold API
    #
    # `io` is deliberately separate from `heavy`: folding is network-bound and
    # never touches ESM-2, so its worker stays light and can run several folds
    # concurrently on a thread pool.
    #
    # Routing lives here, on the app shared by producer and consumer, so the
    # FastAPI side publishes to the same queues the workers consume without any
    # per-call `queue=` argument. Adding a task without a route sends it to
    # task_default_queue, which is `fast` — deliberately the queue that is
    # always being drained, so a missing route degrades to "runs anyway".
    # ------------------------------------------------------------------
    task_default_queue="fast",
    task_routes={
        "app.tasks.predict_sequence": {"queue": "fast"},
        "app.tasks.registry_search": {"queue": "fast"},
        "app.tasks.optimize_sequence": {"queue": "heavy"},
        "app.tasks.train_models": {"queue": "heavy"},
        "app.tasks.fold_structure": {"queue": "io"},
    },
)

celery_app.autodiscover_tasks(["app.tasks"])

# Registers the task_prerun/task_postrun receivers that make task_soft_time_limit
# real under the solo pool. Imported for the side effect; must come after the
# config above so the receivers read the final settings.
from app.core import limits  # noqa: E402,F401
