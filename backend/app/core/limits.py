"""Wall-clock task limits that actually fire under the ``solo`` worker pool.

Celery implements ``task_soft_time_limit`` / ``task_time_limit`` inside the
*pool*, and only the prefork pool implements them. ``celery.concurrency.solo``
dispatches through ``celery.concurrency.base.apply_target``::

    def apply_target(target, args=(), kwargs=None, callback=None,
                     accept_callback=None, pid=None, getpid=os.getpid,
                     propagate=(), monotonic=time.monotonic, **_):

The ``soft_timeout``/``timeout`` keywords the worker passes in land in ``**_``
and are discarded. So with ``worker_pool="solo"`` — which this project must use,
because ESM-2/PyTorch is not fork-safe — both settings were silently inert, and
a wedged task held the single worker slot forever.

Two mechanisms replace them, because neither alone is sufficient:

``SIGALRM`` backstop
    A timer armed in the worker's main thread before every task, raising
    Celery's own :class:`~celery.exceptions.SoftTimeLimitExceeded` inside the
    running task so the failure reaches the client exactly as it would under
    prefork. Generic: covers every task, including ones with no progress loop.

:class:`Deadline`
    A cooperative check callers invoke from a loop they already run (the
    site-saturation scan's progress callback). Precise and portable, and it
    still fires when the signal handler cannot.

Known limits, deliberately not papered over:

* The *hard* limit (``task_time_limit``) still needs prefork — nothing short of
  killing the process can guarantee it. What is enforced here is the soft limit.
* A Python signal handler only runs between bytecodes, so a task blocked inside
  one long C call (a single large PyTorch matmul) sees the exception when that
  call returns. Overshoot is bounded by one ESM-2 forward pass, i.e. seconds.
* If another component already owns ``SIGALRM`` (billiard does, under prefork)
  the backstop stands down rather than stealing the handler.
"""

from __future__ import annotations

import logging
import signal
import threading
import time

from celery.exceptions import SoftTimeLimitExceeded
from celery.signals import task_postrun, task_prerun

logger = logging.getLogger(__name__)

_HAS_ITIMER = hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")


class Deadline:
    """Cooperative wall-clock budget for a long task.

    ``check()`` is cheap enough to call once per scan candidate; it raises
    :class:`SoftTimeLimitExceeded` once the budget is spent, which Celery reports
    as a task failure and the client renders as an error.
    """

    def __init__(self, seconds: float | None) -> None:
        self.seconds = float(seconds) if seconds else None
        self.started_at = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def expired(self) -> bool:
        return self.seconds is not None and self.elapsed > self.seconds

    def check(self) -> None:
        if self.expired:
            raise SoftTimeLimitExceeded(
                f"Task exceeded its {self.seconds:.0f}s time budget after "
                f"{self.elapsed:.0f}s and was stopped. The worker runs one job at a "
                f"time, so an unbounded job blocks every other user. Try a shorter "
                f"sequence, or raise TASK_SOFT_TIME_LIMIT."
            )


def soft_time_limit(task) -> int | None:
    """The soft limit that applies to ``task``, or ``None`` when unset."""
    limit = getattr(task, "soft_time_limit", None)
    if limit is None:
        limit = task.app.conf.task_soft_time_limit
    return int(limit) if limit else None


def _owns_sigalrm() -> bool:
    """True when nothing else has claimed ``SIGALRM``.

    Under prefork, billiard installs its own handler to implement the real time
    limit. Overwriting it would break the mechanism we are standing in for, so
    the backstop only arms itself when the slot is free.
    """
    if not _HAS_ITIMER:
        return False
    if threading.current_thread() is not threading.main_thread():
        # signal.signal() raises ValueError off the main thread.
        return False
    current = signal.getsignal(signal.SIGALRM)
    return current in (signal.SIG_DFL, signal.SIG_IGN, None)


def _raise_soft_time_limit(signum, frame):  # noqa: ARG001 - signal handler ABI
    raise SoftTimeLimitExceeded(
        "Task exceeded its soft time limit and was stopped. The worker runs one "
        "job at a time, so an unbounded job blocks every other user. Try a "
        "shorter sequence, or raise TASK_SOFT_TIME_LIMIT."
    )


@task_prerun.connect
def _arm_soft_time_limit(task=None, **_) -> None:
    if task is None:
        return
    limit = soft_time_limit(task)
    if not limit or not _owns_sigalrm():
        return
    signal.signal(signal.SIGALRM, _raise_soft_time_limit)
    signal.setitimer(signal.ITIMER_REAL, limit)


@task_postrun.connect
def _disarm_soft_time_limit(**_) -> None:
    if not _HAS_ITIMER or threading.current_thread() is not threading.main_thread():
        return
    if signal.getsignal(signal.SIGALRM) is not _raise_soft_time_limit:
        return
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
