from __future__ import annotations

import re

# Whole numbers only: the "504" inside "length 5041" is not a gateway timeout.
_GATEWAY_TIMEOUT = re.compile(r"\b504\b")
_UNPROCESSABLE = re.compile(r"\b422\b")


def format_task_error(error: object | None) -> str:
    if error is None:
        return "Task failed with no error details."

    message = str(error).strip()
    if not message:
        return "Task failed with no error details."

    lowered = message.lower()
    if _GATEWAY_TIMEOUT.search(message) or "gateway timeout" in lowered:
        return (
            "The structure prediction service timed out. This is a server-side "
            "problem, not your sequence — retry in a few minutes."
        )
    if _UNPROCESSABLE.search(message) or "rejected this sequence" in lowered:
        return (
            "ESMFold rejected this sequence. Use a valid protein sequence "
            "between 10 and 400 residues."
        )
    if "exceeds the esmfold limit" in lowered:
        return message
    if "invalid amino acid" in lowered:
        return message
    if "timed out" in lowered:
        return message
    if "could not reach the esmfold" in lowered:
        return message
    if "esmfold api error" in lowered:
        return message

    # Celery often wraps exceptions — surface the last line when readable.
    if "HTTPError" in message or "HTTPStatusError" in message:
        return (
            "Structure folding failed due to an upstream API error. "
            "Please retry in a few minutes."
        )

    return message
