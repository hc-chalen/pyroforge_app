"""Generate a bcrypt hash for a PyroForge account.

Usage (on the server, from the repo root):

    docker compose -f docker-compose.yml -f docker-compose.auth.yml \
        run --rm --no-deps auth-service python hashpw.py alice

Prints a YAML line to paste into auth/users.yml. The password is read from a
prompt, never from argv, so it does not land in your shell history.
"""

from __future__ import annotations

import getpass
import sys

import bcrypt

# Kept low deliberately: this is a short, non-public trial deployment behind a
# rate-limited login, and the issued accounts use passwords of about this
# length. Raise it if the deployment ever outlives the trial.
MIN_LENGTH = 8


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: python {sys.argv[0]} <username>", file=sys.stderr)
        return 2

    username = sys.argv[1].strip()
    if not username:
        print("Username must not be empty.", file=sys.stderr)
        return 2

    password = getpass.getpass(f"Password for {username!r}: ")
    if len(password) < MIN_LENGTH:
        print(
            f"Password must be at least {MIN_LENGTH} characters.",
            file=sys.stderr,
        )
        return 2
    if password != getpass.getpass("Repeat password: "):
        print("Passwords do not match.", file=sys.stderr)
        return 2

    digest = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12))
    print("\nAdd this line under 'users:' in auth/users.yml:\n")
    print(f'  {username}: "{digest.decode("ascii")}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
