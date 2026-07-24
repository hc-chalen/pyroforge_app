#!/usr/bin/env python3
"""Bootstrap iGEM parts FASTA for MMseqs2 indexing."""

from app.engines.registry_engine import MMseqsService


def main() -> None:
    service = MMseqsService()
    service.ensure_database()
    print(f"iGEM parts FASTA: {service.fasta_path}")
    print(f"MMseqs2 DB: {service.db_path}")


if __name__ == "__main__":
    main()
