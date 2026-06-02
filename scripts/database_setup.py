#!/usr/bin/env python3
"""Initialize ChromaDB collection and verify connectivity."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.database.vector_store import VectorStore  # noqa: E402


def main() -> None:
    store = VectorStore()
    count = store.count()
    print(f"Collection ready. Current document count: {count}")


if __name__ == "__main__":
    main()
