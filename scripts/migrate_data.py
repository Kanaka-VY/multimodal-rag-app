#!/usr/bin/env python3
"""Bulk-ingest files from data/ into the vector store."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.ingest import IngestPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate raw data into vector store")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data",
        help="Directory containing PDFs, images, audio",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Creating data directory: {args.data_dir}")
        args.data_dir.mkdir(parents=True)

    pipeline = IngestPipeline()
    results = pipeline.ingest_directory(args.data_dir)
    total_chunks = sum(r.get("chunks", 0) for r in results)
    print(f"Ingested {len(results)} files, {total_chunks} chunks total.")


if __name__ == "__main__":
    main()
