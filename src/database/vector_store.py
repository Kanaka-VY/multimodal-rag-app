from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.utils.config import get_db_config, get_settings


class VectorStore:
    def __init__(self) -> None:
        db_cfg = get_db_config().get("chroma", {})
        settings = get_settings()
        persist = db_cfg.get("persist_directory", "./chroma_data")
        self.collection_name = db_cfg.get("collection_name", "multimodal_rag")

        self._client = chromadb.PersistentClient(
            path=persist,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._host = settings.chroma_host
        self._port = settings.chroma_port

    def add_documents(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> list[str]:
        ids = [str(uuid.uuid4()) for _ in texts]
        stamped = []
        for meta in metadatas:
            m = dict(meta)
            m.setdefault("created_at", datetime.now(timezone.utc).isoformat())
            stamped.append(m)

        self._collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=stamped,
        )
        return ids

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        return self._collection.query(**kwargs)

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
