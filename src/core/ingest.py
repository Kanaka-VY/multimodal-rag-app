from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from src.core.embeddings import embed_image, embed_text
from src.database.vector_store import VectorStore
from src.utils.config import get_model_config
from src.utils.image_processing import image_caption_stub, load_image
from src.utils.monitoring import INGEST_TOTAL
from src.utils.text_processing import chunk_text, iter_pdf_pages

try:
    from unstructured.partition.pdf import partition_pdf
    from unstructured.partition.image import partition_image
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False


class IngestPipeline:
    SUPPORTED = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".wav", ".mp3", ".m4a", ".txt"}

    def __init__(self, store: VectorStore | None = None) -> None:
        self.store = store or VectorStore()
        self._cfg = get_model_config()

    def ingest_path(self, path: Path) -> dict[str, Any]:
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED:
            raise ValueError(f"Unsupported file type: {suffix}")

        if suffix == ".pdf":
            return self._ingest_pdf(path)
        if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            return self._ingest_image(path)
        if suffix in {".wav", ".mp3", ".m4a"}:
            return self._ingest_audio(path)
        return self._ingest_text_file(path)

    def ingest_directory(self, directory: Path) -> list[dict[str, Any]]:
        results = []
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in self.SUPPORTED:
                results.append(self.ingest_path(path))
        return results

    def _ingest_pdf(self, path: Path) -> dict[str, Any]:
        # Use Unstructured.io for advanced parsing if available
        if UNSTRUCTURED_AVAILABLE and self._cfg.get("use_unstructured", True):
            return self._ingest_pdf_unstructured(path)
        
        # Fallback to basic pypdf parsing
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        chunk_cfg = self._cfg.get("chunking", {})
        chunk_size = chunk_cfg.get("text_chunk_size", 512)
        overlap = chunk_cfg.get("text_chunk_overlap", 64)

        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for page_num, page_text in iter_pdf_pages(pages):
            for chunk in chunk_text(page_text, chunk_size, overlap):
                texts.append(chunk)
                metadatas.append(
                    {
                        "source": str(path),
                        "modality": "pdf",
                        "filename": path.name,
                        "page": page_num,
                    }
                )

        if not texts:
            return {"filename": path.name, "chunks": 0, "modality": "pdf"}

        embeddings = embed_text(texts)
        ids = self.store.add_documents(texts, embeddings, metadatas)
        INGEST_TOTAL.labels(modality="pdf").inc()
        return {"filename": path.name, "chunks": len(ids), "modality": "pdf", "ids": ids}

    def _ingest_pdf_unstructured(self, path: Path) -> dict[str, Any]:
        """Advanced PDF parsing with Unstructured.io for tables, charts, and structured content."""
        elements = partition_pdf(
            filename=str(path),
            strategy="hi_res",
            extract_images_in_pdf=True,
            extract_image_block_output_dir=str(path.parent / "extracted_images"),
            extract_table_structure=True,
            chunking_strategy="by_title",
        )
        
        chunk_cfg = self._cfg.get("chunking", {})
        chunk_size = chunk_cfg.get("text_chunk_size", 512)
        overlap = chunk_cfg.get("text_chunk_overlap", 64)
        
        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []
        
        for element in elements:
            element_type = element.category
            element_text = str(element).strip()
            
            if not element_text:
                continue
                
            # Extract metadata from element
            metadata = {
                "source": str(path),
                "modality": "pdf",
                "filename": path.name,
                "element_type": element_type,
                "page_number": getattr(element.metadata, "page_number", 1),
            }
            
            # Handle different element types
            if element_type == "Table":
                # Store tables as structured text
                texts.append(f"Table: {element_text}")
                metadata["is_table"] = True
            elif element_type == "Image":
                # Images are extracted separately
                texts.append(f"Image caption: {element_text}")
                metadata["is_image"] = True
            elif element_type == "Formula":
                texts.append(f"Formula: {element_text}")
                metadata["is_formula"] = True
            else:
                # Regular text chunks
                chunks = chunk_text(element_text, chunk_size, overlap)
                for chunk in chunks:
                    texts.append(chunk)
                    metadatas.append(metadata.copy())
                continue
            
            texts.append(element_text)
            metadatas.append(metadata)
        
        if not texts:
            return {"filename": path.name, "chunks": 0, "modality": "pdf"}
        
        embeddings = embed_text(texts)
        ids = self.store.add_documents(texts, embeddings, metadatas)
        INGEST_TOTAL.labels(modality="pdf").inc()
        return {"filename": path.name, "chunks": len(ids), "modality": "pdf", "ids": ids}

    def _ingest_image(self, path: Path) -> dict[str, Any]:
        image = load_image(path)
        caption = image_caption_stub(path)
        embedding = embed_image(image)
        metadata = {
            "source": str(path),
            "modality": "image",
            "filename": path.name,
            "caption": caption,
        }
        ids = self.store.add_documents([caption], [embedding], [metadata])
        INGEST_TOTAL.labels(modality="image").inc()
        return {"filename": path.name, "chunks": 1, "modality": "image", "ids": ids}

    def _ingest_audio(self, path: Path) -> dict[str, Any]:
        transcript = self._transcribe(path)
        chunk_cfg = self._cfg.get("chunking", {})
        chunks = chunk_text(
            transcript,
            chunk_cfg.get("text_chunk_size", 512),
            chunk_cfg.get("text_chunk_overlap", 64),
        )
        if not chunks:
            return {"filename": path.name, "chunks": 0, "modality": "audio"}

        metadatas = [
            {
                "source": str(path),
                "modality": "audio",
                "filename": path.name,
            }
            for _ in chunks
        ]
        embeddings = embed_text(chunks)
        ids = self.store.add_documents(chunks, embeddings, metadatas)
        INGEST_TOTAL.labels(modality="audio").inc()
        return {"filename": path.name, "chunks": len(ids), "modality": "audio", "ids": ids}

    def _ingest_text_file(self, path: Path) -> dict[str, Any]:
        text = path.read_text(encoding="utf-8", errors="replace")
        chunk_cfg = self._cfg.get("chunking", {})
        chunks = chunk_text(
            text,
            chunk_cfg.get("text_chunk_size", 512),
            chunk_cfg.get("text_chunk_overlap", 64),
        )
        metadatas = [
            {
                "source": str(path),
                "modality": "text",
                "filename": path.name,
            }
            for _ in chunks
        ]
        embeddings = embed_text(chunks)
        ids = self.store.add_documents(chunks, embeddings, metadatas)
        INGEST_TOTAL.labels(modality="text").inc()
        return {"filename": path.name, "chunks": len(ids), "modality": "text", "ids": ids}

    def _transcribe(self, path: Path) -> str:
        try:
            import whisper
        except ImportError as exc:
            raise RuntimeError("openai-whisper is required for audio ingestion") from exc

        model_size = self._cfg.get("whisper", {}).get("model_size", "base")
        model = whisper.load_model(model_size)
        result = model.transcribe(str(path))
        return (result.get("text") or "").strip()
