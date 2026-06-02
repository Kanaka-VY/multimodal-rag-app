from typing import Iterator


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            last_space = chunk.rfind(" ")
            if last_space > chunk_size // 2:
                chunk = chunk[:last_space]
                end = start + last_space
        chunks.append(chunk.strip())
        start = max(end - overlap, start + 1)
        if start >= len(text):
            break
    return [c for c in chunks if c]


def iter_pdf_pages(text_pages: list[str]) -> Iterator[tuple[int, str]]:
    for i, page_text in enumerate(text_pages, start=1):
        cleaned = page_text.strip()
        if cleaned:
            yield i, cleaned
