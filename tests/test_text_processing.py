from src.utils.text_processing import chunk_text


def test_chunk_text_splits_long_content():
    text = "word " * 300
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) > 1
    assert all(len(c) <= 110 for c in chunks)


def test_chunk_text_short_passthrough():
    text = "Short document."
    assert chunk_text(text) == [text]
