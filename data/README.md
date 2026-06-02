# Data Directory

This directory contains sample files for testing the multimodal RAG application.

## Supported File Types

- **PDF**: `.pdf` - Documents are chunked by page
- **Text**: `.txt` - Documents are chunked by character count
- **Images**: `.png`, `.jpg`, `.jpeg`, `.webp` - Uses CLIP embeddings
- **Audio**: `.wav`, `.mp3`, `.m4a` - Transcribed with Whisper before embedding

## Adding Your Data

Place your files in this directory, then run:

```bash
python scripts/migrate_data.py
```

Or use the Streamlit UI to upload files directly.
