# Model Card — Multimodal RAG

## Embeddings

| Model | Use | Dimension |
|-------|-----|-----------|
| `all-MiniLM-L6-v2` | Text & audio (post-Whisper) | 384 |
| `clip-ViT-B-32` | Images & image-style queries | 512 |

## Speech

- **Whisper** (`base` by default) for audio → text before text embedding.

## Generation

| Provider | Behavior |
|----------|----------|
| `local` | Context concatenation + excerpt summary (no API key) |
| `openai` | `gpt-4o-mini` (configurable) with strict context grounding |

## Limitations

- Image ingestion uses filename captions unless a vision LLM is added.
- First run downloads embedding weights (~hundreds of MB).
- Audio ingestion is CPU-heavy with Whisper.

## Ethical use

Index only documents you are permitted to store and query. Do not commit `.env` or API keys.
