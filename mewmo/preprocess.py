"""Convert various file types to plain text for downstream LLM processing."""

import base64
import os
import tempfile
from pathlib import Path

TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".csv", ".json", ".xml", ".html"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".opus"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

# Lazy-loaded Whisper model (expensive to initialise)
_whisper_model = None


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper_model


def _pdf_to_text(data: bytes) -> str:
    import pymupdf
    doc = pymupdf.open(stream=data, filetype="pdf")
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            pages.append(f"[Page {i}]\n{text}")
    return "\n\n".join(pages)


def _audio_to_text(data: bytes, ext: str) -> str:
    model = _get_whisper()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(data)
        tmp_path = f.name
    try:
        segments, _ = model.transcribe(tmp_path)
        return " ".join(seg.text.strip() for seg in segments)
    finally:
        os.unlink(tmp_path)


def _image_to_text(data: bytes, ext: str) -> str:
    """
    Requires the local LLM to support vision (multimodal input).
    Raises RuntimeError if the model returns an error or empty response.
    Note: qwen3.5-8b is text-only. Use a vision-capable model for images.
    """
    from mewmo.llm import chat
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp"}
    mime = mime_map.get(ext.lower(), "image/jpeg")
    b64 = base64.b64encode(data).decode()
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": (
                "Describe all information visible in this image in detail. "
                "Include any text, numbers, labels, or data you can see."
            )},
        ],
    }]
    result = chat(messages, temperature=0.2)
    if not result:
        raise RuntimeError("Image LLM returned empty response. "
                           "Ensure your model supports vision input.")
    return result


def preprocess(data: bytes, extension: str) -> str:
    """Convert raw file bytes to plain text. extension must include leading dot."""
    ext = extension.lower()
    if ext == ".pdf":
        return _pdf_to_text(data)
    if ext in AUDIO_EXTENSIONS:
        return _audio_to_text(data, ext)
    if ext in IMAGE_EXTENSIONS:
        return _image_to_text(data, ext)
    if ext in TEXT_EXTENSIONS:
        return data.decode("utf-8", errors="replace")
    raise ValueError(
        f"Unsupported file extension: {extension!r}. "
        f"Supported: pdf, {', '.join(sorted(AUDIO_EXTENSIONS | IMAGE_EXTENSIONS | TEXT_EXTENSIONS))}"
    )
