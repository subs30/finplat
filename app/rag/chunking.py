DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150


def chunk_text(
    text: str, *, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP
) -> list[str]:
    """Split `text` into overlapping chunks of roughly `chunk_size` characters.

    Character-based, not token-based: no tokenizer/model dependency, and
    chunk boundaries stay legible when inspecting stored chunks directly.
    The defaults (1000 chars, ~150 overlap — ~15%) keep each chunk
    comfortably within a single embedding call while the overlap means a
    sentence split across a chunk boundary still appears whole in at least
    one neighboring chunk, so relevant context isn't lost purely because of
    where a cut fell.

    A chunk boundary snaps back to the nearest preceding space (so words
    aren't split), unless no space is found in that window, in which case
    it falls back to a hard cut.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    stripped = text.strip()
    if not stripped:
        return []

    if len(stripped) <= chunk_size:
        return [stripped]

    chunks: list[str] = []
    step = chunk_size - overlap
    start = 0
    while start < len(stripped):
        end = min(start + chunk_size, len(stripped))
        if end < len(stripped):
            snap = stripped.rfind(" ", start, end)
            if snap > start:
                end = snap
        chunk = stripped[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(stripped):
            break
        start += step
    return chunks
