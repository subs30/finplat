from itertools import pairwise

import pytest

from app.rag.chunking import chunk_text


def test_empty_string_returns_no_chunks():
    assert chunk_text("") == []


def test_whitespace_only_returns_no_chunks():
    assert chunk_text("   \n\t  ") == []


def test_short_text_returns_single_chunk():
    text = "This is a short document."
    assert chunk_text(text, chunk_size=1000, overlap=150) == [text]


def test_text_longer_than_chunk_size_produces_multiple_chunks():
    # 50 words of ~6 chars each + spaces ≈ 300 chars, well over a 100-char chunk.
    text = " ".join(f"word{i:03d}" for i in range(50))
    chunks = chunk_text(text, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    for chunk in chunks:
        # A little slack over chunk_size is fine (whitespace-snap can round
        # down, never up, but .strip() plus the snap logic keeps this safe).
        assert len(chunk) <= 100


def test_chunks_reconstruct_full_content_with_overlap():
    text = " ".join(f"word{i:03d}" for i in range(50))
    chunks = chunk_text(text, chunk_size=100, overlap=20)

    # Every word from the source text appears in at least one chunk — no
    # content is silently dropped between chunks.
    all_chunk_words = " ".join(chunks).split()
    for i in range(50):
        assert f"word{i:03d}" in all_chunk_words


def test_consecutive_chunks_share_overlapping_content():
    text = " ".join(f"word{i:03d}" for i in range(50))
    chunks = chunk_text(text, chunk_size=100, overlap=20)

    for first, second in pairwise(chunks):
        first_words = set(first.split())
        second_words = set(second.split())
        assert first_words & second_words, "consecutive chunks should overlap"


def test_does_not_split_a_word_when_a_space_is_available():
    text = " ".join(f"word{i:03d}" for i in range(50))
    chunks = chunk_text(text, chunk_size=100, overlap=20)

    for chunk in chunks:
        for word in chunk.split():
            assert word.startswith("word") and len(word) == 7


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError, match="overlap must be smaller than chunk_size"):
        chunk_text("hello world", chunk_size=100, overlap=100)


def test_single_long_word_with_no_spaces_falls_back_to_hard_cut():
    text = "a" * 250
    chunks = chunk_text(text, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    assert all(set(chunk) == {"a"} for chunk in chunks)
    assert all(len(chunk) <= 100 for chunk in chunks)
