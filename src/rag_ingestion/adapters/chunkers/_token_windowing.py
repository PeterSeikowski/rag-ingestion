"""Pure helpers shared by RecursiveTextChunker and
SectionBasedHierarchicalChunker: token-windowing and provenance
aggregation.

Kept as standalone functions — neither chunker imports the other — so both
strategies stay independently swappable (see IMPLEMENTATION_PLAN.md
decision 2). `tiktoken` is used purely as an approximate, provider-agnostic
tokenizer for chunk *sizing*; it does not need to match whatever tokenizer
the configured embedding model actually uses.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import tiktoken

from rag_ingestion.domain.documents import ParsedElement
from rag_ingestion.domain.provenance import SourceReference


@dataclass(frozen=True)
class TokenWindow:
    """One windowed slice of a larger text, with its character offsets in
    that source text — used to map back to the ParsedElements it overlaps.
    """

    text: str
    start_char: int
    end_char: int
    token_count: int


def window_text(
    text: str,
    *,
    chunk_size_tokens: int,
    overlap_tokens: int,
    min_chunk_size_tokens: int = 0,
    encoding_name: str = "cl100k_base",
) -> list[TokenWindow]:
    """Split `text` into overlapping token windows.

    Returns a single window covering the whole text if it already fits
    within `chunk_size_tokens`, and an empty list for blank input. A
    trailing window smaller than `min_chunk_size_tokens` is merged into the
    previous window rather than emitted as its own tiny chunk.
    """
    if not text.strip():
        return []

    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)

    if len(tokens) <= chunk_size_tokens:
        return [TokenWindow(text=text, start_char=0, end_char=len(text), token_count=len(tokens))]

    char_offsets = _cumulative_char_offsets(encoding, tokens)

    stride = max(chunk_size_tokens - overlap_tokens, 1)
    windows: list[TokenWindow] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size_tokens, len(tokens))
        start_char = char_offsets[start]
        end_char = char_offsets[end]
        windows.append(
            TokenWindow(text=text[start_char:end_char], start_char=start_char, end_char=end_char, token_count=end - start)
        )
        if end == len(tokens):
            break
        start += stride

    if len(windows) > 1 and windows[-1].token_count < min_chunk_size_tokens:
        prev, last = windows[-2], windows[-1]
        merged_text = text[prev.start_char : last.end_char]
        windows[-2] = TokenWindow(
            text=merged_text,
            start_char=prev.start_char,
            end_char=last.end_char,
            token_count=len(encoding.encode(merged_text)),
        )
        windows.pop()

    return windows


def _cumulative_char_offsets(encoding: "tiktoken.Encoding", tokens: list[int]) -> list[int]:
    """offsets[i] = character length of decode(tokens[:i]), for every i in
    0..len(tokens) — one entry per possible window boundary.

    Computed by decoding each token exactly once, in order (O(n) overall),
    rather than re-decoding the growing prefix from scratch for every
    boundary (O(n^2)) — on a several-hundred-page PDF the latter took tens
    of seconds to minutes in practice; this is milliseconds.

    Decoding tokens one at a time is very rarely off by a character right
    at a boundary where a single multi-byte UTF-8 character's bytes are
    split across two BPE tokens (tiktoken's decode() silently substitutes
    a replacement character for an incomplete trailing byte sequence in
    that case). This is an acceptable trade-off given tiktoken is already
    only an approximate, provider-agnostic tokenizer here for *sizing* —
    see the module docstring — not a source of exact byte-for-byte
    provenance.
    """
    offsets = [0] * (len(tokens) + 1)
    decoded_len = 0
    for i, token in enumerate(tokens, start=1):
        decoded_len += len(encoding.decode([token]))
        offsets[i] = decoded_len
    return offsets


def token_count(text: str, encoding_name: str = "cl100k_base") -> int:
    """Token count for `text` under the given tiktoken encoding — used for
    elements (e.g. tables) that are never windowed but still need a
    ChunkRecord.token_count value.
    """
    if not text:
        return 0
    return len(tiktoken.get_encoding(encoding_name).encode(text))


def content_hash_for(text: str) -> str:
    """Deterministic content hash used for ChunkRecord.content_hash
    (change detection / dedup)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def concatenate_with_spans(elements: list[ParsedElement]) -> tuple[str, list[tuple[ParsedElement, int, int]]]:
    """Join `elements`' text with blank-line separators, returning the
    concatenated text plus each element's (start_char, end_char) span
    within it — used to map a TokenWindow back to the elements it draws
    from.
    """
    parts: list[str] = []
    spans: list[tuple[ParsedElement, int, int]] = []
    cursor = 0
    for element in elements:
        if not element.text:
            continue
        start = cursor
        parts.append(element.text)
        cursor += len(element.text)
        spans.append((element, start, cursor))
        parts.append("\n\n")
        cursor += 2
    return "".join(parts), spans


def elements_overlapping(
    spans: list[tuple[ParsedElement, int, int]], start_char: int, end_char: int
) -> list[ParsedElement]:
    """Every element whose span intersects [start_char, end_char)."""
    return [element for element, span_start, span_end in spans if span_start < end_char and span_end > start_char]


def aggregate_source_refs(
    elements: list[ParsedElement],
) -> tuple[list[SourceReference], int | None, int | None, list[str], str | None]:
    """Build a chunk's `source_refs` plus the denormalized
    page_start/page_end/section_title fields from the ParsedElements it
    draws from. `section_path` is returned too, for callers that don't
    already have it from a groupby key.
    """
    source_refs = [element.source for element in elements]
    all_pages = [page for ref in source_refs for page in ref.page_numbers]
    page_start = min(all_pages) if all_pages else None
    page_end = max(all_pages) if all_pages else None
    section_path = source_refs[0].section_path if source_refs else []
    section_title = source_refs[0].section_title if source_refs else None
    return source_refs, page_start, page_end, section_path, section_title
