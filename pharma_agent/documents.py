"""Deterministic section-aware chunking with exact offsets into original UTF-8 files."""

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

CHUNKER_VERSION = "sections-v1-2400-240"


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    title: str
    section: str
    text: str
    line_start: int
    line_end: int
    char_start: int
    char_end: int
    source_url: str | None = None

    @property
    def embedding_text(self) -> str:
        return f"{self.title}\n{self.source}\n{self.section}\n{self.text}"

    def as_evidence(self) -> dict:
        return asdict(self)


def chunk_document(source: str, text: str, max_chars: int = 2400, overlap: int = 240):
    if max_chars < 100 or not 0 <= overlap < max_chars // 2:
        raise ValueError("Invalid chunk size/overlap")
    title_match = re.search(r"^#\s+(.+)$", text, re.M)
    title = title_match.group(1) if title_match else source
    url_match = re.search(r"^Source:\s*(https?://\S+)", text, re.M)
    url = url_match.group(1) if url_match else None
    headings = list(re.finditer(r"^(#{1,6})\s+(.+)$", text, re.M))
    boundaries = [(0, "Introduction")]
    stack: list[tuple[int, str]] = []
    for heading in headings:
        level, name = len(heading.group(1)), heading.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, name))
        section = " > ".join(name for depth, name in stack if depth > 1) or "Introduction"
        boundaries.append((heading.start(), section))
    chunks = []
    for idx, (begin, section) in enumerate(boundaries):
        finish = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(text)
        cursor = begin
        while cursor < finish:
            while cursor < finish and text[cursor].isspace():
                cursor += 1
            if cursor >= finish:
                break
            end = min(cursor + max_chars, finish)
            if end < finish:
                split = text.rfind("\n", cursor + max_chars // 2, end)
                if split < 0:
                    split = text.rfind(" ", cursor + max_chars // 2, end)
                if split > cursor:
                    end = split
            at_section_end = end == finish
            while end > cursor and text[end - 1].isspace():
                end -= 1
            content = text[cursor:end]
            # Do not index standalone headings/source headers as clinical evidence.
            body = re.sub(r"^#{1,6}\s+.*$|^Source:.*$", "", content, flags=re.M).strip()
            if body:
                identity = f"{CHUNKER_VERSION}|{source}|{section}|{cursor}|{content}"
                chunks.append(
                    Chunk(
                        id=hashlib.sha256(identity.encode()).hexdigest()[:20],
                        source=source,
                        title=title,
                        section=section,
                        text=content,
                        line_start=text.count("\n", 0, cursor) + 1,
                        line_end=text.count("\n", 0, end) + 1,
                        char_start=cursor,
                        char_end=end,
                        source_url=url,
                    )
                )
            if at_section_end:
                break
            cursor = max(cursor + 1, end - overlap)
            # Start overlap at a word boundary rather than midway through a medical term.
            while cursor < end and not text[cursor - 1].isspace():
                cursor += 1
    return chunks


def load_corpus(directory: Path) -> list[Chunk]:
    directory = directory.resolve()
    if not directory.is_dir():
        raise ValueError("Corpus directory is missing")
    chunks = []
    # pathlib's default ordering is case-insensitive on Windows and case-sensitive on Linux.
    # Use the same explicit order so corpus/cache versions and tie-breaking travel across OSes.
    for path in sorted(
        directory.rglob("*"), key=lambda item: item.relative_to(directory).as_posix().casefold()
    ):
        if path.suffix.lower() not in {".md", ".txt"} or not path.is_file():
            continue
        if not path.resolve().is_relative_to(directory):
            raise ValueError("Corpus contains a file outside its root")
        text = path.read_text(encoding="utf-8-sig")
        chunks.extend(chunk_document(path.relative_to(directory).as_posix(), text))
    if not chunks:
        raise ValueError("Corpus contains no nonempty Markdown or text documents")
    if len({chunk.id for chunk in chunks}) != len(chunks):
        raise ValueError("Duplicate chunk identifiers")
    return chunks


def corpus_version(chunks: list[Chunk]) -> str:
    return hashlib.sha256("\n".join(c.embedding_text + c.id for c in chunks).encode()).hexdigest()
