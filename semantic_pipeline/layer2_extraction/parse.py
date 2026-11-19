"""The stored dependency parse: one CoNLL-U file per video.

run_parse.py parses every text span once (StanzaEngine: words, multi-word
tokens, part-of-speech tags, lemmas and dependency relations) and writes the
result in CoNLL-U, the plain-text format of Universal Dependencies. Steps that
need words, tags or the dependency tree read this file instead of running the
tagger themselves, so the text is parsed once and the parse can be inspected.

One block per text span, in text-stream order:

    # newdoc id = <span id>
    # language = <the language the text stream records>
    # text_sha1 = <SHA-1 of the span text>
    # parser = <model id, e.g. stanza1.15.1:default>
    # sent_id = <span id>:1
    # text = <the sentence, line breaks written as spaces>
    1   Die   der   DET   ART   Case=Nom|...   2   det   _   start_char=0|end_char=3
    ...

The ten tab-separated columns are the standard ones (ID, FORM, LEMMA, UPOS,
XPOS, FEATS, HEAD, DEPREL, DEPS, MISC); "_" marks an empty field. Character
offsets into the span text are in MISC, as Stanza writes them. A multi-word
token has a range line ("3-4") with the token's text and offsets. Each word line
holds the word's own offsets: part of the token where the word is written out
(English "that's" = "that" + "'s"), the whole token where it is not (German
"am" = "an dem"). HEAD and DEPREL are "_" for a language without a dependency
parser. Spans without words get no block.

A span is found again by its language and text (through the SHA-1), so a reader
gets exactly the parse of the text it holds. Whether a file still matches its
text stream is checked by the run scripts, from the digest of the whole text
stream recorded in the parse step's run manifest.
"""

from __future__ import annotations

import hashlib
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from .interfaces import Token, TokenTaggerEngine

_EMPTY = "_"


@dataclass(frozen=True)
class ParsedWord:
    """One syntactic word. `head` is the id of the word it depends on (0 for
    the root); `head` and `deprel` are None without a dependency parser.
    `start`/`end` are character offsets into the span text; a word inside a
    multi-word token has the offsets of the whole token."""

    id: int
    text: str
    lemma: str | None
    upos: str
    xpos: str | None
    feats: str | None
    head: int | None
    deprel: str | None
    start: int
    end: int


@dataclass(frozen=True)
class MultiWordToken:
    """A written token that stands for several words (`first`..`last`)."""

    first: int
    last: int
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class ParsedSentence:
    text: str
    words: tuple[ParsedWord, ...]
    multiword: tuple[MultiWordToken, ...] = ()


@dataclass(frozen=True)
class ParsedSpan:
    """The parse of one text span."""

    span_id: str
    language: str
    text_sha1: str
    parser: str
    sentences: tuple[ParsedSentence, ...]


def text_sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def file_sha1(path: str | Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def tokens_of(sentences: Iterable[ParsedSentence]) -> list[Token]:
    """The words as the concept extractor reads them: offsets, text, part of
    speech and lemma (a word without a lemma takes its text)."""
    return [Token(w.start, w.end, w.text, w.upos, w.lemma or w.text)
            for s in sentences for w in s.words]


# --- writing ----------------------------------------------------------------------

def _field(value: str | None) -> str:
    if value is None or value == "":
        return _EMPTY
    return value.replace("\t", " ").replace("\r", " ").replace("\n", " ")


def _one_line(text: str) -> str:
    return text.replace("\r", " ").replace("\n", " ")


def _offsets(start: int, end: int) -> str:
    return f"start_char={start}|end_char={end}"


def format_span(span: ParsedSpan) -> str:
    """One span as a CoNLL-U block (its sentences, each ending in a blank line)."""
    lines = [f"# newdoc id = {span.span_id}", f"# language = {span.language}",
             f"# text_sha1 = {span.text_sha1}", f"# parser = {span.parser}"]
    for k, sentence in enumerate(span.sentences, 1):
        lines.append(f"# sent_id = {span.span_id}:{k}")
        lines.append(f"# text = {_one_line(sentence.text)}")
        ranges = {m.first: m for m in sentence.multiword}
        for w in sentence.words:
            m = ranges.get(w.id)
            if m is not None:
                lines.append("\t".join((f"{m.first}-{m.last}", _field(m.text),
                                        *([_EMPTY] * 7), _offsets(m.start, m.end))))
            lines.append("\t".join((
                str(w.id), _field(w.text), _field(w.lemma), _field(w.upos), _field(w.xpos),
                _field(w.feats), _EMPTY if w.head is None else str(w.head), _field(w.deprel),
                _EMPTY, _offsets(w.start, w.end))))
        lines.append("")
    return "\n".join(lines) + "\n"


def write_conllu(path: str | Path, spans: Iterable[ParsedSpan]) -> int:
    """Write the spans that have words; returns how many were written. Lines end
    in LF on every platform, as CoNLL-U requires."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for span in spans:
            if span.sentences:
                f.write(format_span(span))
                n += 1
    return n


# --- reading ----------------------------------------------------------------------

def _value(field: str) -> str | None:
    return None if field == _EMPTY else field


def _misc_offsets(misc: str) -> tuple[int | None, int | None]:
    items = dict(p.split("=", 1) for p in misc.split("|") if "=" in p)
    start, end = items.get("start_char"), items.get("end_char")
    return (int(start) if start is not None else None, int(end) if end is not None else None)


def _sentence(text: str, rows: list[list[str]], multiword: list[MultiWordToken]) -> ParsedSentence:
    words = []
    for row in rows:
        ident, form, lemma, upos, xpos, feats, head, deprel, _deps, misc = row
        wid = int(ident)
        start, end = _misc_offsets(misc)
        if start is None or end is None:
            token = next((m for m in multiword if m.first <= wid <= m.last), None)
            if token is None:
                raise ValueError(f"word {wid} ({form!r}) has no character offsets")
            start, end = token.start, token.end
        words.append(ParsedWord(
            id=wid, text=form, lemma=lemma if (lemma != _EMPTY or form == _EMPTY) else None,
            upos="" if upos == _EMPTY else upos, xpos=_value(xpos), feats=_value(feats),
            head=None if head == _EMPTY else int(head), deprel=_value(deprel),
            start=start, end=end))
    return ParsedSentence(text=text, words=tuple(words), multiword=tuple(multiword))


def parse_conllu(lines: Iterable[str]) -> Iterator[ParsedSpan]:
    """The spans of a file written by write_conllu, in file order."""
    meta: dict[str, str] | None = None
    sentences: list[ParsedSentence] = []
    text, rows, multiword = "", [], []

    def span() -> ParsedSpan:
        return ParsedSpan(span_id=meta["span_id"], language=meta.get("language", ""),
                          text_sha1=meta.get("text_sha1", ""), parser=meta.get("parser", ""),
                          sentences=tuple(sentences))

    for raw in lines:
        line = raw.rstrip("\r\n")
        if line.startswith("#"):
            key, _, value = line[1:].partition("=")
            key, value = key.strip(), value.strip()
            if key == "newdoc id":
                if meta is not None:
                    yield span()
                meta, sentences = {"span_id": value}, []
            elif key in ("language", "text_sha1", "parser") and meta is not None:
                meta[key] = value
            elif key == "text":
                text = value
        elif line.strip():
            cols = line.split("\t")
            if len(cols) != 10:
                raise ValueError(f"not a CoNLL-U word line: {line!r}")
            if "-" in cols[0]:
                first, last = (int(x) for x in cols[0].split("-"))
                start, end = _misc_offsets(cols[9])
                multiword.append(MultiWordToken(first, last, cols[1],
                                                start if start is not None else 0,
                                                end if end is not None else 0))
            elif "." not in cols[0]:          # empty nodes (enhanced UD) are not used
                rows.append(cols)
        elif rows:
            sentences.append(_sentence(text, rows, multiword))
            text, rows, multiword = "", [], []
    if rows:
        sentences.append(_sentence(text, rows, multiword))
    if meta is not None:
        yield span()


def read_conllu(path: str | Path) -> Iterator[ParsedSpan]:
    with Path(path).open("r", encoding="utf-8") as f:
        yield from parse_conllu(f)


# --- parsers ------------------------------------------------------------------------

class SentenceParser(TokenTaggerEngine):
    """A tagger that also gives the dependency parse of a text. Its tokens are
    the words of the parsed sentences."""

    @abstractmethod
    def sentences(self, text: str, language: str | None) -> tuple[ParsedSentence, ...] | None:
        """The parsed sentences of `text`; None if `language` has no model."""

    def tag_tokens(self, text: str, language: str | None) -> list[Token] | None:
        sentences = self.sentences(text, language)
        return None if sentences is None else tokens_of(sentences)


class StoredParse(SentenceParser):
    """Parsed spans served like a tagger. A span is looked up by its language and
    text, so the concept extractor and the category table's GLiNER filler read
    the stored words instead of running the tagger. A language counts as
    supported if the stored parse holds spans in it. `misses` counts texts in a
    supported language that the parse does not hold."""

    def __init__(self, spans: Iterable[ParsedSpan]):
        self._spans: dict[tuple[str, str], ParsedSpan] = {}
        parsers: set[str] = set()
        for s in spans:
            self._spans.setdefault((s.language, s.text_sha1), s)
            parsers.add(s.parser)
        self._languages = frozenset(language for language, _ in self._spans)
        self._model_id = "+".join(sorted(parsers)) or "none"
        self.misses = 0

    @classmethod
    def from_files(cls, paths: Iterable[str | Path]) -> "StoredParse":
        return cls(s for p in paths for s in read_conllu(p))

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def languages(self) -> frozenset[str]:
        return self._languages

    def __len__(self) -> int:
        return len(self._spans)

    def supports(self, language: str | None) -> bool:
        return language in self._languages

    def sentences(self, text: str, language: str | None) -> tuple[ParsedSentence, ...] | None:
        """The parsed sentences of `text`; None if the language is not in the
        stored parse, empty for a text without words."""
        if not self.supports(language):
            return None
        if not text.strip():
            return ()
        span = self._spans.get((language, text_sha1(text)))
        if span is None:
            self.misses += 1
            return ()
        return span.sentences
