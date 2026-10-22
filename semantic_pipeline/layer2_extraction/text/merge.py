"""Word-by-word merge of subtitle readings into the speech transcript.

Every stretch of speech that a subtitle line renders becomes a slot holding the
recognizer's reading and one reading per OCR read of that subtitle (OCR often
reads the same subtitle several times). Ordered rules pick one reading:

    agree          all readings are the same words
    spacing        same letters, different spacing: the reading made of real
                   words wins, the recognizer's on a tie (OCR drops spaces far
                   more often than the recognizer joins words, as in
                   "zuersticken"; a compound such as "Hitzewelle" stays whole)
    lexicon        readings that are not real words drop out, if at least one
                   reading is a real word (OCR misreadings produce non-words)
    majority       the reading most reads agree on
    tie->subtitle  still tied: the subtitle wins, because the recognizer's
                   errors are real words, often heard with high confidence
                   ("Bereiche" for "Reiche"); configurable to tie->speech

Subtitle words the recognizer missed are inserted where the alignment pins them
on both sides, if they are real words (OCR noise is not a missed word). A short
run of differing words pinned on both sides, by aligned words of the same line
or of the neighbouring subtitle line, is a substitution slot like any other:
"elf" for "11", or a misheard name at the end of a line ("Kolja" for "Coria"),
which string similarity cannot align. Words that cannot be placed safely stay
out of the text and are recorded as "unplaced". Every slot where
the readings differ is kept as a WordDecision with all readings, so the merge
can be audited and redone.

Where the recognizer's reading wins, quotation marks that the subtitles show at
its edges are carried over: they mark reported speech, which the recognizer
cannot hear and which quote attribution (who says what about whom) needs.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from ...config import TextConfig
from .lexicon import Lexicon
from .models import WordDecision
from .tokens import QUOTES, OcrLine, SpeechToken, Token
from .tracks import Block

_HEAD = re.compile(r"^\W*")
_TAIL = re.compile(r"\W*$")


@dataclass(frozen=True)
class _Reading:
    tokens: tuple[Token, ...]
    ref: str | None = None      # OCR observation id; None for the recognizer

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(t.key for t in self.tokens)

    @property
    def joined(self) -> str:
        return "".join(self.keys)

    @property
    def text(self) -> str:
        return " ".join(t.surface.strip() for t in self.tokens)


@dataclass
class _Slot:
    span: tuple[int, int]
    readings: list[_Reading]    # [recognizer, subtitle reads ...]


@dataclass
class _Insert:
    anchor: int                 # speech token index
    before: bool                # before the anchor token (else after it)
    reading: _Reading
    texts: list[str]
    refs: list[str]


@dataclass
class SegmentText:
    text: str = ""
    decisions: list[WordDecision] = field(default_factory=list)
    refs: list[str] = field(default_factory=list)   # OCR units consulted
    checked: bool = False                           # any subtitle reading consulted


@dataclass
class MergeResult:
    segments: dict[str, SegmentText]
    n_agree: int


class _SlotTable:
    def __init__(self, tokens: list[SpeechToken]):
        self._tokens = tokens
        self.slots: dict[tuple[int, int], _Slot] = {}
        self.owner: dict[int, tuple[int, int]] = {}

    def add(self, span: tuple[int, int], reading: _Reading) -> None:
        slot = self.slots.get(span)
        if slot is None:
            if any(j in self.owner for j in range(*span)):
                return  # another reading segmented this stretch differently; keep the first
            recognizer = _Reading(tuple(t.token for t in self._tokens[span[0]:span[1]]))
            slot = self.slots[span] = _Slot(span, [recognizer])
            for j in range(*span):
                self.owner[j] = span
        slot.readings.append(reading)


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    for i in indices:
        if runs and runs[-1][1] == i:
            runs[-1] = (runs[-1][0], i + 1)
        else:
            runs.append((i, i + 1))
    return runs


def decide(readings: list[_Reading], lexicon: Lexicon, tie_break: str) -> tuple[_Reading, str]:
    """Pick one reading of a slot; readings[0] is the recognizer's."""
    speech = readings[0]
    real = [r for r in readings if all(lexicon.is_word(k) for k in r.keys)]
    if len({r.joined for r in readings}) == 1:
        if len({r.keys for r in readings}) == 1:
            return speech, "agree"
        pool = real or readings
        chosen = next((r for r in pool if r is speech), None) or max(pool, key=lambda r: len(r.tokens))
        return chosen, "spacing"
    pool, filtered = readings, False
    if real and len(real) < len(readings):
        pool, filtered = real, True
    votes = Counter(r.joined for r in pool)
    top = max(votes.values())
    winners = {k for k, v in votes.items() if v == top}
    if len(winners) == 1:
        cands = [r for r in pool if r.joined in winners]
        chosen = next((r for r in cands if r is speech), cands[0])
        return chosen, "lexicon" if filtered else "majority"
    if tie_break == "subtitle":
        return next(r for r in pool if r is not speech and r.joined in winners), "tie->subtitle"
    chosen = next((r for r in pool if r is speech and r.joined in winners), None)
    return chosen or next(r for r in pool if r.joined in winners), "tie->speech"


def _edge_quotes(subtitles: list[_Reading]) -> tuple[str, str]:
    """Quotation marks the subtitles show at the edges of a slot (they mark
    reported speech, which the recognizer cannot hear)."""
    lead = trail = ""
    for r in subtitles:
        lead = lead or "".join(c for c in _HEAD.match(r.tokens[0].surface).group() if c in QUOTES)
        trail = trail or "".join(c for c in _TAIL.search(r.tokens[-1].surface).group() if c in QUOTES)
    return lead, trail


def _quoted(surface: str, lead: str, trail: str) -> str:
    space = surface[: len(surface) - len(surface.lstrip())]
    core = surface.strip()
    if lead and not core.startswith(lead):
        core = lead + core
    if trail and not core.endswith(trail):
        core = core + trail
    return space + core


def merge(tokens: list[SpeechToken], blocks: list[Block], lexicon: Lexicon,
          cfg: TextConfig) -> MergeResult:
    """Merge the subtitle blocks into the recognizer's tokens, per segment."""
    blocks = sorted(blocks, key=lambda b: (b.line.t_start, b.line.key))
    table = _SlotTable(tokens)

    # 1) Slots from aligned words.
    for b in blocks:
        for p in b.alignment.pairs:
            table.add(p.speech, _Reading(b.line.tokens[p.ocr[0]:p.ocr[1]], b.line.observation_id))

    # 2) Unmatched subtitle words: substitution, insertion, or unplaced.
    owned = set(table.owner)
    substitutions: list[tuple[tuple[int, int], _Reading]] = []
    inserts: dict[tuple[int, bool, str], _Insert] = {}
    unplaced: list[tuple[OcrLine, _Reading, int]] = []

    def insert(ln: OcrLine, anchor: int, before: bool, reading: _Reading) -> None:
        if not all(lexicon.is_word(k) for k in reading.keys):
            unplaced.append((ln, reading, anchor))   # OCR noise is not a missed word
            return
        key = (anchor, before, reading.joined)
        if key in inserts:
            inserts[key].refs.append(reading.ref)
            inserts[key].texts.append(reading.text)
        else:
            inserts[key] = _Insert(anchor, before, reading, [reading.text], [reading.ref])

    def comparable(reading: _Reading, gap: range) -> bool:
        """A substitution must be of similar length: a two-letter fragment
        cannot stand in for "werden, mit"."""
        a, g = len(reading.joined), sum(len(tokens[j].token.key) for j in gap)
        return min(a, g) >= cfg.substitution_length_ratio_min * max(a, g)

    for b in blocks:
        ln, al = b.line, b.alignment
        if not al.pairs:
            continue
        matched = {i for p in al.pairs for i in range(*p.ocr)}
        for r0, r1 in _runs([i for i in range(len(ln.tokens)) if i not in matched]):
            reading = _Reading(ln.tokens[r0:r1], ln.observation_id)
            prev = next((p for p in reversed(al.pairs) if p.ocr[1] <= r0), None)
            nxt = next((p for p in al.pairs if p.ocr[0] >= r1), None)
            if prev and nxt:
                gap = range(prev.speech[1], nxt.speech[0])
                if not gap:
                    insert(ln, nxt.speech[0], True, reading)
                elif (not owned.intersection(gap) and len(gap) <= cfg.max_gap_tokens
                      and r1 - r0 <= cfg.max_gap_tokens and comparable(reading, gap)):
                    substitutions.append(((gap.start, gap.stop), reading))
                else:
                    unplaced.append((ln, reading, prev.speech[1] - 1))
            elif nxt:   # leading run, pinned on the left by the previous subtitle line
                j = nxt.speech[0]
                lo = j
                while lo > al.window[0] and (lo - 1) not in owned and j - lo < cfg.max_gap_tokens:
                    lo -= 1
                if lo == j:
                    insert(ln, j, True, reading)       # the word before is accounted for
                elif lo - 1 in owned and r1 - r0 <= cfg.max_gap_tokens and comparable(reading, range(lo, j)):
                    substitutions.append(((lo, j), reading))
                else:
                    unplaced.append((ln, reading, j))
            elif prev:  # trailing run, pinned on the right by the next subtitle line
                j = prev.speech[1]
                hi = j
                while hi < al.window[1] and hi not in owned and hi - j < cfg.max_gap_tokens:
                    hi += 1
                if hi == j:
                    insert(ln, j - 1, False, reading)  # the word after is accounted for
                elif hi in owned and r1 - r0 <= cfg.max_gap_tokens and comparable(reading, range(j, hi)):
                    substitutions.append(((j, hi), reading))
                else:
                    unplaced.append((ln, reading, j - 1))
    for span, reading in substitutions:
        table.add(span, reading)

    # 3) Decide every slot and rebuild the text segment by segment.
    decided = {span: decide(slot.readings, lexicon, cfg.tie_break) for span, slot in table.slots.items()}
    start_of = {span[0]: span for span in table.slots}
    before: dict[int, list[_Insert]] = {}
    after: dict[int, list[_Insert]] = {}
    for ins in inserts.values():
        (before if ins.before else after).setdefault(ins.anchor, []).append(ins)

    segments: dict[str, SegmentText] = {}
    pieces: dict[str, list[str]] = {}
    n_agree = 0

    def emit_insert(ins: _Insert, seg: SegmentText, out: list[str]) -> None:
        out.append(" " + ins.reading.text)
        seg.checked = True
        seg.refs.extend(ins.refs)
        tok = tokens[ins.anchor]
        seg.decisions.append(WordDecision(
            t_start=tok.t_start, t_end=tok.t_end, action="insert", chosen=ins.reading.text,
            speech="", subtitle=tuple(ins.texts), rule="missing in speech",
            observation_refs=tuple(dict.fromkeys(ins.refs)),
        ))

    j = 0
    while j < len(tokens):
        seg_id = tokens[j].segment_id
        seg = segments.setdefault(seg_id, SegmentText())
        out = pieces.setdefault(seg_id, [])
        for ins in before.get(j, []):
            emit_insert(ins, seg, out)
        if j in start_of:
            span = start_of[j]
            slot = table.slots[span]
            chosen, rule = decided[span]
            speech, subs = slot.readings[0], slot.readings[1:]
            seg.checked = True
            seg.refs.extend(r.ref for r in subs)
            if chosen is speech:
                # The recognizer's words stay in their own segments (a slot can
                # span a segment boundary), with the subtitles' quotation marks.
                lead, trail = _edge_quotes(subs)
                parts = []
                for k in range(*span):
                    surface = _quoted(tokens[k].token.surface, lead if k == span[0] else "",
                                      trail if k == span[1] - 1 else "")
                    segments.setdefault(tokens[k].segment_id, SegmentText())
                    pieces.setdefault(tokens[k].segment_id, []).append(surface)
                    parts.append(surface)
                text = "".join(parts)
            else:
                text = " " + chosen.text
                out.append(text)
            if any(r.keys != speech.keys for r in subs):
                seg.decisions.append(WordDecision(
                    t_start=tokens[span[0]].t_start, t_end=tokens[span[1] - 1].t_end,
                    action="keep" if chosen is speech else "replace", chosen=text.strip(),
                    speech=speech.text, subtitle=tuple(r.text for r in subs), rule=rule,
                    observation_refs=tuple(dict.fromkeys(r.ref for r in subs)),
                ))
            else:
                n_agree += 1
            # inserts anchored inside a multi-token slot go right after it
            for k in range(span[0] + 1, span[1]):
                for ins in before.get(k, []):
                    emit_insert(ins, seg, out)
            for k in range(span[0], span[1] - 1):
                for ins in after.get(k, []):
                    emit_insert(ins, seg, out)
            last, j = span[1] - 1, span[1]
        else:
            out.append(tokens[j].token.surface)
            last, j = j, j + 1
        for ins in after.get(last, []):
            emit_insert(ins, seg, out)

    for ln, reading, anchor in unplaced:
        seg = segments.setdefault(tokens[anchor].segment_id, SegmentText())
        seg.checked = True
        seg.decisions.append(WordDecision(
            t_start=ln.t_start, t_end=ln.t_end, action="unplaced", chosen="", speech="",
            subtitle=(reading.text,), rule="unplaced", observation_refs=(ln.observation_id,),
        ))

    for seg_id, seg in segments.items():
        seg.text = " ".join("".join(pieces.get(seg_id, [])).split())
        seg.decisions.sort(key=lambda d: (d.t_start, d.t_end, d.action))
        seg.refs = list(dict.fromkeys(seg.refs))
    return MergeResult(segments, n_agree)
