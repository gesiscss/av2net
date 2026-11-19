"""Tests for the Layer 2 text assembly (speech + OCR -> one text stream).

Synthetic videos: speech with word timestamps, and OCR units whose lines sit in
a subtitle band (centred, y ~0.71) or elsewhere (headlines, banners). A small
set-based lexicon keeps the tests independent of word-list data.
"""

from __future__ import annotations

import pytest

from semantic_pipeline.config import TextConfig
from semantic_pipeline.layer2_extraction.context import TextSpan
from semantic_pipeline.layer2_extraction.extractors.entities import LLMEntityExtractor
from semantic_pipeline.layer2_extraction.interfaces import LLMEngine, TruecaseEngine
from semantic_pipeline.layer2_extraction.memory import EpisodicMemory
from semantic_pipeline.layer2_extraction.text import assemble, build_lexicon
from semantic_pipeline.layer2_extraction.text.lexicon import Lexicon
from semantic_pipeline.layer2_extraction.text.tokens import rejoin_line_breaks, split_tokens
from semantic_pipeline.layer2_extraction.text.tracks import _join_line_breaks
from semantic_pipeline.layer3_network.stream import _event_channels
from semantic_pipeline.config import NERConfig
from semantic_pipeline.models.event import SemanticEvent
from semantic_pipeline.models.observation import OCRLine, OCRObservation, SpeechObservation, Word

SRC = "vid"
FRAME = (720, 1280)
SUB = (180, 880, 540, 940)      # subtitle band: centred, y_center ~0.71
HEAD = (60, 500, 660, 540)      # a headline higher up
BANNER = (60, 420, 300, 450)


class SetLexicon(Lexicon):
    def __init__(self, words=()):
        self._words = set(words)

    @property
    def name(self) -> str:
        return "test"

    def is_word(self, key: str) -> bool:
        return key in self._words or any(c.isdigit() for c in key)


def seg(idx: int, words: list[tuple[str, float, float]]) -> SpeechObservation:
    """A speech segment; word texts are given exactly as the recognizer writes them."""
    ws = tuple(Word(text=w, t_start=a, t_end=b, prob=0.9) for w, a, b in words)
    return SpeechObservation(
        observation_id=f"{SRC}:speech:{idx:06d}", source_id=SRC, model="test",
        t_start=ws[0].t_start, t_end=ws[-1].t_end,
        text="".join(w.text for w in ws).strip(), words=ws,
    )


def timed(text: str, t0: float, step: float = 0.4) -> list[tuple[str, float, float]]:
    """Whisper-style words (leading space) of `text`, `step` seconds each from t0."""
    return [(" " + w, t0 + i * step, t0 + (i + 1) * step) for i, w in enumerate(text.split())]


def unit(idx: int, t0: float, t1: float, *lines: tuple[str, tuple]) -> OCRObservation:
    return OCRObservation(
        observation_id=f"{SRC}:ocr:{idx:06d}", source_id=SRC, model="test",
        t_start=t0, t_end=t1, text="\n".join(t for t, _ in lines), region="lower",
        confidence=0.99, bbox=lines[0][1],
        lines=tuple(OCRLine(text=t, confidence=0.99, bbox=b) for t, b in lines),
    )


class FakeTruecaser(TruecaseEngine):
    """Answers from a table keyed by the lower-case text; other texts come back in lower case."""

    def __init__(self, answers=None):
        self.answers = dict(answers or {})
        self.calls: list[list[str]] = []

    @property
    def model_id(self) -> str:
        return "fake"

    def truecase(self, texts):
        self.calls.append(list(texts))
        return [self.answers.get(t.lower(), t.lower()) for t in texts]


def run(speech, ocr, words=(), truecaser=None):
    return assemble(speech, ocr, cfg=TextConfig(), frame_size=FRAME, lexicon=SetLexicon(words),
                    truecaser=truecaser or FakeTruecaser())


def speech_texts(stream) -> list[str]:
    return [s.text for s in stream.spans if s.channel == "speech"]


def decisions(stream):
    return [d for s in stream.spans for d in s.decisions]


def test_subtitles_merge_and_headlines_stay_their_own_channel():
    speech = [seg(0, timed("Die Hitze eskaliert in ganz Europa", 0.0)),
              seg(1, timed("Die Bahn warnt vor Störungen", 3.0))]
    ocr = [unit(0, 0.0, 1.2, ("Die Hitze eskaliert", SUB)),
           unit(1, 1.2, 2.4, ("in ganz Europa", SUB)),
           unit(2, 3.0, 4.2, ("Die Bahn warnt", SUB)),
           unit(3, 4.2, 5.0, ("vor Störungen", SUB)),
           unit(4, 0.0, 5.0, ("Rekordhitze in Spanien", HEAD))]
    stream = run(speech, ocr)

    assert stream.diagnostics["subtitle_track_found"]
    assert speech_texts(stream) == ["Die Hitze eskaliert in ganz Europa", "Die Bahn warnt vor Störungen"]
    onscreen = [s for s in stream.spans if s.channel == "ocr"]
    assert [s.text for s in onscreen] == ["Rekordhitze in Spanien"]   # subtitles are not read twice
    assert all(s.origin == "speech+subtitle" for s in stream.spans if s.channel == "speech")


def test_multi_line_subtitle_blocks_form_one_track():
    # Bottom-anchored blocks: the last line sits at y ~0.94, the block grows upwards.
    row = {k: (180, y - 25, 540, y + 25) for k, y in {1: 1202, 2: 1153, 3: 1104}.items()}
    speech = [seg(0, timed("Hi ich bin Coria von Wollt", 0.0)),
              seg(1, timed("und ich erkläre euch in dieser Reihe unsere Drogenpolitik", 3.0)),
              seg(2, timed("Dieses Mal das portugiesische Modell", 7.0)),
              seg(3, timed("Trotzdem sehen wir es kritisch", 9.5))]
    ocr = [unit(0, 0.0, 2.4, ("Hi, ich bin Kolja von Volt", row[1])),
           unit(1, 3.0, 6.6, ("und ich erkläre euch", row[3]), ("in dieser Reihe", row[2]),
                ("unsere Drogenpolitik.", row[1])),
           unit(2, 7.0, 9.0, ("Dieses Mal: das", row[2]), ("portugiesische Modell.", row[1])),
           unit(3, 9.5, 11.5, ("Trotzdem sehen wir es kritisch.", row[1])),
           unit(4, 0.0, 11.5, ("Rekordhitze in Spanien", HEAD))]
    stream = run(speech, ocr, words={"kolja", "coria", "volt", "wollt"})

    [track] = stream.diagnostics["subtitle_tracks"]
    assert track["n_rows"] == 3 and stream.diagnostics["speech_covered_by_subtitles"] > 0.9
    # "Kolja" sits between aligned words; "Volt" ends its line and is pinned by the next one.
    assert speech_texts(stream)[0] == "Hi ich bin Kolja von Volt"
    assert [s.text for s in stream.spans if s.channel == "ocr"] == ["Rekordhitze in Spanien"]


def test_misheard_real_word_loses_tie_to_subtitle():
    speech = [seg(0, timed("Bereiche heute einmal einen Blick in die Realität waren.", 0.0))]
    ocr = [unit(0, 0.0, 0.8, ("„Reiche,", SUB)),
           unit(1, 0.8, 2.0, ("heute einmal einen Blick", SUB)),
           unit(2, 2.0, 3.2, ("in die Realität wagen.", SUB))]
    stream = run(speech, ocr, words={"reiche", "bereiche", "wagen", "waren"})

    assert speech_texts(stream) == ["„Reiche, heute einmal einen Blick in die Realität wagen."]
    assert [(d.speech, d.chosen, d.rule) for d in decisions(stream)] == [
        ("Bereiche", "„Reiche,", "tie->subtitle"), ("waren.", "wagen.", "tie->subtitle")]
    span = stream.spans[0]
    assert span.observation_refs[0] == span.span_id and len(span.observation_refs) == 4


def test_tie_break_can_favour_speech():
    speech = [seg(0, timed("Bereiche heute einmal einen Blick", 0.0))]
    ocr = [unit(0, 0.0, 0.4, ("Reiche", SUB)), unit(1, 0.4, 1.2, ("heute einmal", SUB)),
           unit(2, 1.2, 2.0, ("einen Blick", SUB))]
    cfg = TextConfig(tie_break="speech")
    stream = assemble(speech, ocr, cfg=cfg, frame_size=FRAME, lexicon=SetLexicon({"reiche", "bereiche"}),
                      truecaser=FakeTruecaser())
    assert speech_texts(stream) == ["Bereiche heute einmal einen Blick"]


def test_lexicon_rejects_ocr_misreading():
    speech = [seg(0, timed("Auf der A2 drohen 9000 Hühner zu ersticken", 0.0))]
    ocr = [unit(0, 0.0, 1.2, ("Auf der A2", SUB)),
           unit(1, 1.2, 2.4, ("drohen 9000 Hüthner", SUB)),
           unit(2, 2.4, 3.2, ("zu ersticken", SUB))]
    stream = run(speech, ocr, words={"hühner"})

    assert speech_texts(stream) == ["Auf der A2 drohen 9000 Hühner zu ersticken"]
    [d] = decisions(stream)
    assert (d.action, d.chosen, d.rule) == ("keep", "Hühner", "lexicon")


def test_ocr_run_on_keeps_the_recognizers_spacing():
    speech = [seg(0, timed("Erneuerbare sind mittlerweile günstig", 0.0))]
    ocr = [unit(0, 0.0, 0.4, ("Erneuerbare", SUB)),
           unit(1, 0.4, 1.2, ("sindmittlerweile", SUB)),
           unit(2, 1.2, 1.6, ("günstig", SUB))]
    stream = run(speech, ocr)
    assert speech_texts(stream) == ["Erneuerbare sind mittlerweile günstig"]
    assert [(d.action, d.rule) for d in decisions(stream)] == [("keep", "spacing")]


def test_kept_words_stay_in_their_segment_across_a_run_on():
    # OCR "8.Juli" spans the recognizer's segment boundary " 8." | " Juli,".
    speech = [seg(0, timed("Die Sitzung ist am 8.", 0.0)), seg(1, timed("Juli, 2026 geschlossen", 2.0))]
    ocr = [unit(0, 0.0, 0.8, ("Die Sitzung", SUB)), unit(1, 0.8, 2.4, ("ist am 8.Juli", SUB)),
           unit(2, 2.4, 3.2, ("2026 geschlossen", SUB))]
    stream = run(speech, ocr, words={"juli"})
    assert speech_texts(stream) == ["Die Sitzung ist am 8.", "Juli, 2026 geschlossen"]


def test_recognizer_run_on_is_split_when_only_the_subtitle_is_real_words():
    speech = [seg(0, timed("droht 9000 Hühnern zuersticken", 0.0))]
    ocr = [unit(0, 0.0, 0.4, ("droht", SUB)), unit(1, 0.4, 1.2, ("9000 Hühnern", SUB)),
           unit(2, 1.2, 1.6, ("zu ersticken", SUB))]
    stream = run(speech, ocr, words={"zu", "ersticken", "hühnern", "droht"})
    assert speech_texts(stream) == ["droht 9000 Hühnern zu ersticken"]
    assert [d.rule for d in decisions(stream)] == ["spacing"]


def test_missing_words_are_inserted_where_the_alignment_pins_them():
    speech = [seg(0, timed("und vielleicht sollte Katharina den Schwerpunkt fast überbetont", 0.0))]
    ocr = [unit(0, 0.0, 1.2, ("und vielleicht sollte", SUB)),
           unit(1, 1.2, 1.6, ("Katharina...", SUB)),
           unit(2, 1.6, 2.4, ('"wir haben den Schwerpunkt', SUB)),
           unit(3, 2.4, 3.2, ('fast überbetont"', SUB))]
    stream = run(speech, ocr, words={"wir", "haben"})
    assert speech_texts(stream) == ['und vielleicht sollte Katharina "wir haben den Schwerpunkt fast überbetont"']
    assert [(d.action, d.chosen) for d in decisions(stream)] == [("insert", '"wir haben')]


def test_ocr_noise_inside_a_subtitle_is_not_inserted():
    speech = [seg(0, timed("Die Hitze eskaliert in ganz Europa", 0.0))]
    ocr = [unit(0, 0.0, 1.2, ("Die l Hitze eskaliert", SUB)), unit(1, 1.2, 1.6, ("in", SUB)),
           unit(2, 1.6, 2.4, ("ganz Europa", SUB))]
    stream = run(speech, ocr, words={"die", "hitze"})
    assert speech_texts(stream) == ["Die Hitze eskaliert in ganz Europa"]
    assert [(d.action, d.subtitle) for d in decisions(stream)] == [("unplaced", ("l",))]


def test_a_merge_cannot_swallow_a_word():
    speech = [seg(0, timed("auf mittlerweile 11 Autobahnen gibt es Probleme", 0.0))]
    ocr = [unit(0, 0.0, 1.6, ("auf mittlerweile elf Autobahnen", SUB)),
           unit(1, 1.6, 2.4, ("gibt es", SUB)),
           unit(2, 2.4, 2.8, ("Probleme", SUB))]
    stream = run(speech, ocr, words={"elf"})
    assert speech_texts(stream) == ["auf mittlerweile elf Autobahnen gibt es Probleme"]
    [d] = decisions(stream)
    assert (d.action, d.speech, d.chosen) == ("replace", "11", "elf")


def test_number_spacing_of_the_recognizer_is_kept():
    words = [(" Auf", 0.0, 0.3), (" der", 0.3, 0.6), (" A2", 0.6, 0.9), (" drohten", 0.9, 1.3),
             (" 9", 1.3, 1.5), (".000", 1.5, 1.7), (" Hühner", 1.7, 2.1)]
    speech = [seg(0, words)]
    ocr = [unit(0, 0.0, 0.6, ("Auf der", SUB)), unit(1, 0.6, 1.3, ("A2 drohten", SUB)),
           unit(2, 1.3, 2.1, ("9.000 Hühner", SUB))]
    stream = run(speech, ocr)
    assert speech_texts(stream) == ["Auf der A2 drohten 9.000 Hühner"]
    assert [d.action for d in decisions(stream)] == ["keep"]


def test_without_subtitles_the_transcript_is_unchanged():
    text = "Wir sind allerdings nicht so im Alarmismus unterwegs"
    speech = [seg(0, timed(text, 0.0))]
    ocr = [unit(0, 0.0, 3.0, ("Hitzewelle in Frankreich", HEAD)),
           unit(1, 0.0, 3.0, ("Ausland", BANNER))]
    stream = run(speech, ocr)
    assert not stream.diagnostics["subtitle_track_found"]
    assert speech_texts(stream) == [text]
    assert sorted(s.text for s in stream.spans if s.channel == "ocr") == ["Ausland", "Hitzewelle in Frankreich"]


def test_without_speech_all_text_is_onscreen():
    stream = run([], [unit(0, 0.0, 1.0, ("Hallo Welt", SUB))])
    assert [(s.channel, s.text) for s in stream.spans] == [("ocr", "Hallo Welt")]


def test_subtitle_band_text_that_renders_no_speech_stays_onscreen():
    # A title during a pause and an end card after the last speech.
    speech = [seg(0, timed("Die Hitze eskaliert heute", 0.0)),
              seg(1, timed("in ganz Europa", 6.0))]
    ocr = [unit(0, 0.0, 0.8, ("Die Hitze", SUB)), unit(1, 0.8, 1.6, ("eskaliert heute", SUB)),
           unit(2, 3.5, 4.5, ("Wir müssen handeln", SUB)), unit(3, 6.0, 7.2, ("in ganz Europa", SUB)),
           unit(4, 9.0, 12.0, ("Dein Direktkandidat", SUB))]
    stream = run(speech, ocr)
    assert stream.diagnostics["subtitle_track_found"]
    assert speech_texts(stream) == ["Die Hitze eskaliert heute", "in ganz Europa"]
    assert [(s.channel, s.origin, s.text) for s in stream.spans if s.channel == "ocr"] == [
        ("ocr", "onscreen", "Wir müssen handeln"), ("ocr", "onscreen", "Dein Direktkandidat")]


def test_quotation_marks_from_subtitles_mark_reported_speech():
    speech = [seg(0, timed("Er sagte wir schaffen das und dann gingen alle nach Hause", 0.0))]
    ocr = [unit(0, 0.0, 0.8, ("Er sagte:", SUB)), unit(1, 0.8, 2.0, ("„Wir schaffen das“", SUB)),
           unit(2, 2.0, 3.6, ("und dann gingen alle", SUB)), unit(3, 3.6, 4.4, ("nach Hause.", SUB))]
    stream = run(speech, ocr)
    assert speech_texts(stream) == ["Er sagte „wir schaffen das“ und dann gingen alle nach Hause"]


def test_block_context_pins_short_lines_and_edge_substitution():
    # Speech misheard "unter Ausnutzung" as "und der Auslutzung". The one-word line
    # "und" belongs to "Notlagen und", which only the block context reveals.
    row1, row2 = (180, 1080, 540, 1130), (180, 1130, 540, 1180)
    speech = [seg(0, timed("Immer öfter werden Menschen und der Auslutzung von Notlagen und Abhängigkeiten "
                           "gezwungen unter schlechten Bedingungen zu arbeiten", 0.0))]
    ocr = [unit(0, 0.0, 1.6, ("Immer öfter werden", row1), ("Menschen unter", row2)),
           unit(1, 2.0, 3.6, ("Ausnutzung von Notlagen", row1), ("und", row2)),
           unit(2, 3.6, 5.2, ("Abhängigkeiten gezwungen", row1), ("unter schlechten", row2)),
           unit(3, 5.2, 6.8, ("Bedingungen", row1), ("zu arbeiten", row2))]
    stream = run(speech, ocr, words={"unter", "und", "der", "ausnutzung"})
    assert stream.diagnostics["subtitle_tracks"][0]["n_rows"] == 2
    assert speech_texts(stream) == ["Immer öfter werden Menschen unter Ausnutzung von Notlagen und Abhängigkeiten "
                                    "gezwungen unter schlechten Bedingungen zu arbeiten"]


def test_only_hyphenated_line_breaks_are_rejoined():
    lex = SetLexicon({"verfassungsgericht", "vereinsverbote", "techunternehmen", "vertretern"})

    def join(*lines):
        return [t.surface for t in _join_line_breaks([split_tokens(ln) for ln in lines], lex)]

    assert join("Das Verfassungs-", "gericht prüft") == ["Das", "Verfassungsgericht", "prüft"]
    assert join("VEREINSVERBOT", "E") == ["VEREINSVERBOT", "E"]     # no hyphen: left alone
    assert join("die Tech-", "Unternehmen") == ["die", "Tech-Unternehmen"]   # capital: hyphen stays
    assert join("DIE VERTRE-", "TERN") == ["DIE", "VERTRETERN"]      # all capitals: no case signal
    assert join("gesell-", "der") == ["gesell-", "der"]              # not a word: left apart


def test_on_screen_words_broken_at_a_line_end_are_written_whole():
    is_word = {"vertretern", "techunternehmen"}.__contains__
    text = "Die Vertre-\ntern der Tech-\nUnternehmen\nund Kranken- und Pflegekassen"
    assert rejoin_line_breaks(text, is_word) == (
        "Die Vertretern\nder Tech-Unternehmen\nund Kranken- und Pflegekassen", 2)
    stream = run([], [unit(0, 0.0, 3.0, ("Die Vertre-", HEAD), ("tern sind da", BANNER))],
                 words={"vertretern"})
    (span,) = [s for s in stream.spans if s.origin == "onscreen"]
    assert (span.text, span.original_text) == ("Die Vertretern\nsind da", "Die Vertre-\ntern sind da")
    assert stream.diagnostics["lines"]["onscreen_words_rejoined"] == 1


def test_a_fragment_cannot_substitute_longer_words():
    row1, row2 = (180, 863, 540, 888), (180, 878, 540, 903)   # wrapped caption: one layout row
    # OCR misread the caption "GEBRANDMAR/KT" as "GEBRANNTMAR/KT"; the stub "KT" is left over.
    speech = [seg(0, timed("die als gebrannt werden mit dem Verbot", 0.0))]
    ocr = [unit(0, 0.0, 0.8, ("DIE ALS", row1)), unit(1, 0.8, 1.2, ("GEBRANNTMAR", row1), ("KT", row2)),
           unit(2, 2.0, 2.8, ("DEM VERBOT", row1))]
    words = {"kt", "werden", "mit", "gebrannt"}
    stream = run(speech, ocr, words=words)
    assert speech_texts(stream) == ["die als gebrannt werden mit dem Verbot"]
    unchecked = assemble(speech, ocr, cfg=TextConfig(substitution_length_ratio_min=0.0),
                         frame_size=FRAME, lexicon=SetLexicon(words), truecaser=FakeTruecaser())
    assert speech_texts(unchecked) != speech_texts(stream)   # the length check is what prevents it


def test_annotations_are_not_inserted_as_speech():
    speech = [seg(0, timed("Das ist richtig so", 0.0)), seg(1, timed("Vielen Dank", 2.0))]
    ocr = [unit(0, 0.0, 0.8, ("Das ist", SUB)), unit(1, 0.8, 1.6, ("richtig so. [Applaus]", SUB)),
           unit(2, 2.0, 2.8, ("Vielen Dank", SUB))]
    stream = run(speech, ocr, words={"applaus"})
    assert speech_texts(stream) == ["Das ist richtig so", "Vielen Dank"]


def test_assembly_is_deterministic():
    speech = [seg(0, timed("Bereiche heute einmal einen Blick", 0.0))]
    ocr = [unit(0, 0.0, 0.4, ("„Reiche,", SUB)), unit(1, 0.4, 1.2, ("heute einmal", SUB)),
           unit(2, 1.2, 2.0, ("einen Blick", SUB)), unit(3, 0.0, 2.0, ("Ausland", BANNER))]
    a, b = run(speech, ocr), run(speech, list(reversed(ocr)))
    assert a.spans == b.spans and a.diagnostics == b.diagnostics


def test_subtitles_in_capitals_keep_the_transcript_in_normal_case():
    speech = [seg(0, timed("Bereiche heute einmal einen Blick in die Realität waren.", 0.0))]
    ocr = [unit(0, 0.0, 0.8, ("„REICHE,", SUB)),
           unit(1, 0.8, 2.0, ("HEUTE EINMAL EINEN BLICK", SUB)),
           unit(2, 2.0, 3.2, ("IN DIE REALITÄT WAGEN.", SUB))]
    truecaser = FakeTruecaser()
    stream = run(speech, ocr, words={"reiche", "bereiche", "wagen", "waren"}, truecaser=truecaser)

    # Agreeing words keep the recognizer's spelling; the winning subtitle readings
    # take their capital from the recognizer's reading ("Bereiche" -> "Reiche").
    assert speech_texts(stream) == ["„Reiche, heute einmal einen Blick in die Realität wagen."]
    assert [(d.speech, d.chosen, d.subtitle) for d in decisions(stream)] == [
        ("Bereiche", "„Reiche,", ("„REICHE,",)), ("waren.", "wagen.", ("WAGEN.",))]
    assert stream.diagnostics["recase"]["subtitle_words_respelled"] == 2
    assert truecaser.calls == []                 # the merge needs no model


def test_onscreen_text_in_capitals_is_respelled_and_keeps_the_original():
    ocr = [unit(0, 0.0, 3.0, ("THORSTEN BECHERER", HEAD), ("BUNDESTAGSKANDIDAT", BANNER)),
           unit(1, 3.0, 4.0, ("Live aus Berlin", HEAD))]
    truecaser = FakeTruecaser({"thorsten becherer\nbundestagskandidat": "Thorsten Becherer, Bundestagskandidat."})
    stream = run([], ocr, truecaser=truecaser)

    head, live = stream.spans
    assert head.text == "Thorsten Becherer\nBundestagskandidat"
    assert head.original_text == "THORSTEN BECHERER\nBUNDESTAGSKANDIDAT"
    assert live.original_text is None
    assert stream.diagnostics["recase"]["model"] == "fake"
    assert stream.diagnostics["recase"]["onscreen_spans_respelled"] == 1

    off = assemble([], ocr, cfg=TextConfig(recase=False), frame_size=FRAME, lexicon=SetLexicon())
    assert off.spans[0].text == "THORSTEN BECHERER\nBUNDESTAGSKANDIDAT"
    assert off.diagnostics["recase"] == {"enabled": False}


def test_respelling_needs_a_truecasing_engine():
    with pytest.raises(ValueError, match="needs a truecasing engine"):
        assemble([], [unit(0, 0.0, 1.0, ("WAHL IN SASSNITZ", HEAD))], cfg=TextConfig(),
                 frame_size=FRAME, lexicon=SetLexicon())


def test_wordfreq_lexicon_with_german_compounds():
    pytest.importorskip("wordfreq")
    lx = build_lexicon("de", [], [], TextConfig())
    assert lx.is_word("hühner") and not lx.is_word("hüthner")
    assert lx.is_word("hitzeschäden")              # compound
    assert not lx.is_word("sindallerdings")        # OCR run-on, not a compound
    assert not lx.is_word("sauereenergie")         # recognizer run-on
    assert build_lexicon("xx", ["a"], ["a"], TextConfig()).name == "attested-in-both-channels"


class _FakeLLM(LLMEngine):
    @property
    def model_id(self) -> str:
        return "fake"

    def complete_json(self, system, user, schema=None):
        return {"entities": [{"text": "Reiche", "type": "PERSON"}]}


def test_entity_events_carry_span_id_channel_and_all_observations():
    span = TextSpan(span_id=f"{SRC}:speech:000001", source_id=SRC, t_start=0.0, t_end=1.0,
                    text="„Reiche, heute", channel="speech", origin="speech+subtitle",
                    observation_id=f"{SRC}:speech:000001",
                    observation_refs=(f"{SRC}:speech:000001", f"{SRC}:ocr:000003"))
    [event] = LLMEntityExtractor(_FakeLLM(), NERConfig()).extract(span, EpisodicMemory())
    assert event.event_id == f"{SRC}:speech:000001:entities"
    assert event.channel == "speech" and event.observation_refs == span.observation_refs


def test_layer3_prefers_the_explicit_channel():
    e = SemanticEvent(event_id=f"{SRC}:speech:000001:entities", source_id=SRC, extractor="entities",
                      model="m", t_start=0.0, t_end=1.0,
                      observation_refs=(f"{SRC}:speech:000001", f"{SRC}:ocr:000003"))
    assert _event_channels(e) == {"ocr", "speech"}           # derived from the ids
    assert _event_channels(e.model_copy(update={"channel": "speech"})) == {"speech"}
