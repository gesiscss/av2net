"""Tests for respelling text in capitals (layer2_extraction/text/casing.py)."""

from __future__ import annotations

from semantic_pipeline.config import TruecaseConfig
from semantic_pipeline.layer2_extraction.engines.punctuators_engine import PunctuatorsTruecaser
from semantic_pipeline.layer2_extraction.interfaces import TruecaseEngine
from semantic_pipeline.layer2_extraction.text.casing import Recaser, caps_share, take_case


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


def recaser(answers=None) -> Recaser:
    return Recaser(FakeTruecaser(answers), share_min=0.7, min_words=2)


def test_only_the_letter_case_of_the_model_output_is_taken():
    text, changed, unmatched = take_case("OSTKONGRESS × LÄNDERRAT IN SASSNITZ",
                                         "Ostkongress × Länderrat, in Sassnitz am Meer.")
    assert text == "Ostkongress × Länderrat in Sassnitz"   # added words and commas are ignored
    assert (changed, unmatched) == (4, 0)
    text, changed, unmatched = take_case("WAHL IN SASSNITZ", "Wahl im Sassnitz")
    assert text == "Wahl in Sassnitz" and unmatched == 1    # a changed word is lowercased
    # Words not in capitals stay as they are; ß for SS is a difference of case.
    assert take_case("derSRAT STRASSE", "Dersrat Straße")[0] == "derSRAT Straße"


def test_caps_share_counts_words_of_two_or_more_letters():
    assert caps_share("SPD und CDU") == (2 / 3, 3)
    assert caps_share("A B 12") == (0.0, 0)


def test_onscreen_respells_text_in_capitals_in_one_call():
    r = recaser({"thorsten becherer\nbundestagskandidat": "Thorsten Becherer, Bundestagskandidat.",
                 "thorsten becherer": "Thorsten Becherer"})
    texts = ["Die Wahl in Erfurt", "SPD", "THORSTEN BECHERER\nBUNDESTAGSKANDIDAT",
             "Heute live:\nTHORSTEN BECHERER"]
    assert r.onscreen(texts) == [None, None, "Thorsten Becherer\nBundestagskandidat",
                                 "Heute live:\nThorsten Becherer"]
    # One call for all texts; a single word may be an abbreviation, and in a text
    # that is not in capitals only its lines in capitals are respelled.
    assert r.engine.calls == [["THORSTEN BECHERER\nBUNDESTAGSKANDIDAT", "THORSTEN BECHERER"]]
    assert r.stats["spans_in_capitals"] == 1 and r.stats["lines_in_capitals"] == 1
    assert r.stats["words_respelled"] == 5


def test_text_without_capitals_does_not_call_the_model():
    r = recaser()
    assert r.onscreen(["Die Wahl in Erfurt", "SPD"]) == [None, None]
    assert r.engine.calls == []


def test_subtitle_reading_takes_capitals_from_the_recognizer():
    r = recaser()
    assert r.reading(("SIE",), ("Die",)) == ("Sie",)          # sentence start in the speech
    assert r.reading(("WURZEL",), ("Wurzeln",)) == ("Wurzel",)
    assert r.reading(("DAS",), ("dass",)) == ("das",)
    assert r.reading(("SPÖ",), ("SPD",)) == ("SPÖ",)          # the recognizer writes capitals too
    assert r.reading(("„REICHE,",), ("Bereiche",)) == ("„Reiche,",)
    assert r.reading(("ZU", "ERST"), ("Zuerst",)) == ("Zu", "erst")  # counts differ: first word only
    assert r.reading(("NEUE", "WAHL"), None) == ("neue", "wahl")     # an insert: no recognizer reading
    assert r.reading(("Reiche",), ("Bereiche",)) == ("Reiche",)      # not in capitals
    assert r.reading(("A",), ("ein",)) == ("A",)                      # no word of two letters
    assert r.engine.calls == []


def test_engine_id_names_the_pinned_model_without_loading_it():
    engine = PunctuatorsTruecaser(TruecaseConfig())
    assert engine.model_id.startswith(
        "punctuators:1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase@d1769a597ce8+punctuators")
    assert engine.truecase([]) == [] and engine._model is None
