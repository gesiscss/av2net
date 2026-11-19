"""Tests for the stored dependency parse: writing and reading CoNLL-U, the stored
parse served as a tagger, the conversion of Stanza's output, and the check that
a parse still matches its text stream.

Fake Stanza objects keep the tests independent of model downloads.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from run_common import PARSE_FILE, PARSE_MANIFEST, parse_problem
from semantic_pipeline.config import ConceptConfig, ExtractionConfig, StanzaConfig
from semantic_pipeline.layer2_extraction.context import TextSpan
from semantic_pipeline.layer2_extraction.engines.stanza_engine import convert_document
from semantic_pipeline.layer2_extraction.extractors.concepts import ConceptExtractor
from semantic_pipeline.layer2_extraction.factory import build_extractors
from semantic_pipeline.layer2_extraction.memory import EpisodicMemory
from semantic_pipeline.layer2_extraction.parse import (
    MultiWordToken,
    ParsedSentence,
    ParsedSpan,
    ParsedWord,
    StoredParse,
    format_span,
    parse_conllu,
    read_conllu,
    text_sha1,
    tokens_of,
    write_conllu,
)

SID = "vid:speech:000001"
TEXT = "Die Polizei räumte am Hafen\nzwei Häuser."
WORDS = (
    ParsedWord(1, "Die", "der", "DET", "ART", "Case=Nom|Definite=Def", 2, "det", 0, 3),
    ParsedWord(2, "Polizei", "Polizei", "NOUN", "NN", "Case=Nom", 3, "nsubj", 4, 11),
    ParsedWord(3, "räumte", "räumen", "VERB", "VVFIN", "Mood=Ind", 0, "root", 12, 18),
    ParsedWord(4, "an", "an", "ADP", "APPR", None, 6, "case", 19, 21),      # "am" = an + dem
    ParsedWord(5, "dem", "der", "DET", "ART", None, 6, "det", 19, 21),
    ParsedWord(6, "Hafen", "Hafen", "NOUN", "NN", None, 3, "obl", 22, 27),
    ParsedWord(7, "zwei", "zwei", "NUM", "CARD", "NumType=Card", 8, "nummod", 28, 32),
    ParsedWord(8, "Häuser", "Haus", "NOUN", "NN", None, 3, "obj", 33, 39),
    ParsedWord(9, ".", ".", "PUNCT", "$.", None, 3, "punct", 39, 40),
)
SENTENCE = ParsedSentence(TEXT, WORDS, (MultiWordToken(4, 5, "am", 19, 21),))
SPAN = ParsedSpan(SID, "de", text_sha1(TEXT), "stanza-test", (SENTENCE,))


def _text_span(text: str = TEXT, language: str | None = "de") -> TextSpan:
    return TextSpan(span_id=SID, source_id="vid", t_start=0.0, t_end=2.0, text=text,
                    channel="speech", origin="speech", observation_id=SID, language=language)


def test_the_example_offsets_match_the_text():
    for w in WORDS:
        expected = "am" if w.id in (4, 5) else w.text
        assert TEXT[w.start:w.end] == expected


def test_conllu_round_trip_keeps_words_offsets_and_multiword_tokens(tmp_path):
    path = tmp_path / PARSE_FILE
    empty = ParsedSpan("vid:ocr:000002", "de", text_sha1("  "), "stanza-test", ())
    assert write_conllu(path, [SPAN, empty]) == 1        # a span without words gets no block
    (back,) = list(read_conllu(path))
    assert (back.span_id, back.language, back.text_sha1, back.parser) == (SID, "de", SPAN.text_sha1, "stanza-test")
    (sentence,) = back.sentences
    assert sentence.words == WORDS
    assert sentence.multiword == SENTENCE.multiword
    assert sentence.text == TEXT.replace("\n", " ")     # a comment holds one line
    assert b"\r" not in path.read_bytes()


def test_word_lines_have_ten_columns_and_every_word_keeps_its_offsets():
    lines = format_span(SPAN).split("\n")
    rows = [line.split("\t") for line in lines if line and not line.startswith("#")]
    assert all(len(r) == 10 for r in rows)
    assert ["4-5", "am"] == rows[3][:2] and rows[3][9] == "start_char=19|end_char=21"
    assert rows[4][0] == "4" and rows[4][9] == rows[5][9] == "start_char=19|end_char=21"
    assert rows[1][:8] == ["2", "Polizei", "Polizei", "NOUN", "NN", "Case=Nom", "3", "nsubj"]
    assert lines[-2:] == ["", ""]                      # the sentence ends in a blank line


def test_words_written_out_inside_a_token_keep_their_own_offsets():
    text = "that's it"
    words = (ParsedWord(1, "that", "that", "PRON", None, None, 0, "root", 0, 4),
             ParsedWord(2, "'s", "be", "AUX", None, None, 1, "cop", 4, 6),
             ParsedWord(3, "it", "it", "PRON", None, None, 1, "nsubj", 7, 9))
    sentence = ParsedSentence(text, words, (MultiWordToken(1, 2, "that's", 0, 6),))
    span = ParsedSpan("vid:speech:000004", "en", text_sha1(text), "stanza-test", (sentence,))
    (back,) = list(parse_conllu(format_span(span).splitlines(keepends=True)))
    assert back.sentences[0].words == words


def test_words_without_offsets_take_their_tokens():
    block = ("# newdoc id = s\n# language = de\n# text_sha1 = x\n# parser = p\n"
             "1-2\tam\t_\t_\t_\t_\t_\t_\t_\tstart_char=0|end_char=2\n"
             "1\tan\tan\tADP\t_\t_\t2\tcase\t_\t_\n"
             "2\tdem\tder\tDET\t_\t_\t0\troot\t_\t_\n\n")
    (span,) = list(parse_conllu(block.splitlines(keepends=True)))
    assert [(w.start, w.end) for w in span.sentences[0].words] == [(0, 2), (0, 2)]


def test_missing_fields_are_written_as_underscore_and_read_back_as_none():
    words = (ParsedWord(1, "Haus", None, "NOUN", None, None, None, None, 0, 4),
             ParsedWord(2, "_", "_", "SYM", None, None, None, None, 5, 6))
    span = ParsedSpan("vid:ocr:000003", "de", text_sha1("Haus _"), "stanza-test",
                      (ParsedSentence("Haus _", words),))
    (back,) = list(parse_conllu(format_span(span).splitlines(keepends=True)))
    assert back.sentences[0].words == words


def test_stored_parse_serves_the_words_by_language_and_text():
    store = StoredParse([SPAN])
    assert store.model_id == "stanza-test" and store.supports("de") and len(store) == 1
    tokens = store.tag_tokens(TEXT, "de")
    assert tokens == tokens_of([SENTENCE])
    assert [(t.text, t.start, t.end) for t in tokens[3:5]] == [("an", 19, 21), ("dem", 19, 21)]
    assert store.tag_tokens(TEXT, "en") is None          # language not in the parse
    assert store.tag_tokens("   ", "de") == []
    assert store.tag_tokens("Ein anderer Text.", "de") == [] and store.misses == 1


def test_concept_extractor_reads_the_stored_parse():
    (event,) = ConceptExtractor(StoredParse([SPAN]), ConceptConfig()).extract(
        _text_span(), EpisodicMemory())
    assert [n.lemma for n in event.nodes] == ["Polizei", "Hafen", "Haus"]
    assert event.model == "stanza-test"


def test_factory_hands_a_given_tagger_to_the_concept_extractor():
    store = StoredParse([SPAN])
    extractors, engines = build_extractors(ExtractionConfig(), names=("concepts",), tagger=store)
    assert engines == {"tagger": store}
    assert [e.model_id for e in extractors] == ["stanza-test"]


def test_stanza_runs_the_dependency_parser_by_default():
    assert "depparse" in StanzaConfig().processors


def _word(id, text, lemma, upos, head, deprel, start=None, end=None):
    return SimpleNamespace(id=id, text=text, lemma=lemma, upos=upos, xpos=None, feats=None,
                           head=head, deprel=deprel, start_char=start, end_char=end)


def _token(ids, text, start, end, words):
    return SimpleNamespace(id=ids, text=text, start_char=start, end_char=end, words=words)


def test_stanza_output_is_converted_with_multiword_tokens_and_without_a_parser():
    doc = SimpleNamespace(sentences=[SimpleNamespace(text="Die Polizei räumte am Hafen", tokens=[
        _token((1,), "Die", 0, 3, [_word(1, "Die", "der", "DET", 2, "det", 0, 3)]),
        _token((2,), "Polizei", 4, 11, [_word(2, "Polizei", "Polizei", "NOUN", 3, "nsubj", 4, 11)]),
        _token((3,), "räumte", 12, 18, [_word(3, "räumte", "räumen", "VERB", 0, "root", 12, 18)]),
        _token((4, 5), "am", 19, 21, [_word(4, "an", "an", "ADP", 6, "case"),
                                      _word(5, "dem", "der", "DET", 6, "det")]),
        _token((6,), "Hafen", 22, 27, [_word(6, "Hafen", "Hafen", "NOUN", None, None, 22, 27)]),
    ])])
    (sentence,) = convert_document(doc)
    assert sentence.multiword == (MultiWordToken(4, 5, "am", 19, 21),)
    assert [(w.id, w.text, w.start, w.end) for w in sentence.words[3:5]] == [(4, "an", 19, 21), (5, "dem", 19, 21)]
    assert sentence.words[1].head == 3 and sentence.words[1].deprel == "nsubj"
    assert sentence.words[5].head is None and sentence.words[5].deprel is None


def test_a_missing_or_outdated_parse_is_detected(tmp_path):
    text = tmp_path / "text_stream.jsonl"
    text.write_text('{"span": 1}\n', encoding="utf-8")
    assert "run run_parse.py first" in parse_problem(tmp_path)
    (tmp_path / PARSE_FILE).write_text("", encoding="utf-8")
    digest = hashlib.sha1(text.read_bytes()).hexdigest()
    (tmp_path / PARSE_MANIFEST).write_text(json.dumps({"text_stream_sha1": digest}), encoding="utf-8")
    assert parse_problem(tmp_path) is None
    text.write_text('{"span": 2}\n', encoding="utf-8")
    assert "out of date" in parse_problem(tmp_path)
