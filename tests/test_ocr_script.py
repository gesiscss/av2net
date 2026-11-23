"""Tests for choosing the script of on-screen text before its language.

The decision rule runs on synthetic readings (no models); the frame layout runs
on a small synthetic video. Language groups come from langdetect's profiles.
"""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from semantic_pipeline.config import Config, OCRConfig
from semantic_pipeline.language import (
    char_script, detect_text_language, label_languages, languages_by_script, script_share,
    text_language_probabilities,
)
from semantic_pipeline.layer1_observation.factory import build_script_ocr
from semantic_pipeline.layer1_observation.interfaces import OCREngine
from semantic_pipeline.layer1_observation.ocr_script import (
    choose_script, decide_script, language_frame_slots,
)
from semantic_pipeline.layer1_observation import ocr_stream
from semantic_pipeline.layer1_observation.ocr_stream import _sample_frames, detect_and_build_ocr

CFG = OCRConfig()


# --------------------------------------------------------------------------- #
# Scripts of characters and languages
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ch,script", [
    ("a", "latin"), ("Ä", "latin"), ("Ж", "cyrillic"), ("한", "hangul"),
    ("中", "cjk"), ("の", "cjk"), ("カ", "cjk"), ("ー", "cjk"), ("ｶ", "cjk"),
    ("Ａ", "latin"), ("ب", "arabic"), ("λ", "greek"), ("क", "devanagari"),
    ("ि", "devanagari"), ("ก", "thai"), ("ಕ", "kannada"),
    ("7", None), ("-", None), (" ", None), ("€", None),
])
def test_char_script(ch, script):
    assert char_script(ch) == script


def test_script_share_counts_letters_only():
    assert script_share("Москва 2019", "cyrillic") == 1.0
    assert script_share("RT Москва", "cyrillic") == pytest.approx(6 / 8)
    assert script_share("2019", "cyrillic") == 0.0


def test_languages_by_script_from_langdetect_profiles():
    groups = languages_by_script()
    assert set(groups["cjk"]) == {"ja", "zh-cn", "zh-tw"}
    assert set(groups["hangul"]) == {"ko"}
    assert {"ru", "uk", "bg", "mk"} <= set(groups["cyrillic"])
    assert {"ar", "fa", "ur"} <= set(groups["arabic"])
    assert "kn" in groups["kannada"]
    assert {"de", "en", "fr"} <= set(groups["latin"])


def test_language_detection_restricted_to_a_script():
    ko = "기상청은 오늘 전국에 폭염 경보를 내렸습니다"
    assert detect_text_language(ko, languages=languages_by_script()["hangul"])[0] == "ko"
    zh = "北京冬奥会竞赛日程发布 新华社记者报道"
    assert detect_text_language(zh, languages=languages_by_script()["cjk"])[0].startswith("zh")
    ru = "Территория инноваций Красноярск"
    assert detect_text_language(ru, languages=languages_by_script()["cyrillic"])[0] == "ru"
    assert detect_text_language("short", languages=("de",)) == (None, 0.0)


def test_label_languages_of_a_script_and_of_the_speech():
    groups = languages_by_script()
    assert label_languages("cyrillic") == groups["cyrillic"]
    assert label_languages("cyrillic", "ru") == ("ru",)
    assert label_languages("cyrillic", "en") == groups["cyrillic"]  # spoken in another script
    assert set(label_languages("cjk", "zh")) == {"zh-cn", "zh-tw"}  # region part ignored
    assert label_languages("cjk", "ja") == ("ja",)
    assert label_languages("hangul", "ko") == ("ko",)
    assert label_languages("klingon", "ru") == ()


def test_short_text_leaves_close_languages_open():
    # Russian news captions, names only: Bulgarian ranks first, Russian stays listed.
    news = "Вашингтон, США\nСирия\nОСТИ\nВашингтон, США\nВашингтон, США\nВашингтон, США\nНОВОСТИ\nСирия"
    ranked = text_language_probabilities(news, languages=languages_by_script()["cyrillic"])
    assert [lang for lang, _ in ranked] == ["bg", "ru"]
    assert detect_text_language(news, languages=languages_by_script()["cyrillic"]) == ranked[0]
    # Traditional Chinese subtitles (no kana) rule Japanese out.
    zh = "當初二十年前第一次舉辦的時候，還是在一個地方很小的體育館\n今後也請各位多多指教，多謝"
    assert [lang for lang, _ in text_language_probabilities(zh, languages=languages_by_script()["cjk"])] == ["zh-tw"]
    assert text_language_probabilities("short") == []


# --------------------------------------------------------------------------- #
# Decision rule
# --------------------------------------------------------------------------- #
def _box(latin, **others):
    """latin: (text, conf) of the Latin model; others: script=(text, conf)."""
    return {"latin": latin, **others}


def test_script_wins_with_two_distinct_texts():
    boxes = [_box(("Teppnropua", 0.6), cyrillic=("Территория", 0.9), greek=("Τeppιτopια", 0.5)),
             _box(("Hhhobaun", 0.5), cyrillic=("Инноваций", 0.85), greek=("", 0.0)),
             _box(("Krasnoyarsk", 0.9))]
    assert decide_script(boxes, CFG) == ("cyrillic", {"cyrillic": 2}, 30)


def test_latin_text_stays_latin():
    boxes = [_box(("Bundestag debattiert", 0.98), cyrillic=("Bundestag debattiert", 0.97)),
             _box(("Heute im Parlament", 0.97), cyrillic=("Heute im Parlament", 0.96))]
    assert decide_script(boxes, CFG)[0] == "latin"


def test_model_must_be_as_confident_as_the_latin_model():
    # Cyrillic capitals that look Latin: the Latin model reads them just as well.
    boxes = [_box(("KPACHOAPCK", 0.95), cyrillic=("КРАСНОЯРСК", 0.90)),
             _box(("POCCNR", 0.93), cyrillic=("РОССИЯ", 0.91)),
             _box(("Nachrichten aus Sibirien", 0.95))]
    assert decide_script(boxes, CFG)[:2] == ("latin", {})


def test_repeated_logo_is_one_text():
    logo = _box(("i", 0.43), thai=("ซี", 0.86))
    filler = _box(("NEWS TODAY LIVE", 0.95), thai=("NEWS TODAY LIVE", 0.9))
    assert decide_script([logo] * 5 + [filler], CFG)[:2] == ("latin", {"thai": 1})


def test_mixed_script_reading_does_not_claim():
    boxes = [_box((".n AInN7 7I", 0.64), greek=(".n aηx7 7ι 71v ε η07", 0.70)),
             _box(("1x7 te np", 0.60), greek=("1×7 τε πρ i7 2 πίν π", 0.77)),
             _box(("Something else here", 0.9))]
    script, texts, _ = decide_script(boxes, CFG)
    assert script == "latin" and texts == {"greek": 1}


def test_low_confidence_does_not_claim():
    boxes = [_box(("", 0.0), hangul=("폭염 경보", 0.65)), _box(("", 0.0), hangul=("기상청", 0.6)),
             _box(("filler text with letters", 0.9))]
    assert decide_script(boxes, CFG)[:2] == ("latin", {})


def test_most_confident_claim_wins_the_box():
    boxes = [_box(("", 0.1), cyrillic=("АБВ", 0.8), greek=("ΑΒΓΔ", 0.95)),
             _box(("", 0.1), cyrillic=("ЕЖЗ", 0.8), greek=("ΕΖΗΘ", 0.9)),
             _box(("enough letters for a decision", 0.9))]
    assert decide_script(boxes, CFG)[:2] == ("greek", {"greek": 2})


def test_too_little_text_is_undetermined():
    boxes = [_box(("", 0.0), hangul=("폭염", 0.95)), _box(("", 0.0), hangul=("경보", 0.95))]
    assert decide_script(boxes, CFG) == ("undetermined", {"hangul": 2}, 4)
    assert decide_script([], CFG) == ("undetermined", {}, 0)


# --------------------------------------------------------------------------- #
# Box selection and the readings used for the language
# --------------------------------------------------------------------------- #
class _FakeEngine(OCREngine):
    """Boxes are crops of a given height; a reading is looked up by that height."""

    def __init__(self, script, readings, boxes_per_frame=None):
        self.script, self.readings, self.boxes = script, readings, boxes_per_frame or {}

    @property
    def model_id(self):
        return f"fake:{self.script}"

    def read(self, image):
        return []

    def detect_text_boxes(self, image):
        return [np.zeros((h, 10, 3), np.uint8) for h in self.boxes.get(int(image[0, 0, 0]), [])]

    def recognize(self, crops):
        return [self.readings.get(c.shape[0], ("", 0.0)) for c in crops]


def test_choose_script_reads_the_largest_boxes_of_each_frame_set():
    frames = [(True, np.full((4, 4, 3), 1, np.uint8)), (False, np.full((4, 4, 3), 2, np.uint8))]
    boxes = {1: [10, 30, 20], 2: [40, 50, 5]}  # heights; areas are height x 10
    cyr = {30: ("Территория", 0.9), 20: ("Инноваций", 0.8), 40: ("Красноярск", 0.9),
           50: ("Россия", 0.4), 10: ("Москва", 0.99), 5: ("Мир", 0.99)}
    engines = {"latin": _FakeEngine("latin", {}, boxes), "cyrillic": _FakeEngine("cyrillic", cyr)}
    choice = choose_script(frames, engines, replace(CFG, script_max_boxes=2))
    # read: 30, 20 (language frame) and 50, 40 (other frame); not 10 and 5
    assert choice.script == "cyrillic"
    assert choice.texts == {"cyrillic": 3}
    assert choice.lines == ("Территория", "Инноваций", "Красноярск")  # "Россия" is below 0.5


# --------------------------------------------------------------------------- #
# Frames and engines
# --------------------------------------------------------------------------- #
def test_language_frames_are_among_the_script_frames():
    assert language_frame_slots(5) == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    assert language_frame_slots(15) == {1: 0, 4: 1, 7: 2, 10: 3, 13: 4}
    for n in (5, 15, 25, 35):
        for j, k in language_frame_slots(n).items():
            assert (j + 0.5) / n == pytest.approx((k + 0.5) / 5)
    for n in (3, 10, 12, 20):
        with pytest.raises(ValueError):
            language_frame_slots(n)


def test_sample_frames_flags_the_language_frames(tmp_path):
    path = str(tmp_path / "v.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32))
    for i in range(60):
        writer.write(np.full((32, 32, 3), i * 4, np.uint8))
    writer.release()
    frames = _sample_frames(path, Config.local_default(), 15)
    assert len(frames) == 15
    assert [j for j, (flag, _) in enumerate(frames) if flag] == [1, 4, 7, 10, 13]
    assert [flag for flag, _ in _sample_frames(path, Config.local_default(), 5)] == [True] * 5


class _EveryFrame(_FakeEngine):
    """The same boxes in every frame."""

    def detect_text_boxes(self, image):
        return [np.zeros((h, 10, 3), np.uint8) for h in self.boxes]


def test_speech_decides_among_the_languages_the_text_leaves_open(tmp_path, monkeypatch):
    path = str(tmp_path / "v.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32))
    for _ in range(60):
        writer.write(np.zeros((32, 32, 3), np.uint8))
    writer.release()
    cyr = {30: ("Вашингтон, США", 0.9), 20: ("Сирия Новости", 0.9)}
    engines = {"latin": _EveryFrame("latin", {}, [30, 20]), "cyrillic": _FakeEngine("cyrillic", cyr)}
    monkeypatch.setattr(ocr_stream, "_script_engines", lambda config: engines)
    seen = []

    def ranking(text, **kw):  # what langdetect reports for the text, fixed here
        seen.append((text, kw["languages"]))
        return [("bg", 0.857), ("ru", 0.143)]

    monkeypatch.setattr(ocr_stream, "text_language_probabilities", ranking)
    config = Config.local_default()
    routing = detect_and_build_ocr(path, config, spoken_language="ru")
    assert (routing.script, routing.language, routing.language_hint) == ("cyrillic", "ru", "ru")
    assert routing.language_confidence == 0.143
    assert routing.engine is engines["cyrillic"]
    assert seen[0][1] == languages_by_script()["cyrillic"]
    assert set(seen[0][0].split("\n")) == {"Вашингтон, США", "Сирия Новости"}  # script test readings
    # no speech, speech in another script, or a language the text rules out
    for spoken in (None, "en", "uk"):
        routing = detect_and_build_ocr(path, config, spoken_language=spoken)
        assert (routing.language, routing.language_confidence, routing.language_hint) == ("bg", 0.857, None)


def test_script_test_is_skipped_for_latin_only_or_other_backends():
    cfg = Config.local_default()
    assert build_script_ocr(replace(cfg, ocr=replace(cfg.ocr, scripts=("latin",)))) is None
    assert build_script_ocr(replace(cfg, ocr=replace(cfg.ocr, backend="paddleocr"))) is None
    with pytest.raises(ValueError, match="klingon"):
        build_script_ocr(replace(cfg, ocr=replace(cfg.ocr, scripts=("latin", "klingon"))))


def test_rapidocr_models_for_scripts_and_languages():
    from rapidocr import LangRec

    from semantic_pipeline.layer1_observation.ocr_rapidocr_onnx import SCRIPT_MODELS, _iso_to_langrec

    assert set(SCRIPT_MODELS) == set(CFG.scripts)
    assert _iso_to_langrec("kn") == LangRec.KA  # RapidOCR's "ka" model is Kannada
    assert _iso_to_langrec("mk") == LangRec.CYRILLIC
    assert _iso_to_langrec("ka") == LangRec.LATIN  # no Georgian model
