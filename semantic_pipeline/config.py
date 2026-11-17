"""Configuration for the pipeline.

Model selection is a configuration concern, not a pipeline-logic concern.
Switching from a local model to a large production model should mean changing
values here (and adding a backend behind the same interface), never editing the
transformation code. Config objects are immutable (frozen dataclasses) so a run
is fully described by the config it was given.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class SpeechConfig:
    backend: str = "faster-whisper"
    model_size: str = "base"        # tiny | base | small | medium | large-v3
    compute_type: str = "int8"      # int8 is CPU-friendly
    device: str = "cpu"
    beam_size: int = 5
    language: str | None = None     # None = autodetect


@dataclass(frozen=True)
class OCRConfig:
    backend: str = "rapidocr-onnx"  # onnxruntime; fast on CPU, language-routed
    min_confidence: float = 0.5     # drop OCR lines below this score
    language: str = "auto"          # "auto" = detect from the frames; else an ISO code
    # Region-tracking stream parameters:
    sample_dt: float = 0.5          # seconds between change-gate samples
    change_threshold: float = 4.0   # mean abs frame diff (0-255) to trigger re-OCR
    text_similarity_min: float = 0.65  # >= this ratio -> same text continues a track
    y_overlap_min: float = 0.3      # min vertical IoU to associate a block with a track
    gap_factor: float = 1.6         # line-gap > factor*median_height starts a new block


@dataclass(frozen=True)
class VisionConfig:
    backend: str = "ollama"
    model: str = "moondream"
    host: str | None = None         # None = default local Ollama server
    temperature: float = 0.0        # 0 for as-deterministic-as-possible captions
    caption_language: str = "auto"  # "auto" = align to the video's main (speech) language
    shot_threshold: float = 27.0    # PySceneDetect ContentDetector cut threshold
    min_shot_seconds: float = 0.6   # minimum shot length (suppresses micro-shots)
    prompt: str = (
        "Describe the visual scene in this single video frame in one or two "
        "sentences: the people present and what they are doing, notable objects, "
        "and the setting or location. Do not read, transcribe, or mention any "
        "on-screen text, captions, or logos (a separate system handles text). Do "
        "not speculate about audio or about what happens before or after."
    )


@dataclass(frozen=True)
class ObservationConfig:
    # Frame sampling for the OCR and vision streams, and a limit on the processed duration.
    frame_position: str = "mid"     # where in a span to sample the frame: start|mid|end
    max_duration_s: float | None = None  # cap processing; None = whole video
    jpeg_quality: int = 90


@dataclass(frozen=True)
class LLMConfig:
    """A shared instruction-tuned LLM engine (the resource axis of Layer 2).

    One engine is built once per run and shared by every extractor that needs an
    LLM. Local mode uses a small instruction-tuned model served by Ollama;
    production swaps the model (and, if needed, the backend) here, never in the
    extractor logic.
    """

    backend: str = "ollama"
    model: str = "qwen2.5:3b"       # small instruction-tuned LM, local via Ollama;
                                    # qwen2.5:14b is the larger option (much slower on CPU)
    host: str | None = None         # None = default local Ollama server
    temperature: float = 0.0        # 0 + fixed seed for as-deterministic-as-possible
    seed: int = 0
    num_ctx: int = 4096


@dataclass(frozen=True)
class GlinerConfig:
    """A GLiNER span-tagging model, the default NER engine.

    GLiNER reads short label strings (not definitions) and returns labelled text
    spans with scores. It runs on CPU without sampling, so the same text gives
    the same spans. The multilingual checkpoint takes English labels for text in
    any language. The model revision is pinned so that a run can be repeated.
    """

    model: str = "urchade/gliner_multi-v2.1"   # Apache-2.0, multilingual
    revision: str | None = "443d26d654e0324125a96bebd8e796c14ff2efe6"  # None = latest
    threshold: float = 0.5          # minimum span score
    device: str = "cpu"
    max_words: int = 150            # longer texts are tagged in chunks of whole words


@dataclass(frozen=True)
class NERConfig:
    """The entity extractor: names in the text, typed with the name types in
    ontology.py. Person names must come from text, never from vision.

    Two engines can run it: "gliner" (default; a dedicated span-tagging model)
    or "llm" (the shared instruction-tuned LLM).
    """

    enabled: bool = True
    engine: str = "gliner"          # "gliner" | "llm"
    # GLiNER labels and the name type each stands for (ontology.NAME_TYPES); a
    # corpus may add labels mapped to existing types. Labels mapped to None are
    # decoys: they draw dates, times and numbers away from the name labels and
    # are then discarded (they are not names, and an event already has a time).
    gliner_labels: tuple[tuple[str, str | None], ...] = (
        ("person", "Person"),
        ("nationality", "Group"),
        ("organization", "Organization"),
        ("location", "Place"),
        ("event", "Event"),
        ("work of art", "Work"),
        ("product", "Product"),
        ("language", "Other"),
        ("date", None),
        ("time", None),
        ("number", None),
    )
    # A single word among the most frequent words of the span's language
    # (pronouns, articles: wordfreq Zipf at least this) is not a name. Skipped
    # for languages wordfreq does not cover.
    drop_function_words: bool = True
    function_word_min_zipf: float = 6.0


@dataclass(frozen=True)
class StanzaConfig:
    """A Stanza pipeline: tokenizer, part-of-speech tagger and lemmatizer.

    Stanza has models for over 100 languages, all tagged with the Universal
    Dependencies part-of-speech set, so the same code serves every language. One
    pipeline is built per language, on first use, from the language each text
    span records. The installed Stanza version fixes the model files (its
    resources version), so pinning the package pins the models.
    """

    processors: tuple[str, ...] = ("tokenize", "mwt", "pos", "lemma")  # mwt only where it exists
    package: str = "default"        # Stanza's default model package per language
    device: str = "cpu"
    model_dir: str | None = None    # None = Stanza's default model cache


@dataclass(frozen=True)
class ConceptConfig:
    """The concept extractor: every common noun in the text, by lemma, with no
    salience judgment. Proper nouns are left to NER. Categories are not assigned
    here; Layer 3 takes them from the generated category table."""

    enabled: bool = True
    upos: tuple[str, ...] = ("NOUN",)   # Universal Dependencies tags taken as concepts


@dataclass(frozen=True)
class ConceptCategoryConfig:
    """How the generated category table is filled (run_concept_table.py).

    Two engines can fill it, as for NER; each writes its own table file:

    * "gliner" (default): GLiNER labels each noun occurrence with category
      words; a label counts only when the labelled stretch contains exactly one
      noun. Each lemma gets the category with the most votes over its
      occurrences, or two when they tie; lemmas without a vote are unassigned.
    * "llm": the shared instruction-tuned LLM (LLMConfig; --model overrides it)
      reads the category definitions with examples in several languages and
      assigns each lemma one category, or two if ambiguous, from one snippet
      where it occurs. Lemmas are sent in batches.
    """

    engine: str = "gliner"          # "gliner" | "llm"
    gliner_labels: tuple[tuple[str, str], ...] = (
        ("people", "agent"),
        ("job or role", "role"),
        ("organization", "organization"),
        ("place", "location"),
        ("event", "event"),
        ("activity", "task"),
        ("object or material", "resource"),
        ("information or idea", "knowledge"),
        ("quality or state", "attribute"),
        ("time", "time"),
    )
    threshold: float = 0.3          # minimum GLiNER score for a category label
    max_categories: int = 2
    llm_batch_size: int = 30        # lemmas per LLM call
    snippet_chars: int = 40         # LLM context: characters on each side of the noun,
                                    # extended to whole words


@dataclass(frozen=True)
class TextConfig:
    """Layer 2 text assembly: speech + OCR observations -> one text stream.

    Burned-in subtitles are found from evidence in each video (no screen region
    is assumed) and merged word by word into the speech transcript; other
    on-screen text stays its own channel. See layer2_extraction/text/.
    """

    # Word-level alignment of an OCR line into the speech tokens.
    token_sim_min: float = 0.72         # min similarity for a one-to-one word match
    merge_sim_min: float = 0.78         # min similarity when a match joins several words
    merge_token_cover_min: float = 0.5  # each joined word must be this much covered
    skip_ocr_penalty: float = 0.4       # cost of leaving an OCR word unmatched
    skip_speech_penalty: float = 0.15   # cost of skipping a spoken word inside a line
    window_discover_s: float = 3.0      # speech searched around a line (track discovery)
    window_refine_s: float = 1.5        # ... and around a subtitle line, offset-corrected
    # Subtitle tracks: layout rows scored by how they render the speech.
    layout_y_gap: float = 0.02          # split rows at vertical gaps (frame fraction)
    layout_height_ratio: float = 1.6    # lines in a row have similar text height
    line_aligned_min: float = 0.8       # a line "aligns" if this share of its words match
    # Multi-line subtitles: adjacent subtitle-like rows stack into one candidate.
    stack_aligned_share_min: float = 0.75  # a row is subtitle-like if most lines align
    stack_gap_lines: float = 1.6           # max vertical step between rows (text heights)
    stack_x_tol: float = 0.03              # same horizontal centre (frame fraction)
    stack_height_ratio: float = 1.25       # same text height
    track_min_lines: int = 3
    track_aligned_share_min: float = 0.6
    track_speech_coverage_min: float = 0.15   # must render this share of the speech
    track_offset_iqr_max_s: float = 1.0       # stable time offset to the speech
    track_median_lifetime_max_s: float = 4.0  # subtitle lines are short-lived
    # Merging subtitle blocks (an OCR unit's subtitle lines) into the transcript.
    merge_line_coverage_min: float = 0.5  # a subtitle block merges if this share aligns
    max_gap_tokens: int = 4               # longest word run treated as a substitution
    substitution_length_ratio_min: float = 0.5  # ... and of similar length (letters)
    tie_break: str = "subtitle"           # "subtitle" | "speech": who wins a tie
    # On-screen text that the speaker reads out (flagged, never merged).
    read_aloud_coverage_min: float = 0.8
    read_aloud_min_tokens: int = 3
    # Lexicon (is a reading a real word?), chosen by the video's language.
    lexicon: str = "wordfreq"             # "wordfreq" | "attested" (both channels only)
    lexicon_min_zipf: float = 1.0         # wordfreq Zipf frequency counting as a word
    compound_languages: tuple[str, ...] = ("de", "nl", "sv", "da", "nb", "no", "fi", "is")
    compound_part_min_len: int = 4
    compound_part_min_zipf: float = 2.5
    compound_part_max_zipf: float = 6.0   # function words (sind, der) are no compound parts
    # Text in capitals is respelled by a truecasing model before any extractor
    # reads it (see layer2_extraction/text/casing.py and TruecaseConfig).
    recase: bool = True
    recase_share_min: float = 0.7         # text is in capitals from this share of its words
    recase_min_words: int = 2             # ... counting words of two or more letters


@dataclass(frozen=True)
class TruecaseConfig:
    """A model that restores letter case (and punctuation, which is not used) in
    lower-case text: an XLM-RoBERTa model trained on 47 languages, run on CPU
    with ONNX Runtime through the punctuators package. Text assembly uses it for
    text in capitals. The model revision is pinned so that a run can be repeated.
    """

    model: str = "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase"
    revision: str = "d1769a597ce8dfaa070d436bc67d4ee761f58884"


@dataclass(frozen=True)
class ExtractionConfig:
    """Layer 2 configuration: text assembly, shared engines, per-extractor settings."""

    text: TextConfig = field(default_factory=TextConfig)
    truecase: TruecaseConfig = field(default_factory=TruecaseConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    gliner: GlinerConfig = field(default_factory=GlinerConfig)
    stanza: StanzaConfig = field(default_factory=StanzaConfig)
    ner: NERConfig = field(default_factory=NERConfig)
    concepts: ConceptConfig = field(default_factory=ConceptConfig)
    concept_categories: ConceptCategoryConfig = field(default_factory=ConceptCategoryConfig)


@dataclass(frozen=True)
class NetworkConfig:
    """Layer 3 (temporal network construction).

    Co-occurrence is grounded in the observation structure, never a tunable time
    window (temporal proximity is a question for the analysis). Three structural
    sources can be toggled independently.
    """

    co_occurrence_same_segment: bool = True  # entities in one observation unit
    co_occurrence_overlap: bool = True       # source segments overlap in time
    co_occurrence_scene: bool = True         # entities within one vision shot (clique)
    casefold: bool = True                    # identifier normalization: casefold
    emit_appearances: bool = True            # unary "appears" interactions
    # Alias resolution (Layer 3, descriptive): merge surface-form variants of the
    # same entity. Exact-normalized merge is always on; fuzzy adds edit-distance
    # merging, gated by type so it never merges across incompatible categories.
    resolve_fuzzy: bool = True
    fuzzy_threshold: float = 0.90            # min SequenceMatcher ratio to merge
    fuzzy_min_length: int = 4                # shorter keys must match exactly (protects "A2")
    # Name types allowed to fuzzy-merge across (besides identical type). None by
    # default: names merge only within one type (Person never with Organization).
    type_merge_groups: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class TimeConfig:
    """Absolute time base. `anchor` is a timezone-aware ISO 8601 instant marking
    offset 0. Default None = zero origin, so timestamps are pure offsets from 0
    and become absolute (anchor + offset) as soon as an anchor is set, with no
    reprocessing. Observation units always store offset seconds; the anchor lives
    in the SourceRecord."""

    anchor: str | None = None


@dataclass(frozen=True)
class Config:
    """Top-level run configuration. `mode` selects the local vs production family."""

    mode: str = "local"
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    time: TimeConfig = field(default_factory=TimeConfig)

    @staticmethod
    def local_default() -> "Config":
        return Config()

    def to_dict(self) -> dict:
        return asdict(self)
