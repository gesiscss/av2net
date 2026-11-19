"""Stanza engine: tokenizer, part-of-speech tagger, lemmatizer and dependency
parser per language.

Stanza (Apache-2.0) has models for over 100 languages, trained on Universal
Dependencies treebanks, so every language gets the same tag set (NOUN, PROPN,
VERB, ...) and the same dependency relations (nsubj, obj, amod, ...). One
pipeline is built per language, on first use, from the ISO code a text span
records. The installed Stanza version fixes the model files (its resources
version), and the models run on CPU without sampling, so the same text always
gives the same tags, lemmas and dependencies.

Language codes are matched to Stanza's in two steps: the code itself, then its
part before a hyphen ("zh-cn" -> "zh"), following Stanza's own aliases
("zh" -> "zh-hans", "no" -> "nb"). A language counts as supported only if
Stanza has both a tagger and a lemmatizer for it; the dependency parser is used
where the language has one.

`parse` reads many texts of one language in one batch and is what run_parse.py
uses; `tag_tokens` tags a single text.
"""

from __future__ import annotations

import warnings
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Sequence

from ...config import StanzaConfig
from ..interfaces import Token
from ..parse import MultiWordToken, ParsedSentence, ParsedWord, SentenceParser

_REQUIRED = ("tokenize", "pos", "lemma")


def convert_document(doc: Any) -> tuple[ParsedSentence, ...]:
    """A Stanza document as parsed sentences. Words inside a multi-word token
    take the token's character offsets."""
    sentences = []
    for sent in doc.sentences:
        words, multiword = [], []
        for token in sent.tokens:
            ids = token.id if isinstance(token.id, tuple) else (token.id,)
            if len(token.words) > 1:
                multiword.append(MultiWordToken(int(ids[0]), int(ids[-1]), token.text,
                                                int(token.start_char), int(token.end_char)))
            for word in token.words:
                start = word.start_char if word.start_char is not None else token.start_char
                end = word.end_char if word.end_char is not None else token.end_char
                words.append(ParsedWord(
                    id=int(word.id), text=word.text, lemma=word.lemma, upos=word.upos or "",
                    xpos=word.xpos, feats=word.feats,
                    head=None if word.head is None else int(word.head), deprel=word.deprel,
                    start=int(start), end=int(end)))
        sentences.append(ParsedSentence(text=sent.text, words=tuple(words),
                                        multiword=tuple(multiword)))
    return tuple(sentences)


class StanzaEngine(SentenceParser):
    def __init__(self, cfg: StanzaConfig):
        self._cfg = cfg
        self._resources: dict[str, Any] | None = None
        self._pipelines: dict[str, Any] = {}
        self._resolved: dict[str | None, str | None] = {}
        try:
            self._version = version("stanza")
        except PackageNotFoundError:
            self._version = "unknown"

    @property
    def model_id(self) -> str:
        return f"stanza{self._version}:{self._cfg.package}"

    @property
    def loaded(self) -> bool:
        return bool(self._pipelines)

    def _resource_table(self) -> dict[str, Any]:
        """Stanza's resources.json: which models exist per language. Downloaded
        once, for the installed Stanza version."""
        if self._resources is None:
            import stanza.resources.common as common  # imports torch: on first use only

            kwargs = {"model_dir": self._cfg.model_dir} if self._cfg.model_dir else {}
            try:
                self._resources = common.load_resources_json(**kwargs)
            except FileNotFoundError:
                common.download_resources_json(**kwargs)
                self._resources = common.load_resources_json(**kwargs)
        return self._resources

    def resources_version(self) -> str:
        import stanza.resources.common as common

        return common.DEFAULT_RESOURCES_VERSION

    def resolve(self, language: str | None) -> str | None:
        """Stanza's code for `language`, or None if it has no tagger and lemmatizer."""
        if language not in self._resolved:
            self._resolved[language] = self._resolve(language)
        return self._resolved[language]

    def _resolve(self, language: str | None) -> str | None:
        if not language:
            return None
        resources = self._resource_table()
        code = language.strip().lower()
        for candidate in dict.fromkeys((code, code.split("-")[0])):
            entry = resources.get(candidate)
            if isinstance(entry, dict) and "alias" in entry:
                candidate, entry = entry["alias"], resources.get(entry["alias"])
            if not isinstance(entry, dict):
                continue
            available = entry.get("default_processors", {})
            if all(p in available for p in _REQUIRED):
                return candidate
        return None

    def supports(self, language: str | None) -> bool:
        return self.resolve(language) is not None

    def processors(self, language: str | None) -> dict[str, str]:
        """The processors used for `language` and Stanza's default model
        package for each (for provenance). Processors a language lacks (most
        languages have no multi-word-token expander) are left out."""
        code = self.resolve(language)
        if code is None:
            return {}
        available = self._resource_table()[code].get("default_processors", {})
        return {p: str(available[p]) for p in self._cfg.processors if p in available}

    def language_models(self) -> dict[str, dict[str, str]]:
        """Processors and model packages of every language loaded so far."""
        return {code: self.processors(code) for code in sorted(self._pipelines)}

    def _pipeline(self, code: str) -> Any:
        if code not in self._pipelines:
            import stanza

            with warnings.catch_warnings():
                # Download notices from huggingface_hub (symlinks on Windows,
                # unauthenticated requests); they do not affect the models.
                warnings.filterwarnings("ignore", category=UserWarning)
                kwargs = {"model_dir": self._cfg.model_dir} if self._cfg.model_dir else {}
                self._pipelines[code] = stanza.Pipeline(
                    code,
                    processors=",".join(self.processors(code)),
                    package=self._cfg.package,
                    use_gpu=self._cfg.device != "cpu",
                    download_method=stanza.DownloadMethod.REUSE_RESOURCES,
                    logging_level="ERROR",
                    **kwargs,
                )
        return self._pipelines[code]

    def parse(self, texts: Sequence[str], language: str | None) -> list[tuple[ParsedSentence, ...]] | None:
        """Parse several texts of one language in one batch: the sentences of
        each text, in input order (none for an empty text). None if `language`
        has no model."""
        code = self.resolve(language)
        if code is None:
            return None
        out: list[tuple[ParsedSentence, ...]] = [() for _ in texts]
        todo = [i for i, text in enumerate(texts) if text.strip()]
        if todo:
            import stanza

            docs = self._pipeline(code)([stanza.Document([], text=texts[i]) for i in todo])
            for i, doc in zip(todo, docs):
                out[i] = convert_document(doc)
        return out

    def sentences(self, text: str, language: str | None) -> tuple[ParsedSentence, ...] | None:
        parsed = self.parse([text], language)
        return None if parsed is None else parsed[0]

    def tag_tokens(self, text: str, language: str | None) -> list[Token] | None:
        code = self.resolve(language)
        if code is None:
            return None
        if not text.strip():
            return []
        doc = self._pipeline(code)(text)
        tokens: list[Token] = []
        for sentence in doc.sentences:
            for word in sentence.words:
                start = word.start_char if word.start_char is not None else word.parent.start_char
                end = word.end_char if word.end_char is not None else word.parent.end_char
                tokens.append(Token(int(start), int(end), word.text, word.upos or "",
                                    word.lemma or word.text))
        return tokens
