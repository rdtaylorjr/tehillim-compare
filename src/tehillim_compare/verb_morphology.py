"""Builds a psalm x (stem, conjugation) tag-count FeatureMatrix."""

from __future__ import annotations

from tehillim_compare.corpus import Psalm, PsalmWord
from tehillim_compare.features import FeatureInfo, FeatureMatrix, build_tag_count_feature_matrix

#: BHSA verb stem (binyan) codes -> human-readable labels.
_STEM_LABELS: dict[str, str] = {
    "qal": "Qal",
    "nif": "Niphal",
    "piel": "Piel",
    "pual": "Pual",
    "hif": "Hiphil",
    "hof": "Hophal",
    "hit": "Hitpael",
    "hsht": "Hishtaphel",
    "etpa": "Etpaal",
    "poel": "Poel",
}

#: BHSA verb conjugation codes (BHSA's `vt`) -> human-readable labels.
_CONJUGATION_LABELS: dict[str, str] = {
    "perf": "Perfect",
    "impf": "Imperfect",
    "wayq": "Wayyiqtol",
    "impv": "Imperative",
    "infc": "Infinitive Construct",
    "infa": "Infinitive Absolute",
    "ptca": "Active Participle",
    "ptcp": "Passive Participle",
}


def verb_words(words: tuple[PsalmWord, ...]) -> list[PsalmWord]:
    """The subset of `words` usable for verb-morphology features."""
    return [w for w in words if w.part_of_speech == "verb" and w.verb_stem and w.verb_mood]


def _tag(word: PsalmWord) -> str:
    return f"{word.verb_stem}.{word.verb_mood}"


def _label(tag: str) -> str:
    stem, mood = tag.split(".", 1)
    return f"{_STEM_LABELS.get(stem, stem)} {_CONJUGATION_LABELS.get(mood, mood)}"


def build_verb_morphology_feature_matrix(psalms: list[Psalm]) -> FeatureMatrix:
    """Build a psalm x (verb stem, mood) tag-count matrix."""

    def tags_of(word: PsalmWord) -> list[tuple[str, FeatureInfo]]:
        """A verb contributes its (stem, mood) pair as one tag."""
        tag = _tag(word)
        return [
            (
                tag,
                FeatureInfo(
                    label=_label(tag),
                    description=f"{_label(tag)} verb form",
                    category="verb-morphology",
                ),
            )
        ]

    return build_tag_count_feature_matrix(psalms, verb_words, tags_of)
