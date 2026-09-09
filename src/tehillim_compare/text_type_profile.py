"""Builds a psalm-by-text-type tag-count matrix from BHSA's `txt` register codes."""

from __future__ import annotations

from tehillim_compare.corpus import Psalm, PsalmWord
from tehillim_compare.features import FeatureInfo, FeatureMatrix, build_tag_count_feature_matrix

#: BHSA text-type single-character codes -> human-readable labels.
_TEXT_TYPE_CHAR_LABELS: dict[str, str] = {
    "?": "Unknown",
    "N": "Narrative",
    "D": "Discursive",
    "Q": "Quotation",
}


def text_type_words(words: tuple[PsalmWord, ...]) -> list[PsalmWord]:
    """Return the subset of `words` carrying a text-type tag."""
    return [w for w in words if w.text_type]


def _label(tag: str) -> str:
    if len(tag) == 1:
        return _TEXT_TYPE_CHAR_LABELS.get(tag, tag)
    return " within ".join(_TEXT_TYPE_CHAR_LABELS.get(c, c) for c in tag)


def build_text_type_feature_matrix(psalms: list[Psalm]) -> FeatureMatrix:
    """Build a psalm x text-type tag-count matrix."""

    def tags_of(word: PsalmWord) -> list[tuple[str, FeatureInfo]]:
        """The tag this word contributes, described the way the profile displays it."""
        tag = word.text_type
        return [(tag, FeatureInfo(label=tag, description=_label(tag), category="text-type"))]

    return build_tag_count_feature_matrix(psalms, text_type_words, tags_of)
