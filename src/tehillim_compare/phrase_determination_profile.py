"""Builds a psalm-by-determination tag-count matrix from BHSA's `det`."""

from __future__ import annotations

from tehillim_compare.corpus import Psalm, PsalmWord
from tehillim_compare.features import FeatureInfo, FeatureMatrix, build_tag_count_feature_matrix

#: BHSA determination codes -> human-readable labels.
_DETERMINATION_LABELS: dict[str, str] = {
    "det": "Determined",
    "und": "Undetermined",
}


def determination_words(words: tuple[PsalmWord, ...]) -> list[PsalmWord]:
    """Return the subset of `words` whose enclosing phrase carries a determination tag."""
    return [w for w in words if w.phrase_determination]


def build_phrase_determination_feature_matrix(psalms: list[Psalm]) -> FeatureMatrix:
    """Build a psalm x phrase-determination tag-count matrix."""

    def tags_of(word: PsalmWord) -> list[tuple[str, FeatureInfo]]:
        """The tag this word contributes, described the way the profile displays it."""
        tag = word.phrase_determination
        label = _DETERMINATION_LABELS.get(tag, tag)
        return [
            (
                tag,
                FeatureInfo(
                    label=label,
                    description=f"{label} (phrase determination)",
                    category="phrase-determination",
                ),
            )
        ]

    return build_tag_count_feature_matrix(psalms, determination_words, tags_of)
