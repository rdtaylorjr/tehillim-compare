"""Builds a psalm x state tag-count FeatureMatrix from BHSA's `st` (construct/absolute)."""

from __future__ import annotations

from tehillim_compare.corpus import Psalm, PsalmWord
from tehillim_compare.features import FeatureInfo, FeatureMatrix, build_tag_count_feature_matrix

#: BHSA state codes -> human-readable labels.
_STATE_LABELS: dict[str, str] = {
    "a": "Absolute",
    "c": "Construct",
    "e": "Emphatic",
}


def state_words(words: tuple[PsalmWord, ...]) -> list[PsalmWord]:
    """Return the subset of `words` carrying a nominal state."""
    return [w for w in words if w.state]


def build_nominal_state_feature_matrix(psalms: list[Psalm]) -> FeatureMatrix:
    """Build a psalm x nominal-state tag-count matrix."""

    def tags_of(word: PsalmWord) -> list[tuple[str, FeatureInfo]]:
        """The tag this word contributes, described the way the profile displays it."""
        tag = word.state
        label = _STATE_LABELS.get(tag, tag)
        return [
            (
                tag,
                FeatureInfo(
                    label=label,
                    description=f"{label} state",
                    category="nominal-state",
                ),
            )
        ]

    return build_tag_count_feature_matrix(psalms, state_words, tags_of)
