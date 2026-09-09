"""Builds a psalm-by-clause-kind tag-count matrix from BHSA's coarse 3-way `kind`."""

from __future__ import annotations

from tehillim_compare.corpus import Psalm, PsalmWord
from tehillim_compare.features import FeatureInfo, FeatureMatrix, build_tag_count_feature_matrix

#: BHSA clause-kind codes -> human-readable labels.
_CLAUSE_KIND_LABELS: dict[str, str] = {
    "VC": "Verbal Clause",
    "NC": "Nominal Clause",
    "WP": "Clause Without Predication",
}


def clause_kind_words(words: tuple[PsalmWord, ...]) -> list[PsalmWord]:
    """Return the subset of `words` carrying a clause-kind tag."""
    return [w for w in words if w.clause_kind]


def build_clause_kind_feature_matrix(psalms: list[Psalm]) -> FeatureMatrix:
    """Build a psalm x clause-kind tag-count matrix."""

    def tags_of(word: PsalmWord) -> list[tuple[str, FeatureInfo]]:
        """The tag this word contributes, described the way the profile displays it."""
        tag = word.clause_kind
        label = _CLAUSE_KIND_LABELS.get(tag, tag)
        return [
            (
                tag,
                FeatureInfo(
                    label=label, description=f"{label} (clause kind)", category="clause-kind"
                ),
            )
        ]

    return build_tag_count_feature_matrix(psalms, clause_kind_words, tags_of)
