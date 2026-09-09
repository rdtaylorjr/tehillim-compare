"""Builds a psalm x clause-relation tag-count FeatureMatrix from BHSA's `rela` codes."""

from __future__ import annotations

from tehillim_compare.corpus import Psalm, PsalmWord
from tehillim_compare.features import FeatureInfo, FeatureMatrix, build_tag_count_feature_matrix

#: BHSA clause-relation codes -> human-readable labels.
_CLAUSE_RELATION_LABELS: dict[str, str] = {
    "Adju": "Adjunctive Clause",
    "Attr": "Attributive Clause",
    "Cmpl": "Complement Clause",
    "Coor": "Coordinated Clause",
    "Objc": "Object Clause",
    "PrAd": "Predicative Adjunct Clause",
    "PreC": "Predicative Complement Clause",
    "ReVo": "Referral to the Vocative",
    "Resu": "Resumptive Clause",
    "RgRc": "Regens/Rectum Connection",
    "Spec": "Specification Clause",
    "Subj": "Subject Clause",
}


def clause_relation_words(words: tuple[PsalmWord, ...]) -> list[PsalmWord]:
    """Return the subset of `words` whose enclosing clause carries a clause-relation tag."""
    return [w for w in words if w.clause_relation]


def build_clause_relation_feature_matrix(psalms: list[Psalm]) -> FeatureMatrix:
    """Build a psalm x clause-relation tag-count matrix."""

    def tags_of(word: PsalmWord) -> list[tuple[str, FeatureInfo]]:
        """The tag this word contributes, described the way the profile displays it."""
        tag = word.clause_relation
        label = _CLAUSE_RELATION_LABELS.get(tag, tag)
        return [
            (
                tag,
                FeatureInfo(
                    label=label,
                    description=f"{label} (clause relation)",
                    category="clause-relation",
                ),
            )
        ]

    return build_tag_count_feature_matrix(psalms, clause_relation_words, tags_of)
