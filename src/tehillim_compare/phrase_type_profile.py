"""Builds a psalm-by-phrase-type tag-count matrix from BHSA's phrase-level `typ`."""

from __future__ import annotations

from tehillim_compare.corpus import Psalm, PsalmWord
from tehillim_compare.features import FeatureInfo, FeatureMatrix, build_tag_count_feature_matrix

#: BHSA phrase-type codes -> human-readable labels.
_PHRASE_TYPE_LABELS: dict[str, str] = {
    "VP": "Verbal Phrase",
    "NP": "Nominal Phrase",
    "PrNP": "Proper-Noun Phrase",
    "AdvP": "Adverbial Phrase",
    "PP": "Prepositional Phrase",
    "CP": "Conjunctive Phrase",
    "PPrP": "Personal Pronoun Phrase",
    "DPrP": "Demonstrative Pronoun Phrase",
    "IPrP": "Interrogative Pronoun Phrase",
    "InjP": "Interjectional Phrase",
    "NegP": "Negative Phrase",
    "InrP": "Interrogative Phrase",
    "AdjP": "Adjective Phrase",
}


def build_phrase_type_feature_matrix(psalms: list[Psalm]) -> FeatureMatrix:
    """Build a psalm x phrase-type tag-count matrix."""

    def tags_of(word: PsalmWord) -> list[tuple[str, FeatureInfo]]:
        """The tag this word contributes, described the way the profile displays it."""
        tag = word.phrase_type
        label = _PHRASE_TYPE_LABELS.get(tag, tag)
        return [
            (
                tag,
                FeatureInfo(
                    label=label,
                    description=f"{label} (phrase type)",
                    category="phrase-type",
                ),
            )
        ]

    return build_tag_count_feature_matrix(psalms, lambda words: words, tags_of)
