"""Builds a psalm x phrase-dependent-part-of-speech tag-count FeatureMatrix from BHSA's `pdp`."""

from __future__ import annotations

from tehillim_compare.corpus import Psalm, PsalmWord
from tehillim_compare.features import FeatureInfo, FeatureMatrix, build_tag_count_feature_matrix

#: BHSA phrase-dependent-part-of-speech codes -> human-readable labels.
_PDP_LABELS: dict[str, str] = {
    "art": "Article",
    "verb": "Verb",
    "subs": "Noun",
    "nmpr": "Proper Noun",
    "advb": "Adverb",
    "prep": "Preposition",
    "conj": "Conjunction",
    "prps": "Personal Pronoun",
    "prde": "Demonstrative Pronoun",
    "prin": "Interrogative Pronoun",
    "intj": "Interjection",
    "nega": "Negative Particle",
    "inrg": "Interrogative Particle",
    "adjv": "Adjective",
}


def build_phrase_dependent_pos_feature_matrix(psalms: list[Psalm]) -> FeatureMatrix:
    """Build a psalm x phrase-dependent-part-of-speech tag-count matrix."""

    def tags_of(word: PsalmWord) -> list[tuple[str, FeatureInfo]]:
        """Every word contributes its phrase-dependent part of speech."""
        tag = word.phrase_dependent_pos
        label = _PDP_LABELS.get(tag, tag)
        return [
            (
                tag,
                FeatureInfo(
                    label=label,
                    description=f"{label} (phrase-dependent)",
                    category="phrase-dependent-pos",
                ),
            )
        ]

    return build_tag_count_feature_matrix(psalms, lambda words: words, tags_of)
