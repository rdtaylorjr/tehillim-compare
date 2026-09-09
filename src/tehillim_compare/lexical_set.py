"""Builds a psalm-by-lexical-set tag-count matrix from BHSA's `ls` subclassification."""

from __future__ import annotations

from tehillim_compare.corpus import Psalm, PsalmWord
from tehillim_compare.features import FeatureInfo, FeatureMatrix, build_tag_count_feature_matrix

#: BHSA lexical-set codes -> human-readable labels.
_LEXICAL_SET_LABELS: dict[str, str] = {
    "nmdi": "Distributive Noun",
    "nmcp": "Copulative Noun",
    "padv": "Potential Adverb",
    "afad": "Anaphoric Adverb",
    "ppre": "Potential Preposition",
    "cjad": "Conjunctive Adverb",
    "ordn": "Ordinal",
    "vbcp": "Copulative Verb",
    "mult": "Noun of Multitude",
    "focp": "Focus Particle",
    "ques": "Interrogative Particle",
    "gntl": "Gentilic",
    "quot": "Quotation Verb",
    "card": "Cardinal",
}


def lexical_set_words(words: tuple[PsalmWord, ...]) -> list[PsalmWord]:
    """Return the subset of `words` carrying a lexical-set subcategory."""
    return [w for w in words if w.lexical_set]


def build_lexical_set_feature_matrix(psalms: list[Psalm]) -> FeatureMatrix:
    """Build a psalm x lexical-set tag-count matrix."""

    def tags_of(word: PsalmWord) -> list[tuple[str, FeatureInfo]]:
        """The tag this word contributes, described the way the profile displays it."""
        tag = word.lexical_set
        label = _LEXICAL_SET_LABELS.get(tag, tag)
        return [
            (
                tag,
                FeatureInfo(
                    label=label,
                    description=f"{label} ({tag})",
                    category="lexical-set",
                ),
            )
        ]

    return build_tag_count_feature_matrix(psalms, lexical_set_words, tags_of)
