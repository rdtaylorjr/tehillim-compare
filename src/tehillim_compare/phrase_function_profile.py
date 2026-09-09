"""Builds a psalm x phrase-function tag-count FeatureMatrix from BHSA's phrase-level `function`."""

from __future__ import annotations

from tehillim_compare.corpus import Psalm, PsalmWord
from tehillim_compare.features import FeatureInfo, FeatureMatrix, build_tag_count_feature_matrix

#: BHSA phrase-function codes -> human-readable labels.
_PHRASE_FUNCTION_LABELS: dict[str, str] = {
    "Adju": "Adjunct",
    "Cmpl": "Complement",
    "Conj": "Conjunction",
    "EPPr": "Enclitic Personal Pronoun",
    "ExsS": "Existence with Subject Suffix",
    "Exst": "Existence",
    "Frnt": "Fronted Element",
    "Intj": "Interjection",
    "IntS": "Interjection with Subject Suffix",
    "Loca": "Locative",
    "Modi": "Modifier",
    "ModS": "Modifier with Subject Suffix",
    "NCop": "Negative Copula",
    "NCoS": "Negative Copula with Subject Suffix",
    "Nega": "Negation",
    "Objc": "Object",
    "PrAd": "Predicative Adjunct",
    "PrcS": "Predicate Complement with Subject Suffix",
    "PreC": "Predicate Complement",
    "Pred": "Predicate",
    "PreO": "Predicate with Object Suffix",
    "PreS": "Predicate with Subject Suffix",
    "PtcO": "Participle with Object Suffix",
    "Ques": "Question",
    "Rela": "Relative",
    "Subj": "Subject",
    "Supp": "Supplementary Constituent",
    "Time": "Time Reference",
    "Unkn": "Unknown",
    "Voct": "Vocative",
}


def build_phrase_function_feature_matrix(psalms: list[Psalm]) -> FeatureMatrix:
    """Build a psalm x phrase-function tag-count matrix."""

    def tags_of(word: PsalmWord) -> list[tuple[str, FeatureInfo]]:
        """The tag this word contributes, described the way the profile displays it."""
        tag = word.phrase_function
        label = _PHRASE_FUNCTION_LABELS.get(tag, tag)
        return [
            (
                tag,
                FeatureInfo(
                    label=label,
                    description=f"{label} (phrase function)",
                    category="phrase-function",
                ),
            )
        ]

    return build_tag_count_feature_matrix(psalms, lambda words: words, tags_of)
