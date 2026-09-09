"""Builds a psalm-by-(person, number) tag-count matrix from inflectional and suffix person."""

from __future__ import annotations

from tehillim_compare.corpus import Psalm, PsalmWord
from tehillim_compare.features import FeatureInfo, FeatureMatrix, build_tag_count_feature_matrix

#: BHSA person codes -> human-readable labels.
_PERSON_LABELS: dict[str, str] = {
    "p1": "1st Person",
    "p2": "2nd Person",
    "p3": "3rd Person",
    "unknown": "Person (ambiguous)",
}

#: BHSA number codes -> human-readable labels.
_NUMBER_LABELS: dict[str, str] = {
    "sg": "Singular",
    "pl": "Plural",
    "du": "Dual",
    "unknown": "Number (ambiguous)",
}


def person_words(words: tuple[PsalmWord, ...]) -> list[PsalmWord]:
    """The subset of `words` carrying person marking of either kind."""
    return [w for w in words if w.person or w.suffix_person]


def _label(person: str, number: str, *, is_suffix: bool) -> str:
    person_label = _PERSON_LABELS.get(person, person)
    number_label = _NUMBER_LABELS.get(number or "unknown", number or "unknown")
    suffix = " Suffix" if is_suffix else ""
    return f"{person_label} {number_label}{suffix}"


def build_person_feature_matrix(psalms: list[Psalm]) -> FeatureMatrix:
    """Build a psalm x (person, number) tag-count matrix."""

    def tags_of(word: PsalmWord) -> list[tuple[str, FeatureInfo]]:
        """A word marks its own person and number and, with a suffix, the suffix's."""
        tags: list[tuple[str, FeatureInfo]] = []
        if word.person:
            own = _label(word.person, word.number, is_suffix=False)
            tags.append(
                (
                    f"word.{word.person}.{word.number or 'unknown'}",
                    FeatureInfo(
                        label=own,
                        description=f"{own} marking",
                        category="grammatical-person",
                    ),
                )
            )
        if word.suffix_person:
            suffixed = _label(word.suffix_person, word.suffix_number, is_suffix=True)
            tags.append(
                (
                    f"suffix.{word.suffix_person}.{word.suffix_number or 'unknown'}",
                    FeatureInfo(
                        label=suffixed,
                        description=f"{suffixed} (pronominal suffix)",
                        category="grammatical-person",
                    ),
                )
            )
        return tags

    return build_tag_count_feature_matrix(psalms, person_words, tags_of)
