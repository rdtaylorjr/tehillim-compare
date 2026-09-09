"""Builds a psalm-by-gender tag-count matrix from inflectional and suffix gender."""

from __future__ import annotations

from tehillim_compare.corpus import Psalm, PsalmWord
from tehillim_compare.features import FeatureInfo, FeatureMatrix, build_tag_count_feature_matrix

#: BHSA gender codes -> human-readable labels.
_GENDER_LABELS: dict[str, str] = {
    "m": "Masculine",
    "f": "Feminine",
    "unknown": "Gender (ambiguous)",
}


def gender_words(words: tuple[PsalmWord, ...]) -> list[PsalmWord]:
    """The subset of `words` carrying gender marking of either kind."""
    return [w for w in words if w.gender or w.suffix_gender]


def _label(gender: str, *, is_suffix: bool) -> str:
    gender_label = _GENDER_LABELS.get(gender, gender)
    suffix = " Suffix" if is_suffix else ""
    return f"{gender_label}{suffix}"


def build_gender_feature_matrix(psalms: list[Psalm]) -> FeatureMatrix:
    """Build a psalm x gender tag-count matrix."""

    def tags_of(word: PsalmWord) -> list[tuple[str, FeatureInfo]]:
        """A word marks its own gender and, when it carries a pronominal suffix, the suffix's."""
        tags: list[tuple[str, FeatureInfo]] = []
        if word.gender:
            own = _label(word.gender, is_suffix=False)
            tags.append(
                (
                    f"word.{word.gender}",
                    FeatureInfo(
                        label=own, description=f"{own} marking", category="grammatical-gender"
                    ),
                )
            )
        if word.suffix_gender:
            suffixed = _label(word.suffix_gender, is_suffix=True)
            tags.append(
                (
                    f"suffix.{word.suffix_gender}",
                    FeatureInfo(
                        label=suffixed,
                        description=f"{suffixed} (pronominal suffix)",
                        category="grammatical-gender",
                    ),
                )
            )
        return tags

    return build_tag_count_feature_matrix(psalms, gender_words, tags_of)
