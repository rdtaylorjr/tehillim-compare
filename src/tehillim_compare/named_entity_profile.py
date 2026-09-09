"""Builds a psalm-by-name-type tag-count matrix from BHSA's `nametype`."""

from __future__ import annotations

from tehillim_compare.corpus import Psalm, PsalmWord
from tehillim_compare.features import FeatureInfo, FeatureMatrix, build_tag_count_feature_matrix

#: BHSA named-entity-type codes -> human-readable labels.
_NAME_TYPE_LABELS: dict[str, str] = {
    "gens": "People",
    "god": "Deity",
    "mens": "Measurement Unit",
    "pers": "Person",
    "ppde": "Demonstrative Personal Pronoun",
    "topo": "Place",
}


def name_type_words(words: tuple[PsalmWord, ...]) -> list[PsalmWord]:
    """Return the subset of `words` marked as a named entity."""
    return [w for w in words if w.name_type]


def _label(name_type: str) -> str:
    parts = name_type.split(",")
    return " / ".join(_NAME_TYPE_LABELS.get(part, part) for part in parts)


def build_named_entity_feature_matrix(psalms: list[Psalm]) -> FeatureMatrix:
    """Build a psalm x name-type tag-count matrix."""

    def tags_of(word: PsalmWord) -> list[tuple[str, FeatureInfo]]:
        """The tag this word contributes, described the way the profile displays it."""
        tag = word.name_type
        return [
            (
                tag,
                FeatureInfo(
                    label=_label(tag),
                    description=f"Named entity: {_label(tag)}",
                    category="named-entity",
                ),
            )
        ]

    return build_tag_count_feature_matrix(psalms, name_type_words, tags_of)
