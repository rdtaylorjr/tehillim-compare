"""`FeatureMatrix`: a generic psalm x vocabulary count matrix."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from tehillim_compare.corpus import Psalm, PsalmWord

#: Part-of-speech codes treated as lexical content words.
CONTENT_POS = frozenset({"subs", "verb", "nmpr", "adjv", "advb", "intj"})


@dataclass(frozen=True, slots=True)
class FeatureInfo:
    """Display metadata for one column of a FeatureMatrix."""

    label: str
    """Short display form (a Hebrew lemma, a human-readable tag name, ...)."""

    description: str
    """Longer explanation (an English gloss, a tag's full meaning, ...)."""

    category: str
    """Grouping tag (part of speech, feature family, ...)."""


@dataclass(frozen=True, slots=True)
class FeatureMatrix:
    """A dense psalm x term count matrix plus display metadata for each term."""

    psalm_numbers: tuple[int, ...]
    terms: tuple[str, ...]
    counts: np.ndarray  # shape (n_psalms, n_terms), dtype int32
    term_info: dict[str, FeatureInfo]


def content_words(words: tuple[PsalmWord, ...]) -> list[PsalmWord]:
    """Return the subset of `words` that count as lexical content words."""
    return [w for w in words if w.part_of_speech in CONTENT_POS]


def build_lexical_feature_matrix(psalms: list[Psalm]) -> FeatureMatrix:
    """Build a psalm x lexeme term-count matrix over content-word lexemes."""

    def tags_of(word: PsalmWord) -> list[tuple[str, FeatureInfo]]:
        """A content word contributes its lexeme, labelled by its lemma and gloss."""
        return [
            (
                word.lexeme,
                FeatureInfo(
                    label=word.lemma,
                    description=word.gloss,
                    category=word.part_of_speech,
                ),
            )
        ]

    return build_tag_count_feature_matrix(psalms, content_words, tags_of)


def build_tag_count_feature_matrix(
    psalms: list[Psalm],
    select: Callable[[tuple[PsalmWord, ...]], Sequence[PsalmWord]],
    tags_of: Callable[[PsalmWord], Iterable[tuple[str, FeatureInfo]]],
) -> FeatureMatrix:
    """Psalm x tag count matrix: counts every tag a selected word carries, describing each once."""
    term_info: dict[str, FeatureInfo] = {}
    per_psalm_counts: list[dict[str, int]] = []

    for psalm in psalms:
        counts: dict[str, int] = {}
        for word in select(psalm.words):
            for tag, info in tags_of(word):
                counts[tag] = counts.get(tag, 0) + 1
                #: setdefault, so the first word carrying a tag is the one that describes it.
                term_info.setdefault(tag, info)
        per_psalm_counts.append(counts)

    return assemble_feature_matrix(psalms, per_psalm_counts, term_info)


def assemble_feature_matrix(
    psalms: list[Psalm],
    per_psalm_counts: list[dict[str, int]],
    term_info: dict[str, FeatureInfo],
) -> FeatureMatrix:
    """Assemble a FeatureMatrix from per-psalm term counts."""
    terms = tuple(sorted(term_info))
    term_column = {term: i for i, term in enumerate(terms)}

    matrix = np.zeros((len(psalms), len(terms)), dtype=np.int32)
    for row, counts in enumerate(per_psalm_counts):
        for term, count in counts.items():
            matrix[row, term_column[term]] = count

    return FeatureMatrix(
        psalm_numbers=tuple(p.number for p in psalms),
        terms=terms,
        counts=matrix,
        term_info=term_info,
    )
