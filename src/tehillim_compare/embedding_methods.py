"""The similarity methods every embeddings representation is scored under, named from its file."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from tehillim_compare.aggregation import mean_pool, soft_alignment
from tehillim_compare.anisotropy_correction import remove_top_principal_components, whiten
from tehillim_compare.half_verse_vectors import HalfVerseMatrix
from tehillim_compare.representation import Representation
from tehillim_compare.similarity import SimilarityResult


@dataclass(frozen=True, slots=True)
class Aggregation:
    """One way of turning a psalm's half-verse rows into a psalm-by-psalm similarity."""

    key: str
    description: str
    compute: Callable[[HalfVerseMatrix], np.ndarray]


@dataclass(frozen=True, slots=True)
class Correction:
    """One anisotropy correction applied to the pooled half-verse rows before soft alignment."""

    key: str
    description: str
    apply: Callable[[np.ndarray], np.ndarray]


MEAN_POOL = Aggregation(
    "mean-pool", "Cosine similarity between mean-pooled half-verse embeddings.", mean_pool
)
SOFT_ALIGNMENT = Aggregation(
    "soft-alignment",
    "Symmetric best-match cosine similarity between two psalms' half-verse embedding sets, "
    "avoiding the single-pooled-vector bottleneck.",
    soft_alignment,
)
AGGREGATIONS: tuple[Aggregation, ...] = (MEAN_POOL, SOFT_ALIGNMENT)

TOP_PC = Correction(
    "top-pc",
    "The pooled corpus is mean-centered and its top principal component removed first.",
    remove_top_principal_components,
)
WHITENED = Correction(
    "whitened",
    "The pooled corpus is mean-centered and whitened to unit covariance first.",
    whiten,
)
CORRECTIONS: tuple[Correction, ...] = (TOP_PC, WHITENED)

#: The contextual Hebrew encoders whose soft alignment is also scored after each correction.
CORRECTED_REPRESENTATIONS: frozenset[str] = frozenset(
    {"alephbert_consonantal", "berel_consonantal", "neodictabert_consonantal"}
)


@dataclass(frozen=True, slots=True)
class EmbeddingMethod:
    """One representation under one aggregation, after an optional correction."""

    representation: Representation
    aggregation: Aggregation
    correction: Correction | None = None

    @property
    def name(self) -> str:
        """The benchmark identifier followed by the aggregation and correction keys."""
        keys = [self.representation.identifier, self.aggregation.key]
        if self.correction is not None:
            keys.append(self.correction.key)
        return "-".join([*keys, "cosine"])

    @property
    def description(self) -> str:
        """The aggregation's prose, the representation's, and the correction's when applied."""
        parts = [self.aggregation.description, self.representation.description]
        if self.correction is not None:
            parts.append(self.correction.description)
        return " ".join(part for part in parts if part)

    def compute(self, matrix: HalfVerseMatrix) -> SimilarityResult:
        """Similarity between psalms, from their half-verse rows."""
        if self.correction is not None:
            if not isinstance(matrix.rows, np.ndarray):
                raise TypeError(f"{self.name}: a correction needs dense rows")
            matrix = HalfVerseMatrix(
                matrix.psalm_numbers,
                matrix.starts,
                matrix.sizes,
                self.correction.apply(matrix.rows),
            )
        return SimilarityResult(
            method=self.name,
            description=self.description,
            domain=self.representation.domain,
            psalm_numbers=matrix.psalm_numbers,
            matrix=self.aggregation.compute(matrix),
            representation=self.representation.identifier,
            aggregation=self.aggregation.key,
            correction=None if self.correction is None else self.correction.key,
        )


def methods_for(representation: Representation) -> list[EmbeddingMethod]:
    """Every method the representation is scored under, corrections only where declared."""
    methods = [EmbeddingMethod(representation, aggregation) for aggregation in AGGREGATIONS]
    if representation.identifier in CORRECTED_REPRESENTATIONS:
        methods.extend(
            EmbeddingMethod(representation, SOFT_ALIGNMENT, correction)
            for correction in CORRECTIONS
        )
    return methods
