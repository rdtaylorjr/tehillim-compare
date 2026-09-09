"""Similarity metrics over psalm feature matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

from tehillim_compare.features import FeatureMatrix


def tfidf_weights(features: FeatureMatrix) -> np.ndarray:
    """TF-IDF weights each psalm's term counts."""
    #: No column means no psalm carries this tag at all, which sklearn reports far from the cause.
    if not features.terms:
        raise ValueError("cannot weight a feature matrix with no terms: no psalm carries this tag")
    dense: np.ndarray = TfidfTransformer().fit_transform(features.counts).toarray()
    return dense


@dataclass(frozen=True, slots=True)
class SimilarityResult:
    """One similarity method's psalm-by-psalm matrix, with the psalms it covers."""

    method: str
    description: str
    psalm_numbers: tuple[int, ...]
    matrix: np.ndarray  # shape (n, n), symmetric, diagonal == 1.0


class SimilarityMethod(Protocol):
    """A named, documented way to turn a FeatureMatrix into psalm similarities."""

    @property
    def name(self) -> str:
        """The name this method is reported under."""

    @property
    def description(self) -> str:
        """What the method computes, for whoever reads the output."""

    def compute(self, features: FeatureMatrix) -> SimilarityResult:
        """Psalm-by-psalm similarity from this feature matrix."""
        ...


@dataclass(frozen=True, slots=True)
class TfidfCosineSimilarity:
    """Cosine similarity between TF-IDF-weighted term vectors."""

    name: str
    description: str

    def compute(self, features: FeatureMatrix) -> SimilarityResult:
        """Psalm-by-psalm similarity from this feature matrix."""
        weights = tfidf_weights(features)
        matrix = cosine_similarity(weights)
        np.fill_diagonal(matrix, 1.0)
        return SimilarityResult(
            method=self.name,
            description=self.description,
            psalm_numbers=features.psalm_numbers,
            matrix=matrix,
        )
