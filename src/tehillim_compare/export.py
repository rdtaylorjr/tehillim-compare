"""Serializes psalms and similarity results into similarity.json."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from tehillim_compare.corpus import Psalm
from tehillim_compare.features import FeatureMatrix
from tehillim_compare.similarity import SimilarityResult

#: Number of top similar psalms to precompute per psalm.
TOP_SIMILAR_COUNT = 12
#: Number of top distinctive terms to include per psalm.
TOP_TERMS_COUNT = 8
#: Number of shared terms used to explain each similar-psalm pairing.
SHARED_TERMS_COUNT = 6

CORPUS_SOURCE = {"name": "ETCBC/BHSA", "version": "2021"}


@dataclass(frozen=True, slots=True)
class MethodComputation:
    """Everything needed to export one similarity method's results."""

    features: FeatureMatrix
    weights: np.ndarray
    result: SimilarityResult


def build_similarity_payload(
    psalms: list[Psalm],
    computations: list[MethodComputation],
    default_method: str,
) -> dict[str, Any]:
    """Assemble the full JSON-serializable payload for the frontend."""
    return {
        "generatedAt": datetime.now(UTC).isoformat(timespec="seconds"),
        "corpus": CORPUS_SOURCE,
        "psalms": [psalm_core(psalm) for psalm in psalms],
        "methods": [_method_payload(psalms, computation) for computation in computations],
        "defaultMethod": default_method,
    }


def psalm_core(psalm: Psalm) -> dict[str, Any]:
    """Method-independent psalm facts, exported once and shared."""
    return {
        "number": psalm.number,
        "verseCount": psalm.verse_count,
        "wordCount": len(psalm.words),
        "incipit": psalm.incipit,
    }


def _method_payload(psalms: list[Psalm], computation: MethodComputation) -> dict[str, Any]:
    features, weights, result = computation.features, computation.weights, computation.result

    def term_payload(col: int, score: float) -> dict[str, Any]:
        """One term's weight and rank, as the payload the site reads."""
        term = features.terms[col]
        info = features.term_info[term]
        return {
            "label": info.label,
            "description": info.description,
            "category": info.category,
            "score": round(float(score), 4),
        }

    def top_term_columns(row: int) -> list[int]:
        """The highest-weighted terms per psalm, in display order."""
        order = np.argsort(weights[row])[::-1]
        return [int(i) for i in order[:TOP_TERMS_COUNT] if weights[row, i] > 0]

    psalm_stats = []
    for row, psalm in enumerate(psalms):
        psalm_stats.append(
            {
                "number": psalm.number,
                "termCount": int(features.counts[row].sum()),
                "uniqueTermCount": int(np.count_nonzero(features.counts[row])),
                "topTerms": [term_payload(col, weights[row, col]) for col in top_term_columns(row)],
            }
        )

    similar_by_psalm: dict[str, list[dict[str, Any]]] = {}
    for row, psalm in enumerate(psalms):
        order = np.argsort(result.matrix[row])[::-1]
        ranked_rows = [int(r) for r in order if r != row][:TOP_SIMILAR_COUNT]

        entries = []
        for other_row in ranked_rows:
            shared_cols = np.flatnonzero(
                (features.counts[row] > 0) & (features.counts[other_row] > 0)
            )
            shared_strength = np.minimum(weights[row, shared_cols], weights[other_row, shared_cols])
            best = np.argsort(shared_strength)[::-1][:SHARED_TERMS_COUNT]
            entries.append(
                {
                    "psalm": psalms[other_row].number,
                    "score": round(float(result.matrix[row, other_row]), 4),
                    "sharedTerms": [
                        term_payload(int(shared_cols[i]), shared_strength[i]) for i in best
                    ],
                }
            )
        similar_by_psalm[str(psalm.number)] = entries

    return {
        "id": result.method,
        "description": result.description,
        "psalmNumbers": list(result.psalm_numbers),
        "psalmStats": psalm_stats,
        "similar": similar_by_psalm,
        "matrix": [[round(float(v), 3) for v in row] for row in result.matrix],
    }
