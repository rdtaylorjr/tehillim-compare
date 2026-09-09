"""Command-line entrypoint: extract psalms and compute every similarity method."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from tehillim_compare.anisotropy_correction import (
    remove_top_principal_components,
    whiten,
)
from tehillim_compare.atomic_write import (
    write_json,
)
from tehillim_compare.build_cache import (
    fingerprint_embeddings,
    load_cached_similarity,
    write_cached_similarity,
)
from tehillim_compare.clause_relation_profile import (
    build_clause_relation_feature_matrix,
)
from tehillim_compare.clause_type_profile import (
    build_clause_type_feature_matrix,
)
from tehillim_compare.corpus import (
    DEFAULT_CHECKOUT,
    Corpus,
    Psalm,
)
from tehillim_compare.dataset import (
    ANALYSIS_COMPARE,
    ui_path,
    write_compare_dataset,
)
from tehillim_compare.export import (
    MethodComputation,
    build_similarity_payload,
)
from tehillim_compare.features import (
    FeatureMatrix,
    build_lexical_feature_matrix,
)
from tehillim_compare.lexical_set import (
    build_lexical_set_feature_matrix,
)
from tehillim_compare.methods import (
    CLAUSE_RELATION_SIMILARITY,
    CLAUSE_TYPE_SIMILARITY,
    LEXICAL_SET_SIMILARITY,
    LEXICAL_SIMILARITY,
    NAMED_ENTITY_IDENTITY_SIMILARITY,
    NAMED_ENTITY_SIMILARITY,
    PERSON_PROFILE_SIMILARITY,
    ROOT_SIMILARITY,
    TEXT_TYPE_SIMILARITY,
    VERB_MORPHOLOGY_SIMILARITY,
    VERB_SENSE_SIMILARITY,
)
from tehillim_compare.named_entity_identity import (
    build_named_entity_identity_feature_matrix,
)
from tehillim_compare.named_entity_profile import (
    build_named_entity_feature_matrix,
)
from tehillim_compare.parallel import (
    map_in_pool,
)
from tehillim_compare.person_profile import (
    build_person_feature_matrix,
)
from tehillim_compare.root_similarity import (
    build_root_feature_matrix,
)
from tehillim_compare.semantic_embedding import (
    ALEPHBERT_MEAN_POOL_SIMILARITY,
    ALEPHBERT_SOFT_ALIGNMENT_SIMILARITY,
    ALEPHBERT_SOFT_ALIGNMENT_TOP_PC_SIMILARITY,
    ALEPHBERT_SOFT_ALIGNMENT_WHITENED_SIMILARITY,
    BEREL_MEAN_POOL_SIMILARITY,
    BEREL_SOFT_ALIGNMENT_SIMILARITY,
    BEREL_SOFT_ALIGNMENT_TOP_PC_SIMILARITY,
    BEREL_SOFT_ALIGNMENT_WHITENED_SIMILARITY,
    BGE_M3_MEAN_POOL_CONSONANTAL_SIMILARITY,
    BGE_M3_MEAN_POOL_SIMILARITY,
    BGE_M3_MEAN_POOL_VOCALIZED_SIMILARITY,
    BGE_M3_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY,
    BGE_M3_SOFT_ALIGNMENT_SIMILARITY,
    BGE_M3_SOFT_ALIGNMENT_VOCALIZED_SIMILARITY,
    BGE_MULTILINGUAL_GEMMA2_MEAN_POOL_CONSONANTAL_SIMILARITY,
    BGE_MULTILINGUAL_GEMMA2_MEAN_POOL_SIMILARITY,
    BGE_MULTILINGUAL_GEMMA2_MEAN_POOL_VOCALIZED_SIMILARITY,
    BGE_MULTILINGUAL_GEMMA2_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY,
    BGE_MULTILINGUAL_GEMMA2_SOFT_ALIGNMENT_SIMILARITY,
    BGE_MULTILINGUAL_GEMMA2_SOFT_ALIGNMENT_VOCALIZED_SIMILARITY,
    COHERE_MEAN_POOL_CONSONANTAL_SIMILARITY,
    COHERE_MEAN_POOL_SIMILARITY,
    COHERE_MEAN_POOL_VOCALIZED_SIMILARITY,
    COHERE_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY,
    COHERE_SOFT_ALIGNMENT_SIMILARITY,
    COHERE_SOFT_ALIGNMENT_VOCALIZED_SIMILARITY,
    GEMINI_MEAN_POOL_CONSONANTAL_SIMILARITY,
    GEMINI_MEAN_POOL_SIMILARITY,
    GEMINI_MEAN_POOL_VOCALIZED_SIMILARITY,
    GEMINI_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY,
    GEMINI_SOFT_ALIGNMENT_SIMILARITY,
    GEMINI_SOFT_ALIGNMENT_VOCALIZED_SIMILARITY,
    GTE_MULTILINGUAL_MEAN_POOL_CONSONANTAL_SIMILARITY,
    GTE_MULTILINGUAL_MEAN_POOL_SIMILARITY,
    GTE_MULTILINGUAL_MEAN_POOL_VOCALIZED_SIMILARITY,
    GTE_MULTILINGUAL_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY,
    GTE_MULTILINGUAL_SOFT_ALIGNMENT_SIMILARITY,
    GTE_MULTILINGUAL_SOFT_ALIGNMENT_VOCALIZED_SIMILARITY,
    KALM_EMBEDDING_MEAN_POOL_CONSONANTAL_SIMILARITY,
    KALM_EMBEDDING_MEAN_POOL_SIMILARITY,
    KALM_EMBEDDING_MEAN_POOL_VOCALIZED_SIMILARITY,
    KALM_EMBEDDING_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY,
    KALM_EMBEDDING_SOFT_ALIGNMENT_SIMILARITY,
    KALM_EMBEDDING_SOFT_ALIGNMENT_VOCALIZED_SIMILARITY,
    LLAMA_EMBED_NEMOTRON_MEAN_POOL_CONSONANTAL_SIMILARITY,
    LLAMA_EMBED_NEMOTRON_MEAN_POOL_SIMILARITY,
    LLAMA_EMBED_NEMOTRON_MEAN_POOL_VOCALIZED_SIMILARITY,
    LLAMA_EMBED_NEMOTRON_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY,
    LLAMA_EMBED_NEMOTRON_SOFT_ALIGNMENT_SIMILARITY,
    LLAMA_EMBED_NEMOTRON_SOFT_ALIGNMENT_VOCALIZED_SIMILARITY,
    ME5_LARGE_INSTRUCT_MEAN_POOL_CONSONANTAL_SIMILARITY,
    ME5_LARGE_INSTRUCT_MEAN_POOL_SIMILARITY,
    ME5_LARGE_INSTRUCT_MEAN_POOL_VOCALIZED_SIMILARITY,
    ME5_LARGE_INSTRUCT_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY,
    ME5_LARGE_INSTRUCT_SOFT_ALIGNMENT_SIMILARITY,
    ME5_LARGE_INSTRUCT_SOFT_ALIGNMENT_VOCALIZED_SIMILARITY,
    MIQRABERT_MEAN_POOL_SIMILARITY,
    MIQRABERT_SOFT_ALIGNMENT_SIMILARITY,
    NEODICTABERT_MEAN_POOL_SIMILARITY,
    NEODICTABERT_SOFT_ALIGNMENT_SIMILARITY,
    NEODICTABERT_SOFT_ALIGNMENT_TOP_PC_SIMILARITY,
    NEODICTABERT_SOFT_ALIGNMENT_WHITENED_SIMILARITY,
    OPENAI_MEAN_POOL_CONSONANTAL_SIMILARITY,
    OPENAI_MEAN_POOL_SIMILARITY,
    OPENAI_MEAN_POOL_VOCALIZED_SIMILARITY,
    OPENAI_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY,
    OPENAI_SOFT_ALIGNMENT_SIMILARITY,
    OPENAI_SOFT_ALIGNMENT_VOCALIZED_SIMILARITY,
    QWEN3_EMBEDDING_MEAN_POOL_CONSONANTAL_SIMILARITY,
    QWEN3_EMBEDDING_MEAN_POOL_SIMILARITY,
    QWEN3_EMBEDDING_MEAN_POOL_VOCALIZED_SIMILARITY,
    QWEN3_EMBEDDING_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY,
    QWEN3_EMBEDDING_SOFT_ALIGNMENT_SIMILARITY,
    QWEN3_EMBEDDING_SOFT_ALIGNMENT_VOCALIZED_SIMILARITY,
    VOYAGE_MEAN_POOL_CONSONANTAL_SIMILARITY,
    VOYAGE_MEAN_POOL_SIMILARITY,
    VOYAGE_MEAN_POOL_VOCALIZED_SIMILARITY,
    VOYAGE_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY,
    VOYAGE_SOFT_ALIGNMENT_SIMILARITY,
    VOYAGE_SOFT_ALIGNMENT_VOCALIZED_SIMILARITY,
    MeanPoolEmbeddingSimilarity,
    SoftAlignmentEmbeddingSimilarity,
)
from tehillim_compare.semantic_embedding_loader import (
    load_semantic_embeddings,
    require_every_semantic_feature,
)
from tehillim_compare.similarity import (
    SimilarityMethod,
    SimilarityResult,
    tfidf_weights,
)
from tehillim_compare.text_type_profile import (
    build_text_type_feature_matrix,
)
from tehillim_compare.verb_morphology import (
    build_verb_morphology_feature_matrix,
)
from tehillim_compare.verb_sense_profile import (
    build_verb_sense_feature_matrix,
)

#: repo_root/data/.
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

#: Root of the partitioned dataset, holding analysis=compare and analysis=cluster.
DEFAULT_DATA_ROOT = _DATA_DIR

#: repo_root/data/similarity_cache.
DEFAULT_CLUSTERING_CACHE_DIR = _DATA_DIR / "similarity_cache"

#: The psalter this package is built for, so a short corpus is reported rather than assumed.
EXPECTED_PSALMS = 150


def semantic_method_names() -> frozenset[str]:
    """Every configured semantic method's name, for callers classifying which domain it scores."""
    return frozenset(method.name for _, method in _SEMANTIC_METHODS)


def default_similarity_output(data_root: Path) -> Path:
    """Where the compare payload lands when no explicit path is given."""
    return ui_path(data_root, ANALYSIS_COMPARE, "similarity.json")


def parse_args(
    argv: list[str] | None = None, env: Mapping[str, str] | None = None
) -> argparse.Namespace:
    """Parses the command line this module documents into its options."""
    env = env if env is not None else os.environ
    parser = argparse.ArgumentParser(description=__doc__)
    env_checkout = env.get("TEHILLIM_CHECKOUT")
    parser.add_argument(
        "--checkout",
        default=env_checkout or DEFAULT_CHECKOUT,
        help=(
            "Text-Fabric checkout spec for BHSA plus its etcbc/valence module "
            f"(default: {DEFAULT_CHECKOUT!r})"
        ),
    )
    env_embeddings_dir = env.get("TEHILLIM_EMBEDDINGS_DIR")
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path(env_embeddings_dir) if env_embeddings_dir else None,
        help=(
            "Path to a local tehillim-embeddings checkout's data/type=semantic/ "
            "directory (default: $TEHILLIM_EMBEDDINGS_DIR, required for semantic-"
            "embedding methods)"
        ),
    )
    env_data_root = env.get("TEHILLIM_DATA_DIR")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(env_data_root) if env_data_root else DEFAULT_DATA_ROOT,
        help=(
            "Root of the partitioned output dataset, written as "
            "analysis=<compare|cluster>/domain=<domain>/stage=<stage>/ "
            f"(default: $TEHILLIM_DATA_DIR, else {DEFAULT_DATA_ROOT})"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Compare payload JSON path (default: under --data-root at analysis=compare/stage=ui)",
    )
    env_cache_dir = env.get("TEHILLIM_CLUSTERING_CACHE_DIR")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(env_cache_dir) if env_cache_dir else DEFAULT_CLUSTERING_CACHE_DIR,
        help=(
            "Directory for cached clustering results, keyed by method name "
            f"(default: {DEFAULT_CLUSTERING_CACHE_DIR})"
        ),
    )
    return parser.parse_args(argv)


_METHODS: tuple[tuple[Callable[[list[Psalm]], FeatureMatrix], SimilarityMethod], ...] = (
    (build_lexical_feature_matrix, LEXICAL_SIMILARITY),
    (build_root_feature_matrix, ROOT_SIMILARITY),
    (build_named_entity_identity_feature_matrix, NAMED_ENTITY_IDENTITY_SIMILARITY),
    (build_verb_morphology_feature_matrix, VERB_MORPHOLOGY_SIMILARITY),
    (build_person_feature_matrix, PERSON_PROFILE_SIMILARITY),
    (build_lexical_set_feature_matrix, LEXICAL_SET_SIMILARITY),
    (build_named_entity_feature_matrix, NAMED_ENTITY_SIMILARITY),
    (build_clause_type_feature_matrix, CLAUSE_TYPE_SIMILARITY),
    (build_text_type_feature_matrix, TEXT_TYPE_SIMILARITY),
    (build_clause_relation_feature_matrix, CLAUSE_RELATION_SIMILARITY),
    (build_verb_sense_feature_matrix, VERB_SENSE_SIMILARITY),
)


#: Internal keys for the anisotropy-corrected embeddings dicts (see anisotropy_correction.py).
_ALEPHBERT_TOP_PC_KEY = "alephbert-top-pc"
_ALEPHBERT_WHITENED_KEY = "alephbert-whitened"
_BEREL_TOP_PC_KEY = "berel-top-pc"
_BEREL_WHITENED_KEY = "berel-whitened"
_NEODICTABERT_TOP_PC_KEY = "neodictabert-top-pc"
_NEODICTABERT_WHITENED_KEY = "neodictabert-whitened"

_SEMANTIC_METHODS: tuple[
    tuple[str, MeanPoolEmbeddingSimilarity | SoftAlignmentEmbeddingSimilarity],
    ...,
] = (
    ("semantic_miqrabert_consonantal", MIQRABERT_MEAN_POOL_SIMILARITY),
    ("semantic_miqrabert_consonantal", MIQRABERT_SOFT_ALIGNMENT_SIMILARITY),
    ("semantic_alephbert_consonantal", ALEPHBERT_MEAN_POOL_SIMILARITY),
    ("semantic_alephbert_consonantal", ALEPHBERT_SOFT_ALIGNMENT_SIMILARITY),
    (_ALEPHBERT_TOP_PC_KEY, ALEPHBERT_SOFT_ALIGNMENT_TOP_PC_SIMILARITY),
    (_ALEPHBERT_WHITENED_KEY, ALEPHBERT_SOFT_ALIGNMENT_WHITENED_SIMILARITY),
    ("semantic_berel_consonantal", BEREL_MEAN_POOL_SIMILARITY),
    ("semantic_berel_consonantal", BEREL_SOFT_ALIGNMENT_SIMILARITY),
    (_BEREL_TOP_PC_KEY, BEREL_SOFT_ALIGNMENT_TOP_PC_SIMILARITY),
    (_BEREL_WHITENED_KEY, BEREL_SOFT_ALIGNMENT_WHITENED_SIMILARITY),
    ("semantic_neodictabert_consonantal", NEODICTABERT_MEAN_POOL_SIMILARITY),
    ("semantic_neodictabert_consonantal", NEODICTABERT_SOFT_ALIGNMENT_SIMILARITY),
    (_NEODICTABERT_TOP_PC_KEY, NEODICTABERT_SOFT_ALIGNMENT_TOP_PC_SIMILARITY),
    (_NEODICTABERT_WHITENED_KEY, NEODICTABERT_SOFT_ALIGNMENT_WHITENED_SIMILARITY),
    ("semantic_bge_multilingual_gemma2_cantillation", BGE_MULTILINGUAL_GEMMA2_MEAN_POOL_SIMILARITY),
    (
        "semantic_bge_multilingual_gemma2_consonantal",
        BGE_MULTILINGUAL_GEMMA2_MEAN_POOL_CONSONANTAL_SIMILARITY,
    ),
    (
        "semantic_bge_multilingual_gemma2_vocalized",
        BGE_MULTILINGUAL_GEMMA2_MEAN_POOL_VOCALIZED_SIMILARITY,
    ),
    (
        "semantic_bge_multilingual_gemma2_cantillation",
        BGE_MULTILINGUAL_GEMMA2_SOFT_ALIGNMENT_SIMILARITY,
    ),
    (
        "semantic_bge_multilingual_gemma2_consonantal",
        BGE_MULTILINGUAL_GEMMA2_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY,
    ),
    (
        "semantic_bge_multilingual_gemma2_vocalized",
        BGE_MULTILINGUAL_GEMMA2_SOFT_ALIGNMENT_VOCALIZED_SIMILARITY,
    ),
    ("semantic_qwen3_embedding_8b_cantillation", QWEN3_EMBEDDING_MEAN_POOL_SIMILARITY),
    ("semantic_qwen3_embedding_8b_consonantal", QWEN3_EMBEDDING_MEAN_POOL_CONSONANTAL_SIMILARITY),
    ("semantic_qwen3_embedding_8b_vocalized", QWEN3_EMBEDDING_MEAN_POOL_VOCALIZED_SIMILARITY),
    ("semantic_qwen3_embedding_8b_cantillation", QWEN3_EMBEDDING_SOFT_ALIGNMENT_SIMILARITY),
    (
        "semantic_qwen3_embedding_8b_consonantal",
        QWEN3_EMBEDDING_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY,
    ),
    ("semantic_qwen3_embedding_8b_vocalized", QWEN3_EMBEDDING_SOFT_ALIGNMENT_VOCALIZED_SIMILARITY),
    ("semantic_kalm_embedding_gemma3_12b_2511_cantillation", KALM_EMBEDDING_MEAN_POOL_SIMILARITY),
    (
        "semantic_kalm_embedding_gemma3_12b_2511_consonantal",
        KALM_EMBEDDING_MEAN_POOL_CONSONANTAL_SIMILARITY,
    ),
    (
        "semantic_kalm_embedding_gemma3_12b_2511_vocalized",
        KALM_EMBEDDING_MEAN_POOL_VOCALIZED_SIMILARITY,
    ),
    (
        "semantic_kalm_embedding_gemma3_12b_2511_cantillation",
        KALM_EMBEDDING_SOFT_ALIGNMENT_SIMILARITY,
    ),
    (
        "semantic_kalm_embedding_gemma3_12b_2511_consonantal",
        KALM_EMBEDDING_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY,
    ),
    (
        "semantic_kalm_embedding_gemma3_12b_2511_vocalized",
        KALM_EMBEDDING_SOFT_ALIGNMENT_VOCALIZED_SIMILARITY,
    ),
    ("semantic_llama_embed_nemotron_8b_cantillation", LLAMA_EMBED_NEMOTRON_MEAN_POOL_SIMILARITY),
    (
        "semantic_llama_embed_nemotron_8b_consonantal",
        LLAMA_EMBED_NEMOTRON_MEAN_POOL_CONSONANTAL_SIMILARITY,
    ),
    (
        "semantic_llama_embed_nemotron_8b_vocalized",
        LLAMA_EMBED_NEMOTRON_MEAN_POOL_VOCALIZED_SIMILARITY,
    ),
    (
        "semantic_llama_embed_nemotron_8b_cantillation",
        LLAMA_EMBED_NEMOTRON_SOFT_ALIGNMENT_SIMILARITY,
    ),
    (
        "semantic_llama_embed_nemotron_8b_consonantal",
        LLAMA_EMBED_NEMOTRON_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY,
    ),
    (
        "semantic_llama_embed_nemotron_8b_vocalized",
        LLAMA_EMBED_NEMOTRON_SOFT_ALIGNMENT_VOCALIZED_SIMILARITY,
    ),
    ("semantic_gemini_embedding_2_cantillation", GEMINI_MEAN_POOL_SIMILARITY),
    ("semantic_gemini_embedding_2_consonantal", GEMINI_MEAN_POOL_CONSONANTAL_SIMILARITY),
    ("semantic_gemini_embedding_2_cantillation", GEMINI_SOFT_ALIGNMENT_SIMILARITY),
    ("semantic_gemini_embedding_2_consonantal", GEMINI_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY),
    ("semantic_gemini_embedding_2_vocalized", GEMINI_MEAN_POOL_VOCALIZED_SIMILARITY),
    ("semantic_gemini_embedding_2_vocalized", GEMINI_SOFT_ALIGNMENT_VOCALIZED_SIMILARITY),
    ("semantic_openai_text_embedding_3_large_cantillation", OPENAI_MEAN_POOL_SIMILARITY),
    ("semantic_openai_text_embedding_3_large_consonantal", OPENAI_MEAN_POOL_CONSONANTAL_SIMILARITY),
    ("semantic_openai_text_embedding_3_large_cantillation", OPENAI_SOFT_ALIGNMENT_SIMILARITY),
    (
        "semantic_openai_text_embedding_3_large_consonantal",
        OPENAI_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY,
    ),
    ("semantic_openai_text_embedding_3_large_vocalized", OPENAI_MEAN_POOL_VOCALIZED_SIMILARITY),
    (
        "semantic_openai_text_embedding_3_large_vocalized",
        OPENAI_SOFT_ALIGNMENT_VOCALIZED_SIMILARITY,
    ),
    ("semantic_cohere_embed_v4_cantillation", COHERE_MEAN_POOL_SIMILARITY),
    ("semantic_cohere_embed_v4_consonantal", COHERE_MEAN_POOL_CONSONANTAL_SIMILARITY),
    ("semantic_cohere_embed_v4_cantillation", COHERE_SOFT_ALIGNMENT_SIMILARITY),
    ("semantic_cohere_embed_v4_consonantal", COHERE_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY),
    ("semantic_cohere_embed_v4_vocalized", COHERE_MEAN_POOL_VOCALIZED_SIMILARITY),
    ("semantic_cohere_embed_v4_vocalized", COHERE_SOFT_ALIGNMENT_VOCALIZED_SIMILARITY),
    ("semantic_voyage_4_cantillation", VOYAGE_MEAN_POOL_SIMILARITY),
    ("semantic_voyage_4_consonantal", VOYAGE_MEAN_POOL_CONSONANTAL_SIMILARITY),
    ("semantic_voyage_4_cantillation", VOYAGE_SOFT_ALIGNMENT_SIMILARITY),
    ("semantic_voyage_4_consonantal", VOYAGE_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY),
    ("semantic_voyage_4_vocalized", VOYAGE_MEAN_POOL_VOCALIZED_SIMILARITY),
    ("semantic_voyage_4_vocalized", VOYAGE_SOFT_ALIGNMENT_VOCALIZED_SIMILARITY),
    ("semantic_bge_m3_cantillation", BGE_M3_MEAN_POOL_SIMILARITY),
    ("semantic_bge_m3_consonantal", BGE_M3_MEAN_POOL_CONSONANTAL_SIMILARITY),
    ("semantic_bge_m3_cantillation", BGE_M3_SOFT_ALIGNMENT_SIMILARITY),
    ("semantic_bge_m3_consonantal", BGE_M3_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY),
    ("semantic_bge_m3_vocalized", BGE_M3_MEAN_POOL_VOCALIZED_SIMILARITY),
    ("semantic_bge_m3_vocalized", BGE_M3_SOFT_ALIGNMENT_VOCALIZED_SIMILARITY),
    ("semantic_gte_multilingual_base_cantillation", GTE_MULTILINGUAL_MEAN_POOL_SIMILARITY),
    (
        "semantic_gte_multilingual_base_consonantal",
        GTE_MULTILINGUAL_MEAN_POOL_CONSONANTAL_SIMILARITY,
    ),
    ("semantic_gte_multilingual_base_cantillation", GTE_MULTILINGUAL_SOFT_ALIGNMENT_SIMILARITY),
    (
        "semantic_gte_multilingual_base_consonantal",
        GTE_MULTILINGUAL_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY,
    ),
    ("semantic_gte_multilingual_base_vocalized", GTE_MULTILINGUAL_MEAN_POOL_VOCALIZED_SIMILARITY),
    (
        "semantic_gte_multilingual_base_vocalized",
        GTE_MULTILINGUAL_SOFT_ALIGNMENT_VOCALIZED_SIMILARITY,
    ),
    ("semantic_me5_large_instruct_cantillation", ME5_LARGE_INSTRUCT_MEAN_POOL_SIMILARITY),
    (
        "semantic_me5_large_instruct_consonantal",
        ME5_LARGE_INSTRUCT_MEAN_POOL_CONSONANTAL_SIMILARITY,
    ),
    ("semantic_me5_large_instruct_cantillation", ME5_LARGE_INSTRUCT_SOFT_ALIGNMENT_SIMILARITY),
    (
        "semantic_me5_large_instruct_consonantal",
        ME5_LARGE_INSTRUCT_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY,
    ),
    ("semantic_me5_large_instruct_vocalized", ME5_LARGE_INSTRUCT_MEAN_POOL_VOCALIZED_SIMILARITY),
    (
        "semantic_me5_large_instruct_vocalized",
        ME5_LARGE_INSTRUCT_SOFT_ALIGNMENT_VOCALIZED_SIMILARITY,
    ),
)


#: Each contextual model whose embeddings get both anisotropy corrections, and the keys they take.
_ANISOTROPY_SOURCES = (
    ("semantic_alephbert_consonantal", _ALEPHBERT_TOP_PC_KEY, _ALEPHBERT_WHITENED_KEY),
    ("semantic_berel_consonantal", _BEREL_TOP_PC_KEY, _BEREL_WHITENED_KEY),
    ("semantic_neodictabert_consonantal", _NEODICTABERT_TOP_PC_KEY, _NEODICTABERT_WHITENED_KEY),
)


def _add_anisotropy_corrected(embeddings_by_feature: dict[str, dict[int, np.ndarray]]) -> None:
    """Adds a top-PC-removed and a whitened variant for every model that takes the corrections."""
    for feature_name, top_pc_key, whitened_key in _ANISOTROPY_SOURCES:
        embeddings = embeddings_by_feature[feature_name]
        embeddings_by_feature[top_pc_key] = remove_top_principal_components(embeddings)
        embeddings_by_feature[whitened_key] = whiten(embeddings)


def _write_payload(path: Path, payload: object) -> None:
    """Writes one site payload atomically and reports the size it published."""
    write_json(path, payload, compact=True)
    print(f"Wrote {path} ({path.stat().st_size / 1024:.0f} KiB)", file=sys.stderr)


def compute_one_similarity(
    item: tuple[list[Psalm], Callable[[list[Psalm]], FeatureMatrix], SimilarityMethod],
) -> MethodComputation:
    """One method's features, weights, and similarity, computed independently of every other."""
    psalms, build_features, method = item
    print(f"Computing {method.name}...", file=sys.stderr)
    features = build_features(psalms)
    return MethodComputation(
        features=features,
        weights=tfidf_weights(features),
        result=method.compute(features),
    )


def _compute_similarities(
    psalms: list[Psalm], *, max_workers: int | None = None
) -> list[MethodComputation]:
    """Every lexical and syntactic similarity method, computed over the same psalms."""
    return map_in_pool(
        compute_one_similarity,
        [(psalms, build_features, method) for build_features, method in _METHODS],
        max_workers=max_workers,
    )


def _semantic_similarity_for(
    cache_dir: Path,
    method: MeanPoolEmbeddingSimilarity | SoftAlignmentEmbeddingSimilarity,
    embeddings: dict[int, np.ndarray],
) -> SimilarityResult:
    """One method's similarity, reused only when the cache holds it for these embeddings."""
    fingerprint = fingerprint_embeddings(embeddings)
    cached = load_cached_similarity(cache_dir, method.name, fingerprint=fingerprint)
    if cached is not None:
        return cached
    computed = method.compute(embeddings)
    write_cached_similarity(cache_dir, computed, fingerprint=fingerprint)
    return computed


def run(
    checkout: str,
    data_root: Path,
    output: Path | None,
    embeddings_dir: Path | None,
    cache_dir: Path = DEFAULT_CLUSTERING_CACHE_DIR,
) -> None:
    """Computes every similarity signal and publishes the compare analysis."""
    output = output or default_similarity_output(data_root)

    print("Loading BHSA corpus via Text-Fabric...", file=sys.stderr)
    corpus = Corpus.load(checkout)

    print("Extracting psalms...", file=sys.stderr)
    psalms = corpus.psalms()
    if len(psalms) != EXPECTED_PSALMS:
        print(f"warning: expected {EXPECTED_PSALMS} psalms, found {len(psalms)}", file=sys.stderr)

    computations = _compute_similarities(psalms)

    if embeddings_dir is None:
        raise ValueError(
            "No tehillim-embeddings checkout given. Set --embeddings-dir or "
            "TEHILLIM_EMBEDDINGS_DIR to a local tehillim-embeddings checkout's "
            "data/domain=semantic/ directory."
        )
    semantic_similarities = _semantic_similarities(psalms, embeddings_dir, cache_dir)

    print("Building export payload...", file=sys.stderr)
    _write_payload(
        output,
        build_similarity_payload(
            psalms=psalms, computations=computations, default_method=_METHODS[0][1].name
        ),
    )

    print("Writing the partitioned dataset...", file=sys.stderr)
    written = write_compare_dataset(
        data_root,
        similarities=[c.result for c in computations] + semantic_similarities,
        psalms=psalms,
    )
    for path in written:
        print(f"Wrote {path} ({path.stat().st_size / 1024:.0f} KiB)", file=sys.stderr)


def _semantic_similarities(
    psalms: list[Psalm], embeddings_dir: Path, cache_dir: Path
) -> list[SimilarityResult]:
    """Every semantic method's similarity, refusing a run that would silently drop one."""
    print("Loading semantic-embedding signals from Parquet...", file=sys.stderr)
    embeddings_by_feature: dict[str, dict[int, np.ndarray]] = {}
    needed_features = {
        feature_name
        for feature_name, _ in _SEMANTIC_METHODS
        if feature_name.startswith("semantic_")
    }
    for feature_name in sorted(needed_features):
        embeddings = load_semantic_embeddings(embeddings_dir, feature_name, psalms)
        if embeddings is not None:
            embeddings_by_feature[feature_name] = embeddings
    require_every_semantic_feature(set(embeddings_by_feature), needed_features, embeddings_dir)

    print(
        "Applying anisotropy corrections to AlephBERT, BEREL, and NeoDictaBERT...",
        file=sys.stderr,
    )
    _add_anisotropy_corrected(embeddings_by_feature)

    similarities: list[SimilarityResult] = []
    for feature_name, method in _SEMANTIC_METHODS:
        if feature_name not in embeddings_by_feature:
            continue
        similarities.append(
            _semantic_similarity_for(cache_dir, method, embeddings_by_feature[feature_name])
        )
    return similarities


def main(argv: list[str] | None = None) -> int:
    """Entry point: parses the command line and runs the requested subcommand."""
    args = parse_args(argv)
    run(
        args.checkout,
        args.data_root,
        args.output,
        args.embeddings_dir,
        args.cache_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
