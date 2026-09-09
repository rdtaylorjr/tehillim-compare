"""Shared fixtures for integration tests against the real BHSA corpus."""

from __future__ import annotations

import os

# Limits BLAS threads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


from pathlib import Path

import numpy as np
import pytest

from tehillim_compare.anisotropy_correction import remove_top_principal_components, whiten
from tehillim_compare.clause_kind_profile import build_clause_kind_feature_matrix
from tehillim_compare.clause_relation_profile import build_clause_relation_feature_matrix
from tehillim_compare.clause_type_profile import build_clause_type_feature_matrix
from tehillim_compare.corpus import DEFAULT_CHECKOUT, Corpus
from tehillim_compare.features import build_lexical_feature_matrix
from tehillim_compare.gender_profile import build_gender_feature_matrix
from tehillim_compare.lexical_set import build_lexical_set_feature_matrix
from tehillim_compare.methods import (
    CLAUSE_KIND_SIMILARITY,
    CLAUSE_RELATION_SIMILARITY,
    CLAUSE_TYPE_SIMILARITY,
    GENDER_PROFILE_SIMILARITY,
    LEXICAL_SET_SIMILARITY,
    LEXICAL_SIMILARITY,
    NAMED_ENTITY_IDENTITY_SIMILARITY,
    NAMED_ENTITY_SIMILARITY,
    NOMINAL_STATE_SIMILARITY,
    PERSON_PROFILE_SIMILARITY,
    PHRASE_DEPENDENT_POS_SIMILARITY,
    PHRASE_DETERMINATION_SIMILARITY,
    PHRASE_FUNCTION_SIMILARITY,
    PHRASE_GRAMMATICAL_ROLE_SIMILARITY,
    PHRASE_TYPE_SIMILARITY,
    PHRASE_VALENCE_SIMILARITY,
    ROOT_SIMILARITY,
    TEXT_TYPE_SIMILARITY,
    VERB_MORPHOLOGY_SIMILARITY,
    VERB_SENSE_SIMILARITY,
)
from tehillim_compare.named_entity_identity import build_named_entity_identity_feature_matrix
from tehillim_compare.named_entity_profile import build_named_entity_feature_matrix
from tehillim_compare.nominal_state import build_nominal_state_feature_matrix
from tehillim_compare.person_profile import build_person_feature_matrix
from tehillim_compare.phrase_dependent_pos import build_phrase_dependent_pos_feature_matrix
from tehillim_compare.phrase_determination_profile import (
    build_phrase_determination_feature_matrix,
)
from tehillim_compare.phrase_function_profile import build_phrase_function_feature_matrix
from tehillim_compare.phrase_grammatical_role_profile import (
    build_phrase_grammatical_role_feature_matrix,
)
from tehillim_compare.phrase_type_profile import build_phrase_type_feature_matrix
from tehillim_compare.phrase_valence_profile import build_phrase_valence_feature_matrix
from tehillim_compare.root_similarity import build_root_feature_matrix
from tehillim_compare.semantic_embedding import (
    ALEPHBERT_MEAN_POOL_SIMILARITY,
    ALEPHBERT_SOFT_ALIGNMENT_SIMILARITY,
    ALEPHBERT_SOFT_ALIGNMENT_TOP_PC_SIMILARITY,
    ALEPHBERT_SOFT_ALIGNMENT_WHITENED_SIMILARITY,
    BEREL_MEAN_POOL_SIMILARITY,
    BEREL_SOFT_ALIGNMENT_SIMILARITY,
    BEREL_SOFT_ALIGNMENT_TOP_PC_SIMILARITY,
    BEREL_SOFT_ALIGNMENT_WHITENED_SIMILARITY,
    BGE_MULTILINGUAL_GEMMA2_MEAN_POOL_CONSONANTAL_SIMILARITY,
    BGE_MULTILINGUAL_GEMMA2_MEAN_POOL_SIMILARITY,
    BGE_MULTILINGUAL_GEMMA2_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY,
    BGE_MULTILINGUAL_GEMMA2_SOFT_ALIGNMENT_SIMILARITY,
    GEMINI_MEAN_POOL_CONSONANTAL_SIMILARITY,
    GEMINI_MEAN_POOL_SIMILARITY,
    GEMINI_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY,
    GEMINI_SOFT_ALIGNMENT_SIMILARITY,
    KALM_EMBEDDING_MEAN_POOL_CONSONANTAL_SIMILARITY,
    KALM_EMBEDDING_MEAN_POOL_SIMILARITY,
    KALM_EMBEDDING_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY,
    KALM_EMBEDDING_SOFT_ALIGNMENT_SIMILARITY,
    MIQRABERT_MEAN_POOL_SIMILARITY,
    MIQRABERT_SOFT_ALIGNMENT_SIMILARITY,
    NEODICTABERT_MEAN_POOL_SIMILARITY,
    NEODICTABERT_SOFT_ALIGNMENT_SIMILARITY,
    NEODICTABERT_SOFT_ALIGNMENT_TOP_PC_SIMILARITY,
    NEODICTABERT_SOFT_ALIGNMENT_WHITENED_SIMILARITY,
    QWEN3_EMBEDDING_MEAN_POOL_CONSONANTAL_SIMILARITY,
    QWEN3_EMBEDDING_MEAN_POOL_SIMILARITY,
    QWEN3_EMBEDDING_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY,
    QWEN3_EMBEDDING_SOFT_ALIGNMENT_SIMILARITY,
)
from tehillim_compare.semantic_embedding_loader import load_semantic_embeddings
from tehillim_compare.similarity import tfidf_weights
from tehillim_compare.text_type_profile import build_text_type_feature_matrix
from tehillim_compare.verb_morphology import build_verb_morphology_feature_matrix
from tehillim_compare.verb_sense_profile import build_verb_sense_feature_matrix


def _checkout() -> str:
    return os.environ.get("TEHILLIM_CHECKOUT", DEFAULT_CHECKOUT)


def _embeddings_dir_from_env() -> Path | None:
    value = os.environ.get("TEHILLIM_EMBEDDINGS_DIR")
    return Path(value) if value else None


def _try_load_corpus() -> tuple[Corpus | None, str]:
    """One load attempt, cached at import time, cheapest way to decide skip-vs-run up front."""
    try:
        return Corpus.load(_checkout()), ""
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


_CORPUS, _CORPUS_LOAD_ERROR = _try_load_corpus()
BHSA_AVAILABLE = _CORPUS is not None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not BHSA_AVAILABLE:
        skip_marker = pytest.mark.skip(
            reason=(
                f"BHSA/valence Text-Fabric data not available via use() (checkout="
                f"{_checkout()!r}): {_CORPUS_LOAD_ERROR}. Set TEHILLIM_CHECKOUT or GHPERS."
            )
        )
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_marker)

    if not os.environ.get("TEHILLIM_OPENROUTER_API_KEY"):
        api_skip_marker = pytest.mark.skip(
            reason="TEHILLIM_OPENROUTER_API_KEY is not set - skipping api_integration tests"
        )
        for item in items:
            if "api_integration" in item.keywords:
                item.add_marker(api_skip_marker)


@pytest.fixture(scope="session")
def corpus():
    assert _CORPUS is not None, "corpus fixture used without checking BHSA_AVAILABLE"
    return _CORPUS


@pytest.fixture(scope="session")
def psalms(corpus):
    return corpus.psalms()


@pytest.fixture(scope="session")
def embeddings_dir() -> Path:
    directory = _embeddings_dir_from_env()
    if directory is None or not directory.is_dir():
        pytest.skip(
            "TEHILLIM_EMBEDDINGS_DIR not set or not a directory (expected a local "
            "tehillim-embeddings checkout's data/type=semantic/ directory)"
        )
    return directory


def _load_embeddings(embeddings_dir: Path, feature_name: str, psalms) -> dict[int, np.ndarray]:
    embeddings = load_semantic_embeddings(embeddings_dir, feature_name, psalms)
    if embeddings is None:
        raise RuntimeError(f"{feature_name} not found in {embeddings_dir}")
    return embeddings


@pytest.fixture(scope="session")
def features(psalms):
    return build_lexical_feature_matrix(psalms)


@pytest.fixture(scope="session")
def weights(features):
    return tfidf_weights(features)


@pytest.fixture(scope="session")
def similarity_result(features):
    return LEXICAL_SIMILARITY.compute(features)


@pytest.fixture(scope="session")
def verb_morphology_features(psalms):
    return build_verb_morphology_feature_matrix(psalms)


@pytest.fixture(scope="session")
def verb_morphology_result(verb_morphology_features):
    return VERB_MORPHOLOGY_SIMILARITY.compute(verb_morphology_features)


@pytest.fixture(scope="session")
def person_profile_features(psalms):
    return build_person_feature_matrix(psalms)


@pytest.fixture(scope="session")
def person_profile_result(person_profile_features):
    return PERSON_PROFILE_SIMILARITY.compute(person_profile_features)


@pytest.fixture(scope="session")
def gender_profile_features(psalms):
    return build_gender_feature_matrix(psalms)


@pytest.fixture(scope="session")
def gender_profile_result(gender_profile_features):
    return GENDER_PROFILE_SIMILARITY.compute(gender_profile_features)


@pytest.fixture(scope="session")
def nominal_state_features(psalms):
    return build_nominal_state_feature_matrix(psalms)


@pytest.fixture(scope="session")
def nominal_state_result(nominal_state_features):
    return NOMINAL_STATE_SIMILARITY.compute(nominal_state_features)


@pytest.fixture(scope="session")
def lexical_set_features(psalms):
    return build_lexical_set_feature_matrix(psalms)


@pytest.fixture(scope="session")
def lexical_set_result(lexical_set_features):
    return LEXICAL_SET_SIMILARITY.compute(lexical_set_features)


@pytest.fixture(scope="session")
def phrase_dependent_pos_features(psalms):
    return build_phrase_dependent_pos_feature_matrix(psalms)


@pytest.fixture(scope="session")
def phrase_dependent_pos_result(phrase_dependent_pos_features):
    return PHRASE_DEPENDENT_POS_SIMILARITY.compute(phrase_dependent_pos_features)


@pytest.fixture(scope="session")
def named_entity_features(psalms):
    return build_named_entity_feature_matrix(psalms)


@pytest.fixture(scope="session")
def named_entity_result(named_entity_features):
    return NAMED_ENTITY_SIMILARITY.compute(named_entity_features)


@pytest.fixture(scope="session")
def root_features(psalms):
    return build_root_feature_matrix(psalms)


@pytest.fixture(scope="session")
def root_result(root_features):
    return ROOT_SIMILARITY.compute(root_features)


@pytest.fixture(scope="session")
def named_entity_identity_features(psalms):
    return build_named_entity_identity_feature_matrix(psalms)


@pytest.fixture(scope="session")
def named_entity_identity_result(named_entity_identity_features):
    return NAMED_ENTITY_IDENTITY_SIMILARITY.compute(named_entity_identity_features)


@pytest.fixture(scope="session")
def clause_type_features(psalms):
    return build_clause_type_feature_matrix(psalms)


@pytest.fixture(scope="session")
def clause_type_result(clause_type_features):
    return CLAUSE_TYPE_SIMILARITY.compute(clause_type_features)


@pytest.fixture(scope="session")
def text_type_features(psalms):
    return build_text_type_feature_matrix(psalms)


@pytest.fixture(scope="session")
def text_type_result(text_type_features):
    return TEXT_TYPE_SIMILARITY.compute(text_type_features)


@pytest.fixture(scope="session")
def clause_relation_features(psalms):
    return build_clause_relation_feature_matrix(psalms)


@pytest.fixture(scope="session")
def clause_relation_result(clause_relation_features):
    return CLAUSE_RELATION_SIMILARITY.compute(clause_relation_features)


@pytest.fixture(scope="session")
def clause_kind_features(psalms):
    return build_clause_kind_feature_matrix(psalms)


@pytest.fixture(scope="session")
def clause_kind_result(clause_kind_features):
    return CLAUSE_KIND_SIMILARITY.compute(clause_kind_features)


@pytest.fixture(scope="session")
def phrase_function_features(psalms):
    return build_phrase_function_feature_matrix(psalms)


@pytest.fixture(scope="session")
def phrase_function_result(phrase_function_features):
    return PHRASE_FUNCTION_SIMILARITY.compute(phrase_function_features)


@pytest.fixture(scope="session")
def phrase_determination_features(psalms):
    return build_phrase_determination_feature_matrix(psalms)


@pytest.fixture(scope="session")
def phrase_determination_result(phrase_determination_features):
    return PHRASE_DETERMINATION_SIMILARITY.compute(phrase_determination_features)


@pytest.fixture(scope="session")
def phrase_type_features(psalms):
    return build_phrase_type_feature_matrix(psalms)


@pytest.fixture(scope="session")
def phrase_type_result(phrase_type_features):
    return PHRASE_TYPE_SIMILARITY.compute(phrase_type_features)


@pytest.fixture(scope="session")
def phrase_valence_features(psalms):
    return build_phrase_valence_feature_matrix(psalms)


@pytest.fixture(scope="session")
def phrase_valence_result(phrase_valence_features):
    return PHRASE_VALENCE_SIMILARITY.compute(phrase_valence_features)


@pytest.fixture(scope="session")
def phrase_grammatical_role_features(psalms):
    return build_phrase_grammatical_role_feature_matrix(psalms)


@pytest.fixture(scope="session")
def phrase_grammatical_role_result(phrase_grammatical_role_features):
    return PHRASE_GRAMMATICAL_ROLE_SIMILARITY.compute(phrase_grammatical_role_features)


@pytest.fixture(scope="session")
def verb_sense_features(psalms):
    return build_verb_sense_feature_matrix(psalms)


@pytest.fixture(scope="session")
def verb_sense_result(verb_sense_features):
    return VERB_SENSE_SIMILARITY.compute(verb_sense_features)


# --- Semantic-embedding signals, session-scoped: computed once and reused ---


@pytest.fixture(scope="session")
def miqrabert_embeddings(embeddings_dir, psalms):
    return _load_embeddings(embeddings_dir, "semantic_miqrabert_consonantal", psalms)


@pytest.fixture(scope="session")
def alephbert_embeddings(embeddings_dir, psalms):
    return _load_embeddings(embeddings_dir, "semantic_alephbert_consonantal", psalms)


@pytest.fixture(scope="session")
def miqrabert_mean_pool_result(miqrabert_embeddings):
    return MIQRABERT_MEAN_POOL_SIMILARITY.compute(miqrabert_embeddings)


@pytest.fixture(scope="session")
def miqrabert_soft_alignment_result(miqrabert_embeddings):
    return MIQRABERT_SOFT_ALIGNMENT_SIMILARITY.compute(miqrabert_embeddings)


@pytest.fixture(scope="session")
def alephbert_mean_pool_result(alephbert_embeddings):
    return ALEPHBERT_MEAN_POOL_SIMILARITY.compute(alephbert_embeddings)


@pytest.fixture(scope="session")
def alephbert_soft_alignment_result(alephbert_embeddings):
    return ALEPHBERT_SOFT_ALIGNMENT_SIMILARITY.compute(alephbert_embeddings)


@pytest.fixture(scope="session")
def neodictabert_embeddings(embeddings_dir, psalms):
    return _load_embeddings(embeddings_dir, "semantic_neodictabert_consonantal", psalms)


@pytest.fixture(scope="session")
def neodictabert_mean_pool_result(neodictabert_embeddings):
    return NEODICTABERT_MEAN_POOL_SIMILARITY.compute(neodictabert_embeddings)


@pytest.fixture(scope="session")
def neodictabert_soft_alignment_result(neodictabert_embeddings):
    return NEODICTABERT_SOFT_ALIGNMENT_SIMILARITY.compute(neodictabert_embeddings)


@pytest.fixture(scope="session")
def berel_embeddings(embeddings_dir, psalms):
    return _load_embeddings(embeddings_dir, "semantic_berel_consonantal", psalms)


@pytest.fixture(scope="session")
def berel_mean_pool_result(berel_embeddings):
    return BEREL_MEAN_POOL_SIMILARITY.compute(berel_embeddings)


@pytest.fixture(scope="session")
def berel_soft_alignment_result(berel_embeddings):
    return BEREL_SOFT_ALIGNMENT_SIMILARITY.compute(berel_embeddings)


# --- Anisotropy-correction ablation, AlephBERT only, fit once per session ---


@pytest.fixture(scope="session")
def alephbert_top_pc_embeddings(alephbert_embeddings):
    return remove_top_principal_components(alephbert_embeddings)


@pytest.fixture(scope="session")
def alephbert_whitened_embeddings(alephbert_embeddings):
    return whiten(alephbert_embeddings)


@pytest.fixture(scope="session")
def alephbert_soft_alignment_top_pc_result(alephbert_top_pc_embeddings):
    return ALEPHBERT_SOFT_ALIGNMENT_TOP_PC_SIMILARITY.compute(alephbert_top_pc_embeddings)


@pytest.fixture(scope="session")
def alephbert_soft_alignment_whitened_result(alephbert_whitened_embeddings):
    return ALEPHBERT_SOFT_ALIGNMENT_WHITENED_SIMILARITY.compute(alephbert_whitened_embeddings)


# --- Anisotropy-correction ablation, BEREL and NeoDictaBERT, soft-alignment only ---


@pytest.fixture(scope="session")
def berel_top_pc_embeddings(berel_embeddings):
    return remove_top_principal_components(berel_embeddings)


@pytest.fixture(scope="session")
def berel_whitened_embeddings(berel_embeddings):
    return whiten(berel_embeddings)


@pytest.fixture(scope="session")
def berel_soft_alignment_top_pc_result(berel_top_pc_embeddings):
    return BEREL_SOFT_ALIGNMENT_TOP_PC_SIMILARITY.compute(berel_top_pc_embeddings)


@pytest.fixture(scope="session")
def berel_soft_alignment_whitened_result(berel_whitened_embeddings):
    return BEREL_SOFT_ALIGNMENT_WHITENED_SIMILARITY.compute(berel_whitened_embeddings)


@pytest.fixture(scope="session")
def neodictabert_top_pc_embeddings(neodictabert_embeddings):
    return remove_top_principal_components(neodictabert_embeddings)


@pytest.fixture(scope="session")
def neodictabert_whitened_embeddings(neodictabert_embeddings):
    return whiten(neodictabert_embeddings)


@pytest.fixture(scope="session")
def neodictabert_soft_alignment_top_pc_result(neodictabert_top_pc_embeddings):
    return NEODICTABERT_SOFT_ALIGNMENT_TOP_PC_SIMILARITY.compute(neodictabert_top_pc_embeddings)


@pytest.fixture(scope="session")
def neodictabert_soft_alignment_whitened_result(neodictabert_whitened_embeddings):
    return NEODICTABERT_SOFT_ALIGNMENT_WHITENED_SIMILARITY.compute(neodictabert_whitened_embeddings)


# --- Third-tier candidates: decoder-based, niqqud-preserving encoders ---


@pytest.fixture(scope="session")
def bge_multilingual_gemma2_embeddings(embeddings_dir, psalms):
    return _load_embeddings(embeddings_dir, "semantic_bge_multilingual_gemma2_cantillation", psalms)


@pytest.fixture(scope="session")
def bge_multilingual_gemma2_unvocalized_embeddings(embeddings_dir, psalms):
    return _load_embeddings(embeddings_dir, "semantic_bge_multilingual_gemma2_consonantal", psalms)


@pytest.fixture(scope="session")
def bge_multilingual_gemma2_mean_pool_result(bge_multilingual_gemma2_embeddings):
    return BGE_MULTILINGUAL_GEMMA2_MEAN_POOL_SIMILARITY.compute(bge_multilingual_gemma2_embeddings)


@pytest.fixture(scope="session")
def bge_multilingual_gemma2_mean_pool_unvocalized_result(
    bge_multilingual_gemma2_unvocalized_embeddings,
):
    return BGE_MULTILINGUAL_GEMMA2_MEAN_POOL_CONSONANTAL_SIMILARITY.compute(
        bge_multilingual_gemma2_unvocalized_embeddings
    )


@pytest.fixture(scope="session")
def bge_multilingual_gemma2_soft_alignment_result(bge_multilingual_gemma2_embeddings):
    return BGE_MULTILINGUAL_GEMMA2_SOFT_ALIGNMENT_SIMILARITY.compute(
        bge_multilingual_gemma2_embeddings
    )


@pytest.fixture(scope="session")
def bge_multilingual_gemma2_soft_alignment_unvocalized_result(
    bge_multilingual_gemma2_unvocalized_embeddings,
):
    return BGE_MULTILINGUAL_GEMMA2_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY.compute(
        bge_multilingual_gemma2_unvocalized_embeddings
    )


@pytest.fixture(scope="session")
def qwen3_embedding_embeddings(embeddings_dir, psalms):
    return _load_embeddings(embeddings_dir, "semantic_qwen3_embedding_8b_cantillation", psalms)


@pytest.fixture(scope="session")
def qwen3_embedding_unvocalized_embeddings(embeddings_dir, psalms):
    return _load_embeddings(embeddings_dir, "semantic_qwen3_embedding_8b_consonantal", psalms)


@pytest.fixture(scope="session")
def qwen3_embedding_mean_pool_result(qwen3_embedding_embeddings):
    return QWEN3_EMBEDDING_MEAN_POOL_SIMILARITY.compute(qwen3_embedding_embeddings)


@pytest.fixture(scope="session")
def qwen3_embedding_mean_pool_unvocalized_result(qwen3_embedding_unvocalized_embeddings):
    return QWEN3_EMBEDDING_MEAN_POOL_CONSONANTAL_SIMILARITY.compute(
        qwen3_embedding_unvocalized_embeddings
    )


@pytest.fixture(scope="session")
def qwen3_embedding_soft_alignment_result(qwen3_embedding_embeddings):
    return QWEN3_EMBEDDING_SOFT_ALIGNMENT_SIMILARITY.compute(qwen3_embedding_embeddings)


@pytest.fixture(scope="session")
def qwen3_embedding_soft_alignment_unvocalized_result(qwen3_embedding_unvocalized_embeddings):
    return QWEN3_EMBEDDING_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY.compute(
        qwen3_embedding_unvocalized_embeddings
    )


@pytest.fixture(scope="session")
def kalm_embedding_embeddings(embeddings_dir, psalms):
    return _load_embeddings(
        embeddings_dir, "semantic_kalm_embedding_gemma3_12b_2511_cantillation", psalms
    )


@pytest.fixture(scope="session")
def kalm_embedding_unvocalized_embeddings(embeddings_dir, psalms):
    return _load_embeddings(
        embeddings_dir, "semantic_kalm_embedding_gemma3_12b_2511_consonantal", psalms
    )


@pytest.fixture(scope="session")
def kalm_embedding_mean_pool_result(kalm_embedding_embeddings):
    return KALM_EMBEDDING_MEAN_POOL_SIMILARITY.compute(kalm_embedding_embeddings)


@pytest.fixture(scope="session")
def kalm_embedding_mean_pool_unvocalized_result(kalm_embedding_unvocalized_embeddings):
    return KALM_EMBEDDING_MEAN_POOL_CONSONANTAL_SIMILARITY.compute(
        kalm_embedding_unvocalized_embeddings
    )


@pytest.fixture(scope="session")
def kalm_embedding_soft_alignment_result(kalm_embedding_embeddings):
    return KALM_EMBEDDING_SOFT_ALIGNMENT_SIMILARITY.compute(kalm_embedding_embeddings)


@pytest.fixture(scope="session")
def kalm_embedding_soft_alignment_unvocalized_result(kalm_embedding_unvocalized_embeddings):
    return KALM_EMBEDDING_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY.compute(
        kalm_embedding_unvocalized_embeddings
    )


# --- Gemini Embedding 2, read via load_semantic_embeddings like the rest ---


@pytest.fixture(scope="session")
def gemini_vocalized_embeddings(embeddings_dir, psalms):
    return _load_embeddings(embeddings_dir, "semantic_gemini_embedding_2_cantillation", psalms)


@pytest.fixture(scope="session")
def gemini_unvocalized_embeddings(embeddings_dir, psalms):
    return _load_embeddings(embeddings_dir, "semantic_gemini_embedding_2_consonantal", psalms)


@pytest.fixture(scope="session")
def gemini_mean_pool_result(gemini_vocalized_embeddings):
    return GEMINI_MEAN_POOL_SIMILARITY.compute(gemini_vocalized_embeddings)


@pytest.fixture(scope="session")
def gemini_mean_pool_unvocalized_result(gemini_unvocalized_embeddings):
    return GEMINI_MEAN_POOL_CONSONANTAL_SIMILARITY.compute(gemini_unvocalized_embeddings)


@pytest.fixture(scope="session")
def gemini_soft_alignment_result(gemini_vocalized_embeddings):
    return GEMINI_SOFT_ALIGNMENT_SIMILARITY.compute(gemini_vocalized_embeddings)


@pytest.fixture(scope="session")
def gemini_soft_alignment_unvocalized_result(gemini_unvocalized_embeddings):
    return GEMINI_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY.compute(gemini_unvocalized_embeddings)
