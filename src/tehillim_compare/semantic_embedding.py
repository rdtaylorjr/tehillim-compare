"""Semantic similarity over psalm half-verse embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from tehillim_compare.similarity import SimilarityResult


def mean_pool_vectors(embeddings: dict[int, np.ndarray]) -> tuple[np.ndarray, tuple[int, ...]]:
    """Collapses each psalm's half-verse embeddings to their mean."""
    psalm_numbers = tuple(embeddings.keys())
    #: Averaging stored float32 rows in float32 loses bits the cosine then cannot recover.
    pooled = np.array([embeddings[p].mean(axis=0, dtype=np.float64) for p in psalm_numbers])
    return pooled, psalm_numbers


def mean_pool_similarity(
    embeddings: dict[int, np.ndarray],
    *,
    name: str = "semantic-mean-pool-cosine",
    description: str = "Cosine similarity between mean-pooled half-verse embeddings.",
) -> SimilarityResult:
    """Cosine-compare each psalm's mean-pooled half-verse embeddings."""
    pooled, psalm_numbers = mean_pool_vectors(embeddings)
    matrix = cosine_similarity(pooled)
    np.fill_diagonal(matrix, 1.0)
    return SimilarityResult(
        method=name, description=description, psalm_numbers=psalm_numbers, matrix=matrix
    )


def soft_alignment_similarity(
    embeddings: dict[int, np.ndarray],
    *,
    name: str = "semantic-soft-alignment-cosine",
    description: str = (
        "Symmetric best-match cosine similarity between two psalms' half-verse "
        "embedding sets, avoiding the single-pooled-vector bottleneck."
    ),
) -> SimilarityResult:
    """Chamfer-style soft alignment: each half-verse averaged against its best counterpart."""
    psalm_numbers = tuple(embeddings.keys())
    n = len(psalm_numbers)
    matrix = np.empty((n, n))
    for i, psalm_i in enumerate(psalm_numbers):
        matrix[i, i] = 1.0
        for j in range(i + 1, n):
            psalm_j = psalm_numbers[j]
            score = _soft_alignment_score(embeddings[psalm_i], embeddings[psalm_j])
            matrix[i, j] = score
            matrix[j, i] = score
    return SimilarityResult(
        method=name, description=description, psalm_numbers=psalm_numbers, matrix=matrix
    )


def _soft_alignment_score(a: np.ndarray, b: np.ndarray) -> float:
    #: Widened for the same reason as the pooled path, so both methods score at one precision.
    pairwise = cosine_similarity(a.astype(np.float64), b.astype(np.float64))
    a_to_b = pairwise.max(axis=1).mean()
    b_to_a = pairwise.max(axis=0).mean()
    return float((a_to_b + b_to_a) / 2)


@dataclass(frozen=True, slots=True)
class MeanPoolEmbeddingSimilarity:
    """A named `mean_pool_similarity` configuration."""

    name: str
    description: str

    def compute(self, embeddings: dict[int, np.ndarray]) -> SimilarityResult:
        """Similarity between psalms, from their half-verse embeddings."""
        return mean_pool_similarity(embeddings, name=self.name, description=self.description)


@dataclass(frozen=True, slots=True)
class SoftAlignmentEmbeddingSimilarity:
    """A named, documented `soft_alignment_similarity` configuration."""

    name: str
    description: str

    def compute(self, embeddings: dict[int, np.ndarray]) -> SimilarityResult:
        """Similarity between psalms, from their half-verse embeddings."""
        return soft_alignment_similarity(embeddings, name=self.name, description=self.description)


# --- Configured method instances, wired into cli.py's _SEMANTIC_METHODS ---

MIQRABERT_MEAN_POOL_SIMILARITY = MeanPoolEmbeddingSimilarity(
    name="miqrabert-mean-pool-cosine",
    description=(
        "Cosine similarity between MiqraBERT half-verse embeddings, mean-pooled "
        "per psalm. The naive extension of MiqraBERT's whole-verse pooling "
        "to half-verse input. Kept as a baseline, not assumed to be sufficient."
    ),
)

MIQRABERT_SOFT_ALIGNMENT_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="miqrabert-soft-alignment-cosine",
    description=(
        "Symmetric best-match cosine similarity between two psalms' MiqraBERT "
        "half-verse embedding sets. Avoids collapsing each psalm to one pooled "
        "vector, targeting MiqraBERT's diagnosed poetic-pooling "
        "weakness rather than just moving it to a finer input granularity."
    ),
)

ALEPHBERT_MEAN_POOL_SIMILARITY = MeanPoolEmbeddingSimilarity(
    name="alephbert-mean-pool-cosine",
    description=(
        "Cosine similarity between unfinetuned AlephBERT half-verse embeddings, "
        "mean-pooled per psalm. Ablation baseline: MiqraBERT was fine-tuned to "
        "detect textual parallels, not genre register, so this checks whether "
        "that fine-tuning helped or hurt for this project's actual question."
    ),
)

ALEPHBERT_SOFT_ALIGNMENT_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="alephbert-soft-alignment-cosine",
    description=(
        "Symmetric best-match cosine similarity between two psalms' unfinetuned "
        "AlephBERT half-verse embedding sets. Same ablation logic as "
        "alephbert-mean-pool-cosine, paired with the soft-alignment aggregation."
    ),
)

NEODICTABERT_MEAN_POOL_SIMILARITY = MeanPoolEmbeddingSimilarity(
    name="neodictabert-mean-pool-cosine",
    description=(
        "Cosine similarity between mean-pooled NeoDictaBERT half-verse embeddings. "
        "Tests whether Dicta's newest, empirically strongest Hebrew encoder "
        "generation raises genre-family AMI over AlephBERT's existing signal."
    ),
)

NEODICTABERT_SOFT_ALIGNMENT_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="neodictabert-soft-alignment-cosine",
    description=(
        "Symmetric best-match cosine similarity between two psalms' NeoDictaBERT "
        "half-verse embedding sets. Same ablation logic as "
        "neodictabert-mean-pool-cosine, paired with the soft-alignment aggregation."
    ),
)

BEREL_MEAN_POOL_SIMILARITY = MeanPoolEmbeddingSimilarity(
    name="berel-mean-pool-cosine",
    description=(
        "Cosine similarity between mean-pooled BEREL half-verse embeddings. Tests "
        "whether register proximity to Biblical Hebrew (BEREL is trained on "
        "Rabbinic Hebrew, not Modern Hebrew) raises genre-family AMI more than "
        "raw model scale or benchmark strength does."
    ),
)

BEREL_SOFT_ALIGNMENT_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="berel-soft-alignment-cosine",
    description=(
        "Symmetric best-match cosine similarity between two psalms' BEREL "
        "half-verse embedding sets. Same ablation logic as "
        "berel-mean-pool-cosine, paired with the soft-alignment aggregation."
    ),
)

#: Top-PC-removed and whitened mean-pool counterparts were built and tested, then not shipped.

ALEPHBERT_SOFT_ALIGNMENT_TOP_PC_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="alephbert-soft-alignment-top-pc-cosine",
    description=(
        "Symmetric best-match cosine similarity between two psalms' AlephBERT "
        "half-verse embedding sets, after removing the corpus's top principal "
        "component (Mu & Viswanath 2018's 'All-but-the-Top'). Tests whether "
        "surgically removing AlephBERT's dominant, generic-information "
        "direction helps or hurts genre-family AMI relative to raw AlephBERT."
    ),
)

ALEPHBERT_SOFT_ALIGNMENT_WHITENED_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="alephbert-soft-alignment-whitened-cosine",
    description=(
        "Symmetric best-match cosine similarity between two psalms' AlephBERT "
        "half-verse embedding sets, after whitening the corpus so its "
        "covariance is (approximately) the identity (Su et al. 2021). A more "
        "aggressive anisotropy correction than top-PC removal, normalizing "
        "spread in every direction, not just the dominant one."
    ),
)

#: BEREL and NeoDictaBERT get the same soft-alignment-only treatment as AlephBERT above.

BEREL_SOFT_ALIGNMENT_TOP_PC_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="berel-soft-alignment-top-pc-cosine",
    description=(
        "Symmetric best-match cosine similarity between two psalms' BEREL "
        "half-verse embedding sets, after removing the corpus's top principal "
        "component (Mu & Viswanath 2018's 'All-but-the-Top'). BEREL's raw "
        "soft-alignment result already sits close to AlephBERT's, so "
        "this tests whether the correction that helped AlephBERT sharpens "
        "BEREL's signal the same way."
    ),
)

BEREL_SOFT_ALIGNMENT_WHITENED_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="berel-soft-alignment-whitened-cosine",
    description=(
        "Symmetric best-match cosine similarity between two psalms' BEREL "
        "half-verse embedding sets, after whitening the corpus so its "
        "covariance is (approximately) the identity (Su et al. 2021), the "
        "more aggressive of the two corrections, tested for the same reason "
        "as berel-soft-alignment-top-pc-cosine."
    ),
)

NEODICTABERT_SOFT_ALIGNMENT_TOP_PC_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="neodictabert-soft-alignment-top-pc-cosine",
    description=(
        "Symmetric best-match cosine similarity between two psalms' "
        "NeoDictaBERT half-verse embedding sets, after removing the corpus's "
        "top principal component (Mu & Viswanath 2018's 'All-but-the-Top'). "
        "NeoDictaBERT's raw soft-alignment AMI is statistically "
        "indistinguishable from noise despite the model's strong general "
        "Hebrew-NLP benchmarks. Tests whether that null result reflects no "
        "real genre structure to find, or an anisotropic raw geometry "
        "burying a real signal correction could reveal."
    ),
)

NEODICTABERT_SOFT_ALIGNMENT_WHITENED_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="neodictabert-soft-alignment-whitened-cosine",
    description=(
        "Symmetric best-match cosine similarity between two psalms' "
        "NeoDictaBERT half-verse embedding sets, after whitening the corpus "
        "so its covariance is (approximately) the identity (Su et al. 2021) "
        "- the more aggressive of the two corrections, tested for the same "
        "reason as neodictabert-soft-alignment-top-pc-cosine."
    ),
)

#: A third tier of candidates (see this module's docstring): decoder-based.

BGE_MULTILINGUAL_GEMMA2_MEAN_POOL_SIMILARITY = MeanPoolEmbeddingSimilarity(
    name="bge-multilingual-gemma2-mean-pool-cosine",
    description=(
        "Cosine similarity between mean-pooled bge-multilingual-gemma2 half-verse "
        "embeddings. Tests whether a Gemma2-9B-based multilingual encoder topping the "
        "MIRACL benchmark (confirmed to preserve niqqud unlike every Hebrew-specific "
        "model already shipped) raises genre-family AMI over the existing signals."
    ),
)

BGE_MULTILINGUAL_GEMMA2_MEAN_POOL_CONSONANTAL_SIMILARITY = MeanPoolEmbeddingSimilarity(
    name="bge-multilingual-gemma2-mean-pool-consonantal-cosine",
    description=(
        "Same as bge-multilingual-gemma2-mean-pool-cosine, over unvocalized "
        "(consonantal-only) half-verses. The vocalized-vs-unvocalized ablation is "
        "real for this encoder, unlike every WordPiece-tokenized model above (see "
        "this module's docstring)."
    ),
)

BGE_MULTILINGUAL_GEMMA2_SOFT_ALIGNMENT_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="bge-multilingual-gemma2-soft-alignment-cosine",
    description=(
        "Symmetric best-match cosine similarity between two psalms' "
        "bge-multilingual-gemma2 half-verse embedding sets. Same ablation logic as "
        "bge-multilingual-gemma2-mean-pool-cosine, paired with the soft-alignment "
        "aggregation."
    ),
)

BGE_MULTILINGUAL_GEMMA2_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="bge-multilingual-gemma2-soft-alignment-consonantal-cosine",
    description=(
        "Same as bge-multilingual-gemma2-soft-alignment-cosine, over unvocalized "
        "(consonantal-only) half-verses. See "
        "bge-multilingual-gemma2-mean-pool-consonantal-cosine."
    ),
)

BGE_MULTILINGUAL_GEMMA2_MEAN_POOL_VOCALIZED_SIMILARITY = MeanPoolEmbeddingSimilarity(
    name="bge-multilingual-gemma2-mean-pool-vocalized-cosine",
    description=(
        "Same as bge-multilingual-gemma2-mean-pool-cosine, over niqqud-preserved, accent-stripped "
        "half-verses, a third, distinct text variant from vocalized (niqqud + "
        "cantillation accents) and unvocalized (neither), real for encoders whose "
        "tokenizer treats accents as distinct content from niqqud (see this module's "
        "docstring)."
    ),
)

BGE_MULTILINGUAL_GEMMA2_SOFT_ALIGNMENT_VOCALIZED_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="bge-multilingual-gemma2-soft-alignment-vocalized-cosine",
    description=(
        "Same as bge-multilingual-gemma2-soft-alignment-cosine, over niqqud-preserved, "
        "accent-stripped half-verses, see "
        "bge-multilingual-gemma2-mean-pool-vocalized-cosine."
    ),
)

QWEN3_EMBEDDING_MEAN_POOL_SIMILARITY = MeanPoolEmbeddingSimilarity(
    name="qwen3-embedding-mean-pool-cosine",
    description=(
        "Cosine similarity between mean-pooled Qwen3-Embedding-8B half-verse "
        "embeddings. Tests whether the top open-weight MTEB v2 multilingual model at "
        "release (70.58, confirmed to preserve niqqud) raises genre-family AMI over "
        "the existing signals."
    ),
)

QWEN3_EMBEDDING_MEAN_POOL_CONSONANTAL_SIMILARITY = MeanPoolEmbeddingSimilarity(
    name="qwen3-embedding-mean-pool-consonantal-cosine",
    description=(
        "Same as qwen3-embedding-mean-pool-cosine, over unvocalized "
        "(consonantal-only) half-verses. The vocalized-vs-unvocalized ablation is "
        "real for this encoder, unlike every WordPiece-tokenized model above (see "
        "this module's docstring)."
    ),
)

QWEN3_EMBEDDING_SOFT_ALIGNMENT_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="qwen3-embedding-soft-alignment-cosine",
    description=(
        "Symmetric best-match cosine similarity between two psalms' "
        "Qwen3-Embedding-8B half-verse embedding sets. Same ablation logic as "
        "qwen3-embedding-mean-pool-cosine, paired with the soft-alignment aggregation."
    ),
)

QWEN3_EMBEDDING_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="qwen3-embedding-soft-alignment-consonantal-cosine",
    description=(
        "Same as qwen3-embedding-soft-alignment-cosine, over unvocalized "
        "(consonantal-only) half-verses. See "
        "qwen3-embedding-mean-pool-consonantal-cosine."
    ),
)

QWEN3_EMBEDDING_MEAN_POOL_VOCALIZED_SIMILARITY = MeanPoolEmbeddingSimilarity(
    name="qwen3-embedding-mean-pool-vocalized-cosine",
    description=(
        "Same as qwen3-embedding-mean-pool-cosine, over niqqud-preserved, accent-stripped "
        "half-verses, a third, distinct text variant from vocalized (niqqud + "
        "cantillation accents) and unvocalized (neither), real for encoders whose "
        "tokenizer treats accents as distinct content from niqqud (see this module's "
        "docstring)."
    ),
)

QWEN3_EMBEDDING_SOFT_ALIGNMENT_VOCALIZED_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="qwen3-embedding-soft-alignment-vocalized-cosine",
    description=(
        "Same as qwen3-embedding-soft-alignment-cosine, over niqqud-preserved, accent-stripped "
        "half-verses. See qwen3-embedding-mean-pool-vocalized-cosine."
    ),
)

KALM_EMBEDDING_MEAN_POOL_SIMILARITY = MeanPoolEmbeddingSimilarity(
    name="kalm-embedding-mean-pool-cosine",
    description=(
        "Cosine similarity between mean-pooled KaLM-Embedding-Gemma3-12B half-verse "
        "embeddings. Tests whether the top MMTEB multilingual model (72.32, "
        "Gemma3-12B-based, confirmed to preserve niqqud) raises genre-family AMI "
        "over the existing signals."
    ),
)

KALM_EMBEDDING_MEAN_POOL_CONSONANTAL_SIMILARITY = MeanPoolEmbeddingSimilarity(
    name="kalm-embedding-mean-pool-consonantal-cosine",
    description=(
        "Same as kalm-embedding-mean-pool-cosine, over unvocalized (consonantal-only) "
        "half-verses. The vocalized-vs-unvocalized ablation is real for this "
        "encoder, unlike every WordPiece-tokenized model above (see this module's "
        "docstring)."
    ),
)

KALM_EMBEDDING_SOFT_ALIGNMENT_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="kalm-embedding-soft-alignment-cosine",
    description=(
        "Symmetric best-match cosine similarity between two psalms' "
        "KaLM-Embedding-Gemma3-12B half-verse embedding sets. Same ablation logic as "
        "kalm-embedding-mean-pool-cosine, paired with the soft-alignment aggregation."
    ),
)

KALM_EMBEDDING_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="kalm-embedding-soft-alignment-consonantal-cosine",
    description=(
        "Same as kalm-embedding-soft-alignment-cosine, over unvocalized "
        "(consonantal-only) half-verses. See "
        "kalm-embedding-mean-pool-consonantal-cosine."
    ),
)

KALM_EMBEDDING_MEAN_POOL_VOCALIZED_SIMILARITY = MeanPoolEmbeddingSimilarity(
    name="kalm-embedding-mean-pool-vocalized-cosine",
    description=(
        "Same as kalm-embedding-mean-pool-cosine, over niqqud-preserved, accent-stripped "
        "half-verses, a third, distinct text variant from vocalized (niqqud + "
        "cantillation accents) and unvocalized (neither), real for encoders whose "
        "tokenizer treats accents as distinct content from niqqud (see this module's "
        "docstring)."
    ),
)

KALM_EMBEDDING_SOFT_ALIGNMENT_VOCALIZED_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="kalm-embedding-soft-alignment-vocalized-cosine",
    description=(
        "Same as kalm-embedding-soft-alignment-cosine, over niqqud-preserved, accent-stripped "
        "half-verses, see kalm-embedding-mean-pool-vocalized-cosine."
    ),
)

#: Llama-Embed-Nemotron-8B, the top multilingual MTEB model as of October 2025.

LLAMA_EMBED_NEMOTRON_MEAN_POOL_SIMILARITY = MeanPoolEmbeddingSimilarity(
    name="llama-embed-nemotron-mean-pool-cosine",
    description=(
        "Cosine similarity between mean-pooled Llama-Embed-Nemotron-8B half-verse "
        "embeddings. Tests whether the top multilingual MTEB model (as of October 2025, "
        "fine-tuned from Llama-3.1-8B with bidirectional attention) raises "
        "genre-family AMI over the existing signals."
    ),
)

LLAMA_EMBED_NEMOTRON_MEAN_POOL_CONSONANTAL_SIMILARITY = MeanPoolEmbeddingSimilarity(
    name="llama-embed-nemotron-mean-pool-consonantal-cosine",
    description=(
        "Same as llama-embed-nemotron-mean-pool-cosine, over unvocalized "
        "(consonantal-only) half-verses."
    ),
)

LLAMA_EMBED_NEMOTRON_SOFT_ALIGNMENT_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="llama-embed-nemotron-soft-alignment-cosine",
    description=(
        "Symmetric best-match cosine similarity between two psalms' "
        "Llama-Embed-Nemotron-8B half-verse embedding sets. Same ablation logic as "
        "llama-embed-nemotron-mean-pool-cosine, paired with the soft-alignment "
        "aggregation."
    ),
)

LLAMA_EMBED_NEMOTRON_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="llama-embed-nemotron-soft-alignment-consonantal-cosine",
    description=(
        "Same as llama-embed-nemotron-soft-alignment-cosine, over unvocalized "
        "(consonantal-only) half-verses. See "
        "llama-embed-nemotron-mean-pool-consonantal-cosine."
    ),
)

LLAMA_EMBED_NEMOTRON_MEAN_POOL_VOCALIZED_SIMILARITY = MeanPoolEmbeddingSimilarity(
    name="llama-embed-nemotron-mean-pool-vocalized-cosine",
    description=(
        "Same as llama-embed-nemotron-mean-pool-cosine, over niqqud-preserved, accent-stripped "
        "half-verses, a third, distinct text variant from vocalized (niqqud + "
        "cantillation accents) and unvocalized (neither), real for encoders whose "
        "tokenizer treats accents as distinct content from niqqud (see this module's "
        "docstring)."
    ),
)

LLAMA_EMBED_NEMOTRON_SOFT_ALIGNMENT_VOCALIZED_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="llama-embed-nemotron-soft-alignment-vocalized-cosine",
    description=(
        "Same as llama-embed-nemotron-soft-alignment-cosine, over niqqud-preserved, "
        "accent-stripped half-verses. See "
        "llama-embed-nemotron-mean-pool-vocalized-cosine."
    ),
)

#: Gemini Embedding 2, a paid-API candidate.

GEMINI_MEAN_POOL_SIMILARITY = MeanPoolEmbeddingSimilarity(
    name="gemini-mean-pool-cosine",
    description=(
        "Cosine similarity between mean-pooled Gemini Embedding 2 half-verse "
        "embeddings. Tests whether the top BEIR/MTEB-retrieval-subset model "
        "(67.71, confirmed to preserve niqqud) raises genre-family AMI over the "
        "existing signals, the one paid-API candidate in this ablation."
    ),
)

GEMINI_MEAN_POOL_CONSONANTAL_SIMILARITY = MeanPoolEmbeddingSimilarity(
    name="gemini-mean-pool-consonantal-cosine",
    description=(
        "Same as gemini-mean-pool-cosine, over unvocalized (consonantal-only) "
        "half-verses. The vocalized-vs-unvocalized ablation is real for this "
        "encoder, unlike every WordPiece-tokenized model above (see this module's "
        "docstring)."
    ),
)

GEMINI_SOFT_ALIGNMENT_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="gemini-soft-alignment-cosine",
    description=(
        "Symmetric best-match cosine similarity between two psalms' Gemini "
        "Embedding 2 half-verse embedding sets. Same ablation logic as "
        "gemini-mean-pool-cosine, paired with the soft-alignment aggregation."
    ),
)

GEMINI_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="gemini-soft-alignment-consonantal-cosine",
    description=(
        "Same as gemini-soft-alignment-cosine, over unvocalized (consonantal-only) "
        "half-verses. See gemini-mean-pool-consonantal-cosine."
    ),
)

GEMINI_MEAN_POOL_VOCALIZED_SIMILARITY = MeanPoolEmbeddingSimilarity(
    name="gemini-mean-pool-vocalized-cosine",
    description=(
        "Same as gemini-mean-pool-cosine, over niqqud-preserved, accent-stripped "
        "half-verses, a third, distinct text variant from vocalized (niqqud + "
        "cantillation accents) and unvocalized (neither), real for encoders whose "
        "tokenizer treats accents as distinct content from niqqud (see this module's "
        "docstring)."
    ),
)

GEMINI_SOFT_ALIGNMENT_VOCALIZED_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="gemini-soft-alignment-vocalized-cosine",
    description=(
        "Same as gemini-soft-alignment-cosine, over niqqud-preserved, accent-stripped "
        "half-verses. See gemini-mean-pool-vocalized-cosine."
    ),
)

#: Cohere Embed v4, a Tier 2 paid-API candidate.

COHERE_MEAN_POOL_SIMILARITY = MeanPoolEmbeddingSimilarity(
    name="cohere-mean-pool-cosine",
    description=(
        "Cosine similarity between mean-pooled Cohere Embed v4 half-verse "
        "embeddings. Tests whether a real 65.2 MTEB multilingual encoder "
        "(confirmed to preserve niqqud via a disclosed proxy) raises genre-family "
        "AMI over the existing signals, one of Tier 2's two paid-API candidates."
    ),
)

COHERE_MEAN_POOL_CONSONANTAL_SIMILARITY = MeanPoolEmbeddingSimilarity(
    name="cohere-mean-pool-consonantal-cosine",
    description=(
        "Same as cohere-mean-pool-cosine, over unvocalized (consonantal-only) "
        "half-verses. The vocalized-vs-unvocalized ablation is real for this "
        "encoder, unlike every WordPiece-tokenized model above (see this module's "
        "docstring)."
    ),
)

COHERE_SOFT_ALIGNMENT_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="cohere-soft-alignment-cosine",
    description=(
        "Symmetric best-match cosine similarity between two psalms' Cohere Embed "
        "v4 half-verse embedding sets. Same ablation logic as "
        "cohere-mean-pool-cosine, paired with the soft-alignment aggregation."
    ),
)

COHERE_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="cohere-soft-alignment-consonantal-cosine",
    description=(
        "Same as cohere-soft-alignment-cosine, over unvocalized (consonantal-only) "
        "half-verses. See cohere-mean-pool-consonantal-cosine."
    ),
)

COHERE_MEAN_POOL_VOCALIZED_SIMILARITY = MeanPoolEmbeddingSimilarity(
    name="cohere-mean-pool-vocalized-cosine",
    description=(
        "Same as cohere-mean-pool-cosine, over niqqud-preserved, accent-stripped "
        "half-verses, a third, distinct text variant from vocalized (niqqud + "
        "cantillation accents) and unvocalized (neither), real for encoders whose "
        "tokenizer treats accents as distinct content from niqqud (see this module's "
        "docstring)."
    ),
)

COHERE_SOFT_ALIGNMENT_VOCALIZED_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="cohere-soft-alignment-vocalized-cosine",
    description=(
        "Same as cohere-soft-alignment-cosine, over niqqud-preserved, accent-stripped "
        "half-verses, see cohere-mean-pool-vocalized-cosine."
    ),
)

#: OpenAI text-embedding-3-large, Tier 2's second paid-API candidate.

OPENAI_MEAN_POOL_SIMILARITY = MeanPoolEmbeddingSimilarity(
    name="openai-mean-pool-cosine",
    description=(
        "Cosine similarity between mean-pooled OpenAI text-embedding-3-large "
        "half-verse embeddings. Tests whether a real 64.6 MTEB encoder (confirmed "
        "to preserve niqqud, at a real measured ~3x token-fragmentation cost on "
        "vocalized Hebrew) raises genre-family AMI over the existing signals, the "
        "second of Tier 2's two paid-API candidates."
    ),
)

OPENAI_MEAN_POOL_CONSONANTAL_SIMILARITY = MeanPoolEmbeddingSimilarity(
    name="openai-mean-pool-consonantal-cosine",
    description=(
        "Same as openai-mean-pool-cosine, over unvocalized (consonantal-only) "
        "half-verses. The vocalized-vs-unvocalized ablation is real for this "
        "encoder, unlike every WordPiece-tokenized model above (see this module's "
        "docstring)."
    ),
)

OPENAI_SOFT_ALIGNMENT_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="openai-soft-alignment-cosine",
    description=(
        "Symmetric best-match cosine similarity between two psalms' OpenAI "
        "text-embedding-3-large half-verse embedding sets. Same ablation logic as "
        "openai-mean-pool-cosine, paired with the soft-alignment aggregation."
    ),
)

OPENAI_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="openai-soft-alignment-consonantal-cosine",
    description=(
        "Same as openai-soft-alignment-cosine, over unvocalized (consonantal-only) "
        "half-verses. See openai-mean-pool-consonantal-cosine."
    ),
)

OPENAI_MEAN_POOL_VOCALIZED_SIMILARITY = MeanPoolEmbeddingSimilarity(
    name="openai-mean-pool-vocalized-cosine",
    description=(
        "Same as openai-mean-pool-cosine, over niqqud-preserved, accent-stripped "
        "half-verses, a third, distinct text variant from vocalized (niqqud + "
        "cantillation accents) and unvocalized (neither), real for encoders whose "
        "tokenizer treats accents as distinct content from niqqud (see this module's "
        "docstring)."
    ),
)

OPENAI_SOFT_ALIGNMENT_VOCALIZED_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="openai-soft-alignment-vocalized-cosine",
    description=(
        "Same as openai-soft-alignment-cosine, over niqqud-preserved, accent-stripped "
        "half-verses. See openai-mean-pool-vocalized-cosine."
    ),
)

#: Voyage 4, Tier 3's paid-API candidate, same shape as Cohere/Gemini/OpenAI above.

VOYAGE_MEAN_POOL_SIMILARITY = MeanPoolEmbeddingSimilarity(
    name="voyage-mean-pool-cosine",
    description=(
        "Cosine similarity between mean-pooled Voyage 4 half-verse embeddings. "
        "Tests whether Voyage AI's January 2026 general-purpose model raises "
        "genre-family AMI over the existing signals."
    ),
)

VOYAGE_MEAN_POOL_CONSONANTAL_SIMILARITY = MeanPoolEmbeddingSimilarity(
    name="voyage-mean-pool-consonantal-cosine",
    description=(
        "Same as voyage-mean-pool-cosine, over unvocalized (consonantal-only) half-verses."
    ),
)

VOYAGE_SOFT_ALIGNMENT_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="voyage-soft-alignment-cosine",
    description=(
        "Symmetric best-match cosine similarity between two psalms' Voyage 4 "
        "half-verse embedding sets. Same ablation logic as voyage-mean-pool-cosine, "
        "paired with the soft-alignment aggregation."
    ),
)

VOYAGE_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="voyage-soft-alignment-consonantal-cosine",
    description=(
        "Same as voyage-soft-alignment-cosine, over unvocalized (consonantal-only) "
        "half-verses. See voyage-mean-pool-consonantal-cosine."
    ),
)

VOYAGE_MEAN_POOL_VOCALIZED_SIMILARITY = MeanPoolEmbeddingSimilarity(
    name="voyage-mean-pool-vocalized-cosine",
    description=(
        "Same as voyage-mean-pool-cosine, over niqqud-preserved, accent-stripped "
        "half-verses, a third, distinct text variant from vocalized (niqqud + "
        "cantillation accents) and unvocalized (neither), real for encoders whose "
        "tokenizer treats accents as distinct content from niqqud (see this module's "
        "docstring)."
    ),
)

VOYAGE_SOFT_ALIGNMENT_VOCALIZED_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="voyage-soft-alignment-vocalized-cosine",
    description=(
        "Same as voyage-soft-alignment-cosine, over niqqud-preserved, accent-stripped "
        "half-verses. See voyage-mean-pool-vocalized-cosine."
    ),
)

#: A fourth tier of candidates (see this module's docstring): three more niqqud-preserving.

BGE_M3_MEAN_POOL_SIMILARITY = MeanPoolEmbeddingSimilarity(
    name="bge-m3-mean-pool-cosine",
    description=(
        "Cosine similarity between mean-pooled BGE-M3 half-verse embeddings. Tests "
        "whether a real ~63.0 MTEB multilingual encoder (XLM-RoBERTa-large based, "
        "confirmed to preserve niqqud, CLS-token pooled) raises genre-family AMI "
        "over the existing signals."
    ),
)

BGE_M3_MEAN_POOL_CONSONANTAL_SIMILARITY = MeanPoolEmbeddingSimilarity(
    name="bge-m3-mean-pool-consonantal-cosine",
    description=(
        "Same as bge-m3-mean-pool-cosine, over unvocalized (consonantal-only) "
        "half-verses. The vocalized-vs-unvocalized ablation is real for this "
        "encoder, unlike every WordPiece-tokenized model above (see this module's "
        "docstring)."
    ),
)

BGE_M3_SOFT_ALIGNMENT_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="bge-m3-soft-alignment-cosine",
    description=(
        "Symmetric best-match cosine similarity between two psalms' BGE-M3 "
        "half-verse embedding sets. Same ablation logic as bge-m3-mean-pool-cosine, "
        "paired with the soft-alignment aggregation."
    ),
)

BGE_M3_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="bge-m3-soft-alignment-consonantal-cosine",
    description=(
        "Same as bge-m3-soft-alignment-cosine, over unvocalized (consonantal-only) "
        "half-verses. See bge-m3-mean-pool-consonantal-cosine."
    ),
)

BGE_M3_MEAN_POOL_VOCALIZED_SIMILARITY = MeanPoolEmbeddingSimilarity(
    name="bge-m3-mean-pool-vocalized-cosine",
    description=(
        "Same as bge-m3-mean-pool-cosine, over niqqud-preserved, accent-stripped "
        "half-verses, a third, distinct text variant from vocalized (niqqud + "
        "cantillation accents) and unvocalized (neither), real for encoders whose "
        "tokenizer treats accents as distinct content from niqqud (see this module's "
        "docstring)."
    ),
)

BGE_M3_SOFT_ALIGNMENT_VOCALIZED_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="bge-m3-soft-alignment-vocalized-cosine",
    description=(
        "Same as bge-m3-soft-alignment-cosine, over niqqud-preserved, accent-stripped "
        "half-verses. See bge-m3-mean-pool-vocalized-cosine."
    ),
)

GTE_MULTILINGUAL_MEAN_POOL_SIMILARITY = MeanPoolEmbeddingSimilarity(
    name="gte-multilingual-base-mean-pool-cosine",
    description=(
        "Cosine similarity between mean-pooled GTE-multilingual-base half-verse "
        "embeddings. Tests whether Alibaba's mGTE encoder (RoPE-based bidirectional, "
        "confirmed to preserve niqqud, CLS-token pooled) raises genre-family AMI "
        "over the existing signals."
    ),
)

GTE_MULTILINGUAL_MEAN_POOL_CONSONANTAL_SIMILARITY = MeanPoolEmbeddingSimilarity(
    name="gte-multilingual-base-mean-pool-consonantal-cosine",
    description=(
        "Same as gte-multilingual-base-mean-pool-cosine, over unvocalized "
        "(consonantal-only) half-verses. The vocalized-vs-unvocalized ablation is "
        "real for this encoder, unlike every WordPiece-tokenized model above (see "
        "this module's docstring)."
    ),
)

GTE_MULTILINGUAL_SOFT_ALIGNMENT_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="gte-multilingual-base-soft-alignment-cosine",
    description=(
        "Symmetric best-match cosine similarity between two psalms' "
        "GTE-multilingual-base half-verse embedding sets. Same ablation logic as "
        "gte-multilingual-base-mean-pool-cosine, paired with the soft-alignment "
        "aggregation."
    ),
)

GTE_MULTILINGUAL_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="gte-multilingual-base-soft-alignment-consonantal-cosine",
    description=(
        "Same as gte-multilingual-base-soft-alignment-cosine, over unvocalized "
        "(consonantal-only) half-verses. See "
        "gte-multilingual-base-mean-pool-consonantal-cosine."
    ),
)

GTE_MULTILINGUAL_MEAN_POOL_VOCALIZED_SIMILARITY = MeanPoolEmbeddingSimilarity(
    name="gte-multilingual-base-mean-pool-vocalized-cosine",
    description=(
        "Same as gte-multilingual-base-mean-pool-cosine, over niqqud-preserved, accent-stripped "
        "half-verses, a third, distinct text variant from vocalized (niqqud + "
        "cantillation accents) and unvocalized (neither), real for encoders whose "
        "tokenizer treats accents as distinct content from niqqud (see this module's "
        "docstring)."
    ),
)

GTE_MULTILINGUAL_SOFT_ALIGNMENT_VOCALIZED_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="gte-multilingual-base-soft-alignment-vocalized-cosine",
    description=(
        "Same as gte-multilingual-base-soft-alignment-cosine, over niqqud-preserved, "
        "accent-stripped half-verses. See "
        "gte-multilingual-base-mean-pool-vocalized-cosine."
    ),
)

ME5_LARGE_INSTRUCT_MEAN_POOL_SIMILARITY = MeanPoolEmbeddingSimilarity(
    name="me5-large-instruct-mean-pool-cosine",
    description=(
        "Cosine similarity between mean-pooled multilingual-e5-large-instruct "
        "half-verse embeddings. Tests whether a real, widely-used multilingual "
        "encoder lineage (XLM-RoBERTa-large based, confirmed to preserve niqqud, "
        "mean-token pooled) raises genre-family AMI over the existing signals."
    ),
)

ME5_LARGE_INSTRUCT_MEAN_POOL_CONSONANTAL_SIMILARITY = MeanPoolEmbeddingSimilarity(
    name="me5-large-instruct-mean-pool-consonantal-cosine",
    description=(
        "Same as me5-large-instruct-mean-pool-cosine, over unvocalized "
        "(consonantal-only) half-verses. The vocalized-vs-unvocalized ablation is "
        "real for this encoder, unlike every WordPiece-tokenized model above (see "
        "this module's docstring)."
    ),
)

ME5_LARGE_INSTRUCT_SOFT_ALIGNMENT_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="me5-large-instruct-soft-alignment-cosine",
    description=(
        "Symmetric best-match cosine similarity between two psalms' "
        "multilingual-e5-large-instruct half-verse embedding sets. Same ablation "
        "logic as me5-large-instruct-mean-pool-cosine, paired with the "
        "soft-alignment aggregation."
    ),
)

ME5_LARGE_INSTRUCT_SOFT_ALIGNMENT_CONSONANTAL_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="me5-large-instruct-soft-alignment-consonantal-cosine",
    description=(
        "Same as me5-large-instruct-soft-alignment-cosine, over unvocalized "
        "(consonantal-only) half-verses. See "
        "me5-large-instruct-mean-pool-consonantal-cosine."
    ),
)

ME5_LARGE_INSTRUCT_MEAN_POOL_VOCALIZED_SIMILARITY = MeanPoolEmbeddingSimilarity(
    name="me5-large-instruct-mean-pool-vocalized-cosine",
    description=(
        "Same as me5-large-instruct-mean-pool-cosine, over niqqud-preserved, accent-stripped "
        "half-verses, a third, distinct text variant from vocalized (niqqud + "
        "cantillation accents) and unvocalized (neither), real for encoders whose "
        "tokenizer treats accents as distinct content from niqqud (see this module's "
        "docstring)."
    ),
)

ME5_LARGE_INSTRUCT_SOFT_ALIGNMENT_VOCALIZED_SIMILARITY = SoftAlignmentEmbeddingSimilarity(
    name="me5-large-instruct-soft-alignment-vocalized-cosine",
    description=(
        "Same as me5-large-instruct-soft-alignment-cosine, over niqqud-preserved, accent-stripped "
        "half-verses. See me5-large-instruct-mean-pool-vocalized-cosine."
    ),
)
