"""Shared fixtures for integration tests against the real BHSA corpus."""

from __future__ import annotations

import os

# Limits BLAS threads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


from pathlib import Path

import pytest

from tehillim_compare.corpus import DEFAULT_CHECKOUT, Corpus


def _checkout() -> str:
    return os.environ.get("TEHILLIM_CHECKOUT", DEFAULT_CHECKOUT)


def _embeddings_root_from_env() -> Path | None:
    value = os.environ.get("TEHILLIM_EMBEDDINGS_DATA")
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
                f"BHSA Text-Fabric data not available via use() (checkout="
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
def embeddings_root() -> Path:
    directory = _embeddings_root_from_env()
    if directory is None or not directory.is_dir():
        pytest.skip("TEHILLIM_EMBEDDINGS_DATA not set to a tehillim-embeddings data directory")
    return directory
