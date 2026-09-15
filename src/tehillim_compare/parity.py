"""Checks every representation reached its domain's method table, or was skipped on record."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
from core.datasets import discover_domains
from core.skips import skipped_in_log

from tehillim_compare.dataset import ANALYSIS_COMPARE, partition_path
from tehillim_compare.embedding_methods import AGGREGATIONS
from tehillim_compare.representation import read_representation
from tehillim_compare.stages import Roots, domain_datasets


class ParityError(RuntimeError):
    """A domain's table lacks a representation with no recorded skip, or holds a stale one."""


def scored_representations(roots: Roots, domain: str) -> set[str]:
    """The representations a domain's method table holds under every aggregation, else absent."""
    path = partition_path(roots.data_root, ANALYSIS_COMPARE, domain, "raw", "methods.parquet")
    if not path.exists():
        return set()
    methods = set(pq.read_table(path, columns=["method"]).column("method").to_pylist())
    suffixes = [f"-{aggregation.key}-cosine" for aggregation in AGGREGATIONS]
    candidates = {m.removesuffix(suffixes[0]) for m in methods if m.endswith(suffixes[0])}
    return {c for c in candidates if all(f"{c}{suffix}" in methods for suffix in suffixes)}


def check_parity(roots: Roots, log_root: Path) -> dict[str, dict[str, object]]:
    """Compares every domain's table with its datasets, raising on an unexplained gap."""
    report: dict[str, dict[str, object]] = {}
    failures: list[str] = []
    for domain in discover_domains(roots.embeddings_root):
        datasets = {read_representation(p).identifier for p in domain_datasets(roots, domain)}
        scored = scored_representations(roots, domain)
        skipped = skipped_in_log(log_root / f"compare.{domain}.log")
        missing = datasets - scored
        unexplained = sorted(missing - skipped)
        stale = sorted(scored - datasets)
        report[domain] = {
            "datasets": len(datasets),
            "scored": len(scored & datasets),
            "skipped": sorted(missing & skipped),
            "unexplained": unexplained,
            "stale": stale,
        }
        if unexplained or stale:
            failures.append(f"compare.{domain}: unexplained {unexplained} stale {stale}")
    if failures:
        raise ParityError("\n".join(failures))
    return report
