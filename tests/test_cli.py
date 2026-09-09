from __future__ import annotations

from pathlib import Path

from tehillim_compare import cli
from tehillim_compare.cli import (
    DEFAULT_CLUSTERING_CACHE_DIR,
    DEFAULT_DATA_ROOT,
    default_similarity_output,
    parse_args,
)
from tehillim_compare.corpus import DEFAULT_CHECKOUT


def test_parse_args_defaults_checkout_without_env():
    args = parse_args([], env={})
    assert args.checkout == DEFAULT_CHECKOUT
    assert args.output is None


def test_parse_args_reads_checkout_from_env():
    args = parse_args([], env={"TEHILLIM_CHECKOUT": "v1.0"})
    assert args.checkout == "v1.0"


def test_parse_args_explicit_checkout_flag_overrides_env():
    args = parse_args(
        ["--checkout", "hot", "--output", "/tmp/out.json"], env={"TEHILLIM_CHECKOUT": "v1.0"}
    )
    assert args.checkout == "hot"
    assert args.output == Path("/tmp/out.json")


def test_parse_args_defaults_embeddings_dir_to_none_without_env():
    args = parse_args([], env={})
    assert args.embeddings_dir is None


def test_parse_args_reads_embeddings_dir_from_env():
    args = parse_args([], env={"TEHILLIM_EMBEDDINGS_DIR": "/some/embeddings/dir"})
    assert args.embeddings_dir == Path("/some/embeddings/dir")


def test_parse_args_explicit_embeddings_dir_flag_overrides_env():
    args = parse_args(
        ["--embeddings-dir", "/flag/dir"], env={"TEHILLIM_EMBEDDINGS_DIR": "/env/dir"}
    )
    assert args.embeddings_dir == Path("/flag/dir")


def test_default_data_root_points_into_this_repos_data_dir():
    assert Path(cli.__file__).resolve().parents[2] / "data" == DEFAULT_DATA_ROOT


def test_parse_args_defaults_data_root_without_env():
    assert parse_args([], env={}).data_root == DEFAULT_DATA_ROOT


def test_parse_args_reads_data_root_from_env():
    args = parse_args([], env={"TEHILLIM_DATA_DIR": "/data"})
    assert args.data_root == Path("/data")


def test_parse_args_explicit_data_root_flag_overrides_env():
    args = parse_args(["--data-root", "/flag"], env={"TEHILLIM_DATA_DIR": "/env"})
    assert args.data_root == Path("/flag")


def test_the_compare_payload_defaults_under_its_analysis():
    assert default_similarity_output(Path("/d")) == Path(
        "/d/analysis=compare/stage=ui/similarity.json"
    )


def test_parse_args_defaults_cache_dir_without_env():
    args = parse_args([], env={})
    assert args.cache_dir == DEFAULT_CLUSTERING_CACHE_DIR


def test_parse_args_reads_cache_dir_from_env():
    args = parse_args([], env={"TEHILLIM_CLUSTERING_CACHE_DIR": "/some/cache/dir"})
    assert args.cache_dir == Path("/some/cache/dir")


def test_parse_args_explicit_cache_dir_flag_overrides_env():
    args = parse_args(
        ["--cache-dir", "/flag/path"], env={"TEHILLIM_CLUSTERING_CACHE_DIR": "/env/path"}
    )
    assert args.cache_dir == Path("/flag/path")
