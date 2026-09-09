"""The shared BHSA loading procedure: local clone first, Text-Fabric's use() only as a fallback."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tehillim_compare.corpus import (
    DEFAULT_BHSA_CLONE,
    Corpus,
    bhsa_clone_location,
)


class TestBhsaCloneLocation:
    def test_prefers_the_environment_variable_when_it_is_set(self) -> None:
        assert bhsa_clone_location(env={"TEHILLIM_BHSA_PATH": "/custom/tf"}) == Path("/custom/tf")

    def test_falls_back_to_the_conventional_clone_path_when_unset(self) -> None:
        assert bhsa_clone_location(env={}) == DEFAULT_BHSA_CLONE

    def test_treats_an_empty_variable_as_unset(self) -> None:
        assert bhsa_clone_location(env={"TEHILLIM_BHSA_PATH": ""}) == DEFAULT_BHSA_CLONE


def _api() -> SimpleNamespace:
    """A stand-in Text-Fabric api exposing the features Corpus.load checks for."""
    features = SimpleNamespace(otype=object(), book=object())
    return SimpleNamespace(F=features, TF=SimpleNamespace(load=lambda *a, **k: None))


class TestCorpusLoadPrefersTheLocalClone:
    def test_loads_from_the_local_clone_without_calling_use(self, tmp_path: Path) -> None:
        called: list[str] = []

        def _fabric(locations, silent):
            called.append("fabric")
            return SimpleNamespace(loadAll=lambda silent: _api())

        def _use(*args, **kwargs):
            called.append("use")
            raise AssertionError("use() must not run when the local clone works")

        Corpus.load(
            fabric_class=_fabric,
            use_fn=_use,
            env={"TEHILLIM_BHSA_PATH": str(tmp_path)},
            required_features="otype book",
        )

        assert called == ["fabric"]

    def test_passes_the_env_var_path_to_text_fabric(self, tmp_path: Path) -> None:
        seen: dict[str, object] = {}

        def _fabric(locations, silent):
            seen["locations"] = locations
            return SimpleNamespace(loadAll=lambda silent: _api())

        Corpus.load(
            fabric_class=_fabric,
            env={"TEHILLIM_BHSA_PATH": str(tmp_path)},
            required_features="otype book",
        )

        assert str(tmp_path) in seen["locations"]

    def test_falls_back_to_use_when_the_local_clone_is_missing(self, tmp_path: Path) -> None:
        def _fabric(locations, silent):
            raise FileNotFoundError("no local clone")

        with pytest.warns(RuntimeWarning, match="falling back to use"):
            Corpus.load(
                fabric_class=_fabric,
                use_fn=lambda *a, **k: SimpleNamespace(api=_api()),
                env={"TEHILLIM_BHSA_PATH": str(tmp_path / "absent")},
                required_features="otype book",
            )

    def test_raises_only_when_both_the_local_clone_and_use_fail(self, tmp_path: Path) -> None:
        def _fabric(locations, silent):
            raise FileNotFoundError("no local clone")

        with (
            pytest.raises(RuntimeError, match="failed to load BHSA"),
            pytest.warns(RuntimeWarning),
        ):
            Corpus.load(
                fabric_class=_fabric,
                use_fn=lambda *a, **k: None,
                env={"TEHILLIM_BHSA_PATH": str(tmp_path / "absent")},
                required_features="otype book",
            )

    def test_does_not_wait_for_a_slow_use_call_past_the_timeout(self, tmp_path: Path) -> None:
        import time

        def _fabric(locations, silent):
            raise FileNotFoundError("no local clone")

        def _slow_use(*args, **kwargs):
            time.sleep(5)
            return SimpleNamespace(api=_api())

        started = time.monotonic()
        with (
            pytest.raises(RuntimeError, match="failed to load BHSA"),
            pytest.warns(RuntimeWarning),
        ):
            Corpus.load(
                fabric_class=_fabric,
                use_fn=_slow_use,
                env={"TEHILLIM_BHSA_PATH": str(tmp_path / "absent")},
                required_features="otype book",
                timeout_seconds=0.2,
            )

        assert time.monotonic() - started < 3


class TestNonApiLocalResults:
    def test_falls_back_when_the_local_clone_returns_false(self, tmp_path: Path) -> None:
        """A failed Text-Fabric load is reported as False, which must not be taken as success."""

        def _fabric(locations, silent):
            return SimpleNamespace(loadAll=lambda silent: False)

        with pytest.warns(RuntimeWarning, match="falling back to use"):
            corpus = Corpus.load(
                fabric_class=_fabric,
                use_fn=lambda *a, **k: SimpleNamespace(api=_api()),
                env={"TEHILLIM_BHSA_PATH": str(tmp_path)},
                required_features="otype book",
            )

        assert corpus is not None

    def test_raises_when_both_sides_return_a_non_api(self, tmp_path: Path) -> None:
        def _fabric(locations, silent):
            return SimpleNamespace(loadAll=lambda silent: False)

        with (
            pytest.raises(RuntimeError, match="failed to load BHSA"),
            pytest.warns(RuntimeWarning),
        ):
            Corpus.load(
                fabric_class=_fabric,
                use_fn=lambda *a, **k: SimpleNamespace(api=False),
                env={"TEHILLIM_BHSA_PATH": str(tmp_path)},
                required_features="otype book",
            )
