"""Loads the ETCBC BHSA corpus via Text-Fabric into the psalm facts every consumer reads."""

from __future__ import annotations

import os
import queue
import threading
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Checkout spec used only when the local clone is unusable; override with TEHILLIM_CHECKOUT.
DEFAULT_CHECKOUT = "latest"
#: The local BHSA clone this repo prefers, overridden by $TEHILLIM_BHSA_PATH.
DEFAULT_BHSA_CLONE = Path.home() / "Developer" / "hebrew" / "bhsa" / "tf" / "2021"
BHSA_PATH_ENV = "TEHILLIM_BHSA_PATH"
#: use() re-verifies the release against GitHub even when cached, which can hang for minutes.
DEFAULT_USE_TIMEOUT_SECONDS = 30.0

#: The Text-Fabric features the psalm facts are read from.
_REQUIRED_FEATURES = "otype book chapter verse g_word_utf8 trailer_utf8"

_PSALMS_BOOK_NAME = "Psalmi"


def bhsa_clone_location(env: Mapping[str, str] | None = None) -> Path:
    """The local BHSA directory: $TEHILLIM_BHSA_PATH when set, else the conventional clone."""
    environment = os.environ if env is None else env
    return Path(environment.get(BHSA_PATH_ENV) or DEFAULT_BHSA_CLONE)


def _real_use(spec: str, checkout: str, silent: str) -> Any:
    """Returns a real Text-Fabric app downloaded through `use()`."""
    from tf.app import use

    return use(spec, checkout=checkout, silent=silent)


def _real_fabric(locations: list[str], silent: str) -> Any:
    """Returns a real Text-Fabric `Fabric` for `locations`."""
    from tf.fabric import Fabric

    return Fabric(locations=locations, silent=silent)


def _call_with_timeout(
    fn: Callable[..., Any], timeout_seconds: float, /, *args: Any, **kwargs: Any
) -> Any:
    """Runs fn in a daemon thread; returns None (without waiting further) past timeout_seconds."""
    result: queue.Queue[Any] = queue.Queue(maxsize=1)

    def _run() -> None:
        try:
            result.put(fn(*args, **kwargs))
        except Exception:  # noqa: BLE001 -- any failure means the call did not produce an api
            result.put(None)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)
    if thread.is_alive():
        return None
    return result.get_nowait()


def _as_api(result: Any) -> Any:
    """Text-Fabric reports failure as False and bare success as True, so demand a real api."""
    return result if getattr(result, "F", None) is not None else None


@dataclass(frozen=True, slots=True)
class Psalm:
    """A single psalm (one BHSA chapter within the book of Psalms)."""

    number: int
    verse_count: int
    word_count: int
    #: Vocalized Hebrew text of the psalm's first verse.
    incipit: str
    #: BHSA `half_verse` node ids in order, the rows every embeddings representation is read by.
    half_verse_nodes: tuple[int, ...] = ()


class Corpus:
    """A loaded Text-Fabric BHSA corpus, scoped to psalm extraction."""

    def __init__(self, api: Any) -> None:
        # Text-Fabric ships no type stubs.
        """Holds the Text-Fabric handles this corpus reads through."""
        self._api = api

    @classmethod
    def load(
        cls,
        checkout: str = DEFAULT_CHECKOUT,
        *,
        fabric_class: Callable[..., Any] = _real_fabric,
        use_fn: Callable[..., Any] = _real_use,
        env: Mapping[str, str] | None = None,
        required_features: str = _REQUIRED_FEATURES,
        timeout_seconds: float = DEFAULT_USE_TIMEOUT_SECONDS,
    ) -> Corpus:
        """Loads BHSA from the local clone, falling back to use()."""
        clone = bhsa_clone_location(env)
        try:
            api = _as_api(
                fabric_class(locations=[str(clone)], silent="deep").loadAll(silent="deep")
            )
        except Exception as error:  # noqa: BLE001 -- Text-Fabric raises anything; fall back
            api = None
            reason: object = error
        else:
            reason = "no usable Text-Fabric data there" if api is None else None
        if api is None:
            #: A silent fallback here looks identical to a cold cache and hides real loader bugs.
            warnings.warn(
                f"local BHSA clone at {clone} unusable ({reason!r}), falling back to use()",
                RuntimeWarning,
                stacklevel=2,
            )
            app = _call_with_timeout(use_fn, timeout_seconds, "etcbc/bhsa", checkout, "deep")
            api = _as_api(getattr(app, "api", None)) if app is not None else None
        if api is None:
            raise RuntimeError(
                f"Text-Fabric failed to load BHSA from {clone} or via use(checkout={checkout!r}). "
                "Set TEHILLIM_BHSA_PATH to a local clone or TEHILLIM_CHECKOUT to pin a version."
            )
        api.TF.load(required_features, add=True, silent="deep")
        missing = [f for f in required_features.split() if not hasattr(api.F, f)]
        if missing:
            raise RuntimeError(f"required BHSA features not loaded: {missing}")
        return cls(api)

    def psalms(self) -> list[Psalm]:
        """Extract all 150 psalms, in canonical order, as structured data."""
        F, L, T = self._api.F, self._api.L, self._api.T  # noqa: N806

        book_nodes = [b for b in F.otype.s("book") if F.book.v(b) == _PSALMS_BOOK_NAME]
        if not book_nodes:
            raise RuntimeError(f"Book '{_PSALMS_BOOK_NAME}' not found in loaded corpus")

        psalms: list[Psalm] = []
        for chapter_node in L.d(book_nodes[0], otype="chapter"):
            _, psalm_number = T.sectionFromNode(chapter_node)
            verse_nodes = L.d(chapter_node, otype="verse")
            incipit = T.text(L.d(verse_nodes[0], otype="word")).strip() if verse_nodes else ""
            psalms.append(
                Psalm(
                    number=psalm_number,
                    verse_count=len(verse_nodes),
                    word_count=len(L.d(chapter_node, otype="word")),
                    incipit=incipit,
                    half_verse_nodes=tuple(L.d(chapter_node, otype="half_verse")),
                )
            )

        psalms.sort(key=lambda p: p.number)
        return psalms
