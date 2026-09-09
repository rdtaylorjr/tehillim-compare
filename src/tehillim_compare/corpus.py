"""Loads the ETCBC BHSA corpus via Text-Fabric into plain, serializable Python objects."""

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
_MOD_CACHE_ROOT = Path.home() / "text-fabric-data" / "github"

_VALENCE_MOD = "etcbc/valence/tf"


def bhsa_clone_location(env: Mapping[str, str] | None = None) -> Path:
    """The local BHSA directory: $TEHILLIM_BHSA_PATH when set, else the conventional clone."""
    environment = os.environ if env is None else env
    return Path(environment.get(BHSA_PATH_ENV) or DEFAULT_BHSA_CLONE)


def _real_use(spec: str, checkout: str, mod: str, silent: str) -> Any:
    """Returns a real Text-Fabric app downloaded through `use()`."""
    from tf.app import use

    return use(spec, checkout=checkout, mod=mod, silent=silent)


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


def _mod_cache_location(mod: str) -> Path:
    """Resolves an "org/repo/tf" mod spec to its cached local data directory."""
    org, repo, _tf = mod.split(":", 1)[0].split("/")
    tf_root = _MOD_CACHE_ROOT / org / repo / "tf"
    versions = sorted(p for p in tf_root.iterdir() if p.is_dir()) if tf_root.exists() else []
    if not versions:
        raise RuntimeError(f"No cached Text-Fabric data found for {org}/{repo} under {tf_root}")
    return versions[-1]


#: Text-Fabric features required for extraction.
_REQUIRED_FEATURES = (
    "otype book chapter verse "
    "lex voc_lex_utf8 g_word_utf8 g_cons_utf8 trailer_utf8 sp gloss vs vt ps nu prs_ps prs_nu "
    "gn prs_gn st ls pdp nametype root "
    "typ txt rela kind function det "
    "valence grammatical sense"
)

#: BHSA's registered Text-Fabric format for consonantal-only text (no niqqud/vowel points).
_UNVOCALIZED_FORMAT = "text-orig-plain"

_PSALMS_BOOK_NAME = "Psalmi"

#: Unicode range of Hebrew cantillation/accent marks (te'amim).
_ACCENT_RANGE = range(0x0591, 0x05B0)


def _strip_accents(text: str) -> str:
    """Removes cantillation/accent marks (see `_ACCENT_RANGE`) while keeping niqqud."""
    return "".join(ch for ch in text if ord(ch) not in _ACCENT_RANGE)


@dataclass(frozen=True, slots=True)
class PsalmWord:
    """A single word occurrence within a psalm."""

    node: int
    """Text-Fabric word node id. Kept so later phases (clause/phrase
    structure, disjunctive accents, ...) can re-query the corpus for a word
    without re-running extraction from scratch."""

    lexeme: str
    """Stable ETCBC lexeme identifier (e.g. ``JHWH/``). Used as the unit of
    lexical comparison since it collapses inflected surface forms."""

    lemma: str
    """Vocalized Hebrew dictionary form of the lexeme, for display."""

    surface: str
    """Vocalized Hebrew surface form as it occurs in the text."""

    part_of_speech: str
    """BHSA part-of-speech code (``subs``, ``verb``, ``nmpr``, ...)."""

    gloss: str
    """Short English gloss of the lexeme."""

    verb_stem: str
    """BHSA verb stem/binyan code (e.g. ``piel``, ``hif``), or "" if this
    word is not a verb."""

    verb_mood: str
    """BHSA verbal-tense/conjugation code (``vt``: e.g. ``impv``, ``impf``,
    or the non-finite ``ptca``/``ptcp`` participles), or "" if this word is
    not a verb. Named `verb_mood` for historical reasons. The value domain
    is BHSA's conjugation system, not grammatical mood in the strict
    linguistic sense."""

    person: str
    """BHSA grammatical person of the word itself (``p1``/``p2``/``p3``, or
    ``unknown`` if marked but ambiguous), or "" if not person-marked."""

    number: str
    """BHSA grammatical number of the word itself (``sg``/``pl``/``du``, or
    ``unknown``), or "" if not number-marked. General word-level feature,
    not exclusive to person-marked words. E.g. plain plural nouns have a
    number but no person."""

    suffix_person: str
    """Person of the word's pronominal suffix (e.g. the "my" in "my God"),
    or "" if the word has no pronominal suffix."""

    suffix_number: str
    """Number of the word's pronominal suffix, or "" if none."""

    gender: str
    """BHSA grammatical gender of the word itself (``m``/``f``, or
    ``unknown``), or "" if not gender-marked."""

    suffix_gender: str
    """Gender of the word's pronominal suffix, or "" if none."""

    state: str
    """BHSA nominal state (``c`` construct / ``a`` absolute), or "" if not
    applicable (verbs, particles, ...). Construct-chain density is a
    register marker independent of person/verb morphology."""

    lexical_set: str
    """BHSA lexical set, a finer subcategory than part-of-speech (e.g.
    ``nmdi`` demonstrative, ``ppre`` preposition-as-noun, ``padv`` adverbial
    particle), or "" if the word has no lexical-set subcategory."""

    phrase_dependent_pos: str
    """BHSA part-of-speech as used in this word's specific phrase context
    (e.g. an adjective substantivized to function as a noun). Differs from
    ``part_of_speech`` for about 5.6% of Psalter words, a real, distinct
    syntactic-function signal, not a duplicate of ``sp``."""

    name_type: str
    """BHSA named-entity type (``pers``, ``topo``, ``gens``, or comma-joined
    combinations), or "" if the word is not a proper name."""

    root: str
    """BHSA triliteral consonantal root, collapsing derivationally related
    lexemes (e.g. a verb and its cognate noun) that ``lexeme`` keeps
    distinct. Only populated for a subset of content words, "" otherwise."""

    clause_type: str
    """BHSA clause-atom type of the word's enclosing clause (e.g. ``Way0``
    wayyiqtol-null, ``NmCl`` nominal clause), a fine-grained classification
    of clause structure, denormalized from the clause down onto each of its
    words so it composes with the same per-word tag-counting machinery as
    every other feature."""

    text_type: str
    """BHSA text type of the word's enclosing clause: ``N`` narrative,
    ``D`` discursive, ``Q`` quotation, or ``?`` unknown, with embedding
    shown by repetition (e.g. ``QND`` = discursive within narrative within
    quotation). A quotation-heavy psalm reads differently from a
    narrative-heavy one."""

    clause_relation: str
    """BHSA syntactic relation of the word's enclosing clause to its
    context (e.g. ``Coor`` coordinated, ``Attr`` attributive, ``Objc``
    object clause), or "" if not applicable (most clauses)."""

    clause_kind: str
    """BHSA coarse clause kind derived from ``clause_type``: ``VC`` verbal,
    ``NC`` nominal, or ``WP`` without predication."""

    phrase_function: str
    """BHSA syntactic function of the word's enclosing phrase (e.g.
    ``Pred`` predicate, ``Subj`` subject, ``Voct`` vocative)."""

    phrase_determination: str
    """BHSA determination of the word's enclosing phrase: ``det``
    determined, ``und`` undetermined, or "" if not applicable."""

    phrase_type: str
    """BHSA phrase-atom type of the word's enclosing phrase (e.g. ``VP``
    verbal phrase, ``PP`` prepositional phrase, ``PrNP`` proper-noun
    phrase), the phrase-level counterpart of ``clause_type``, sharing the
    same underlying `typ` feature but scoped to a different object type."""

    phrase_valence: str
    """ETCBC/valence classification of the word's enclosing phrase as a
    verbal argument: ``core`` (core argument), ``complement``, or
    ``adjunct`` (peripheral). From a companion Text-Fabric module layered
    on BHSA, not core BHSA itself, "" if unannotated."""

    phrase_grammatical_role: str
    """ETCBC/valence fine-grained constituent role of the word's enclosing
    phrase (e.g. ``direct_object``, ``subject``, ``indirect_object``,
    ``L_object``), a finer-grained sibling of ``phrase_function``. From
    the same companion module as ``phrase_valence``, "" if unannotated."""

    verb_sense: str
    """ETCBC/valence sense/argument-realization code for this word if it is
    a verb occurrence (e.g. ``d-`` takes a direct object, ``-p`` takes a
    prepositional complement), or "" otherwise. Distinct from
    ``verb_stem``/``verb_mood``: this describes complementation pattern,
    not morphology."""


@dataclass(frozen=True, slots=True)
class Psalm:
    """A single psalm (one BHSA chapter within the book of Psalms)."""

    number: int
    verse_count: int
    words: tuple[PsalmWord, ...]
    incipit: str
    """Vocalized Hebrew text of the psalm's first verse."""

    half_verses: tuple[str, ...] = ()
    """Vocalized Hebrew text of each of the psalm's half-verses, in order,
    via BHSA's `half_verse` sectional otype (the Masoretic verse-
    internal division, e.g. at the atnach), not a heuristic split. This is
    the granularity semantic_embedding.py's MiqraBERT-based signal uses,
    matching the unit Smiley's training pairs were built from, rather
    than mean-pooling a whole verse (the pooling granularity his paper
    identifies as the specific cause of MiqraBERT's poor poetic recall)."""

    half_verses_unvocalized: tuple[str, ...] = ()
    """Consonantal-only Hebrew text (no niqqud) of the same half-verses as
    `half_verses`, same order, one-to-one. BHSA's `g_cons_utf8` feature
    via the `text-orig-plain` format, not a Unicode diacritic strip. Built
    to test whether AlephBERT/MiqraBERT's genre-clustering signal changes
    with unvocalized input. It doesn't: both models' shared tokenizer
    (inherited unchanged from the 2021 AlephBERT release) strips Unicode
    combining marks, including niqqud, before either model ever sees the
    text (see semantic_embedding.py's `MIQRABERT_MODEL` comment and
    `test_semantic_embedding_integration.py`), so `half_verses` and
    `half_verses_unvocalized` are functionally the same input to those two
    models specifically. Kept as a general-purpose field regardless -
    correct, independently useful, and not tied to that one finding."""

    half_verses_niqqud_only: tuple[str, ...] = ()
    """Niqqud-preserved, accent-stripped Hebrew text of the same half-verses
    as `half_verses` (see `_strip_accents`), a third text variant,
    distinct from both `half_verses` (niqqud + cantillation accents) and
    `half_verses_unvocalized` (neither). Real ablation axis only for the
    SentencePiece/BPE-family encoders that tokenize accents as distinct
    content from niqqud (see semantic_embedding.py's module docstring)."""

    half_verse_nodes: tuple[int, ...] = ()
    """Real BHSA Text-Fabric node ids of the same half-verses as
    `half_verses`, same order, one-to-one. The actual `half_verse` otype
    node numbers `half_verses`/`half_verses_unvocalized`/
    `half_verses_niqqud_only` were all rendered from via `T.text()`, kept
    here so downstream consumers (e.g. a companion Text-Fabric feature
    module distributing this project's embeddings) can key its data
    to BHSA's real node numbering instead of psalm/half-verse-index pairs,
    matching every other ETCBC companion module's convention."""


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
        mod_cache_location_fn: Callable[[str], Path] = _mod_cache_location,
    ) -> Corpus:
        """Loads BHSA plus etcbc/valence from the local clone, falling back to use()."""
        clone = bhsa_clone_location(env)
        try:
            locations = [str(clone), str(mod_cache_location_fn(_VALENCE_MOD))]
            api = _as_api(fabric_class(locations=locations, silent="deep").loadAll(silent="deep"))
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
            mod = f"{_VALENCE_MOD}:{checkout}"
            app = _call_with_timeout(use_fn, timeout_seconds, "etcbc/bhsa", checkout, mod, "deep")
            api = _as_api(getattr(app, "api", None)) if app is not None else None
        if api is None:
            raise RuntimeError(
                f"Text-Fabric failed to load BHSA plus etcbc/valence from {clone} or via "
                f"use(checkout={checkout!r}). Set TEHILLIM_BHSA_PATH to a local clone, GHPERS "
                "to a GitHub token for private modules, or TEHILLIM_CHECKOUT to pin a version."
            )
        api.TF.load(required_features, add=True, silent="deep")
        missing = [f for f in required_features.split() if not hasattr(api.F, f)]
        if missing:
            raise RuntimeError(f"required BHSA/valence features not loaded: {missing}")
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
            word_nodes = L.d(chapter_node, otype="word")
            words = tuple(self._word(node) for node in word_nodes)

            verse_nodes = L.d(chapter_node, otype="verse")
            incipit = T.text(L.d(verse_nodes[0], otype="word")).strip() if verse_nodes else ""

            half_verse_nodes = L.d(chapter_node, otype="half_verse")
            half_verses = tuple(T.text(L.d(hv, otype="word")).strip() for hv in half_verse_nodes)
            half_verses_unvocalized = tuple(
                T.text(L.d(hv, otype="word"), fmt=_UNVOCALIZED_FORMAT).strip()
                for hv in half_verse_nodes
            )
            half_verses_niqqud_only = tuple(_strip_accents(hv) for hv in half_verses)

            psalms.append(
                Psalm(
                    number=psalm_number,
                    verse_count=len(verse_nodes),
                    words=words,
                    incipit=incipit,
                    half_verses=half_verses,
                    half_verses_unvocalized=half_verses_unvocalized,
                    half_verses_niqqud_only=half_verses_niqqud_only,
                    half_verse_nodes=tuple(half_verse_nodes),
                )
            )

        psalms.sort(key=lambda p: p.number)
        return psalms

    def _word(self, node: int) -> PsalmWord:
        F, L = self._api.F, self._api.L  # noqa: N806
        lex_node = L.u(node, otype="lex")
        gloss = F.gloss.v(lex_node[0]) if lex_node else ""

        clause_nodes = L.u(node, otype="clause")
        clause = clause_nodes[0] if clause_nodes else None
        phrase_nodes = L.u(node, otype="phrase")
        phrase = phrase_nodes[0] if phrase_nodes else None

        return PsalmWord(
            node=node,
            lexeme=F.lex.v(node),
            lemma=F.voc_lex_utf8.v(node),
            surface=F.g_word_utf8.v(node),
            part_of_speech=F.sp.v(node),
            gloss=gloss or "",
            verb_stem=_na_to_empty(F.vs.v(node)),
            verb_mood=_na_to_empty(F.vt.v(node)),
            person=_na_to_empty(F.ps.v(node)),
            number=_na_to_empty(F.nu.v(node)),
            suffix_person=_na_to_empty(F.prs_ps.v(node)),
            suffix_number=_na_to_empty(F.prs_nu.v(node)),
            gender=_na_to_empty(F.gn.v(node)),
            suffix_gender=_na_to_empty(F.prs_gn.v(node)),
            state=_na_to_empty(F.st.v(node)),
            lexical_set=_na_to_empty(F.ls.v(node), sentinel="none"),
            phrase_dependent_pos=F.pdp.v(node) or "",
            name_type=_na_to_empty(F.nametype.v(node)),
            root=_na_to_empty(F.root.v(node)),
            clause_type=_na_to_empty(F.typ.v(clause)) if clause else "",
            text_type=_na_to_empty(F.txt.v(clause)) if clause else "",
            clause_relation=_na_to_empty(F.rela.v(clause)) if clause else "",
            clause_kind=_na_to_empty(F.kind.v(clause)) if clause else "",
            phrase_function=_na_to_empty(F.function.v(phrase)) if phrase else "",
            phrase_determination=_na_to_empty(F.det.v(phrase)) if phrase else "",
            phrase_type=_na_to_empty(F.typ.v(phrase)) if phrase else "",
            phrase_valence=_na_to_empty(F.valence.v(phrase)) if phrase else "",
            phrase_grammatical_role=_na_to_empty(F.grammatical.v(phrase)) if phrase else "",
            verb_sense=_na_to_empty(F.sense.v(node)),
        )


def _na_to_empty(value: str | None, *, sentinel: str = "NA") -> str:
    """BHSA uses a literal sentinel string, usually "NA", but "none" for `ls`, rather than None."""
    return "" if value is None or value == sentinel else value
