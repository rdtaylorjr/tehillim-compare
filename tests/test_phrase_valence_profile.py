from __future__ import annotations

from tehillim_compare.corpus import Psalm, PsalmWord
from tehillim_compare.phrase_valence_profile import (
    build_phrase_valence_feature_matrix,
    valence_words,
)


def _word(
    pos: str = "subs", phrase_valence: str = "", lexeme: str = "X", node: int = 0
) -> PsalmWord:
    return PsalmWord(
        node=node,
        lexeme=lexeme,
        lemma=lexeme,
        surface=lexeme,
        part_of_speech=pos,
        gloss="",
        verb_stem="",
        verb_mood="",
        person="",
        number="",
        suffix_person="",
        suffix_number="",
        gender="",
        suffix_gender="",
        state="",
        lexical_set="",
        phrase_dependent_pos=pos,
        name_type="",
        root="",
        clause_type="",
        text_type="",
        clause_relation="",
        clause_kind="",
        phrase_function="",
        phrase_determination="",
        phrase_type="",
        phrase_valence=phrase_valence,
        phrase_grammatical_role="",
        verb_sense="",
    )


def test_valence_words_excludes_words_with_no_valence():
    words = (_word(phrase_valence="core"), _word())
    assert len(valence_words(words)) == 1


def test_build_matrix_counts_valence_tags():
    psalms = [
        Psalm(
            number=1,
            verse_count=1,
            incipit="",
            words=(
                _word(phrase_valence="core"),
                _word(phrase_valence="core"),
                _word(phrase_valence="adjunct"),
            ),
        ),
    ]
    fm = build_phrase_valence_feature_matrix(psalms)
    assert set(fm.terms) == {"core", "adjunct"}
    assert fm.counts[0, fm.terms.index("core")] == 2
    assert fm.counts[0, fm.terms.index("adjunct")] == 1


def test_term_info_has_human_readable_labels():
    word = _word(phrase_valence="complement")
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(word,))]
    fm = build_phrase_valence_feature_matrix(psalms)
    assert fm.term_info["complement"].label == "Complement"


def test_build_matrix_handles_psalm_with_no_valence_words():
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(_word(),))]
    fm = build_phrase_valence_feature_matrix(psalms)
    assert fm.terms == ()
    assert fm.counts.shape == (1, 0)


def test_build_matrix_handles_empty_psalm_list():
    fm = build_phrase_valence_feature_matrix([])
    assert fm.psalm_numbers == ()
    assert fm.counts.shape == (0, 0)


def test_build_matrix_preserves_psalm_order_and_numbers():
    psalms = [
        Psalm(number=5, verse_count=1, incipit="", words=(_word(phrase_valence="core"),)),
        Psalm(number=2, verse_count=1, incipit="", words=(_word(phrase_valence="adjunct"),)),
    ]
    fm = build_phrase_valence_feature_matrix(psalms)
    assert fm.psalm_numbers == (5, 2)
