from __future__ import annotations

from tehillim_compare.corpus import Psalm, PsalmWord
from tehillim_compare.phrase_determination_profile import (
    build_phrase_determination_feature_matrix,
    determination_words,
)


def _word(
    pos: str = "subs", phrase_determination: str = "", lexeme: str = "X", node: int = 0
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
        phrase_determination=phrase_determination,
        phrase_type="",
        phrase_valence="",
        phrase_grammatical_role="",
        verb_sense="",
    )


def test_determination_words_excludes_words_with_no_determination():
    words = (_word(phrase_determination="det"), _word())
    assert len(determination_words(words)) == 1


def test_build_matrix_counts_determination_tags():
    psalms = [
        Psalm(
            number=1,
            verse_count=1,
            incipit="",
            words=(
                _word(phrase_determination="det"),
                _word(phrase_determination="det"),
                _word(phrase_determination="und"),
            ),
        ),
    ]
    fm = build_phrase_determination_feature_matrix(psalms)
    assert set(fm.terms) == {"det", "und"}
    assert fm.counts[0, fm.terms.index("det")] == 2
    assert fm.counts[0, fm.terms.index("und")] == 1


def test_term_info_has_human_readable_labels():
    psalms = [
        Psalm(number=1, verse_count=1, incipit="", words=(_word(phrase_determination="det"),))
    ]
    fm = build_phrase_determination_feature_matrix(psalms)
    assert fm.term_info["det"].label == "Determined"


def test_build_matrix_handles_psalm_with_no_determination_words():
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(_word(),))]
    fm = build_phrase_determination_feature_matrix(psalms)
    assert fm.terms == ()
    assert fm.counts.shape == (1, 0)


def test_build_matrix_handles_empty_psalm_list():
    fm = build_phrase_determination_feature_matrix([])
    assert fm.psalm_numbers == ()
    assert fm.counts.shape == (0, 0)


def test_build_matrix_preserves_psalm_order_and_numbers():
    psalms = [
        Psalm(number=5, verse_count=1, incipit="", words=(_word(phrase_determination="det"),)),
        Psalm(number=2, verse_count=1, incipit="", words=(_word(phrase_determination="und"),)),
    ]
    fm = build_phrase_determination_feature_matrix(psalms)
    assert fm.psalm_numbers == (5, 2)
