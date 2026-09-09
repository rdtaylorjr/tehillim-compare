from __future__ import annotations

from tehillim_compare.clause_relation_profile import (
    build_clause_relation_feature_matrix,
    clause_relation_words,
)
from tehillim_compare.corpus import Psalm, PsalmWord


def _word(
    pos: str = "subs", clause_relation: str = "", lexeme: str = "X", node: int = 0
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
        clause_relation=clause_relation,
        clause_kind="",
        phrase_function="",
        phrase_determination="",
        phrase_type="",
        phrase_valence="",
        phrase_grammatical_role="",
        verb_sense="",
    )


def test_clause_relation_words_excludes_words_with_no_clause_relation():
    words = (_word(clause_relation="Coor"), _word())
    assert len(clause_relation_words(words)) == 1


def test_build_matrix_counts_clause_relation_tags():
    psalms = [
        Psalm(
            number=1,
            verse_count=1,
            incipit="",
            words=(
                _word(clause_relation="Coor"),
                _word(clause_relation="Coor"),
                _word(clause_relation="Attr"),
            ),
        ),
    ]
    fm = build_clause_relation_feature_matrix(psalms)
    assert set(fm.terms) == {"Coor", "Attr"}
    assert fm.counts[0, fm.terms.index("Coor")] == 2
    assert fm.counts[0, fm.terms.index("Attr")] == 1


def test_term_info_has_human_readable_labels():
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(_word(clause_relation="Coor"),))]
    fm = build_clause_relation_feature_matrix(psalms)
    assert fm.term_info["Coor"].label == "Coordinated Clause"


def test_unknown_clause_relation_code_falls_back_to_the_raw_code():
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(_word(clause_relation="zzzz"),))]
    fm = build_clause_relation_feature_matrix(psalms)
    assert fm.term_info["zzzz"].label == "zzzz"


def test_build_matrix_handles_psalm_with_no_clause_relation_words():
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(_word(),))]
    fm = build_clause_relation_feature_matrix(psalms)
    assert fm.terms == ()
    assert fm.counts.shape == (1, 0)


def test_build_matrix_handles_empty_psalm_list():
    fm = build_clause_relation_feature_matrix([])
    assert fm.psalm_numbers == ()
    assert fm.counts.shape == (0, 0)


def test_build_matrix_preserves_psalm_order_and_numbers():
    psalms = [
        Psalm(number=5, verse_count=1, incipit="", words=(_word(clause_relation="Coor"),)),
        Psalm(number=2, verse_count=1, incipit="", words=(_word(clause_relation="Attr"),)),
    ]
    fm = build_clause_relation_feature_matrix(psalms)
    assert fm.psalm_numbers == (5, 2)
