from __future__ import annotations

from tehillim_compare.clause_kind_profile import (
    build_clause_kind_feature_matrix,
    clause_kind_words,
)
from tehillim_compare.corpus import Psalm, PsalmWord


def _word(pos: str = "subs", clause_kind: str = "", lexeme: str = "X", node: int = 0) -> PsalmWord:
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
        clause_kind=clause_kind,
        phrase_function="",
        phrase_determination="",
        phrase_type="",
        phrase_valence="",
        phrase_grammatical_role="",
        verb_sense="",
    )


def test_clause_kind_words_excludes_words_with_no_clause_kind():
    words = (_word(clause_kind="VC"), _word())
    assert len(clause_kind_words(words)) == 1


def test_build_matrix_counts_clause_kind_tags():
    psalms = [
        Psalm(
            number=1,
            verse_count=1,
            incipit="",
            words=(_word(clause_kind="VC"), _word(clause_kind="VC"), _word(clause_kind="NC")),
        ),
    ]
    fm = build_clause_kind_feature_matrix(psalms)
    assert set(fm.terms) == {"VC", "NC"}
    assert fm.counts[0, fm.terms.index("VC")] == 2
    assert fm.counts[0, fm.terms.index("NC")] == 1


def test_term_info_has_human_readable_labels():
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(_word(clause_kind="WP"),))]
    fm = build_clause_kind_feature_matrix(psalms)
    assert fm.term_info["WP"].label == "Clause Without Predication"


def test_build_matrix_handles_psalm_with_no_clause_kind_words():
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(_word(),))]
    fm = build_clause_kind_feature_matrix(psalms)
    assert fm.terms == ()
    assert fm.counts.shape == (1, 0)


def test_build_matrix_handles_empty_psalm_list():
    fm = build_clause_kind_feature_matrix([])
    assert fm.psalm_numbers == ()
    assert fm.counts.shape == (0, 0)


def test_build_matrix_preserves_psalm_order_and_numbers():
    psalms = [
        Psalm(number=5, verse_count=1, incipit="", words=(_word(clause_kind="VC"),)),
        Psalm(number=2, verse_count=1, incipit="", words=(_word(clause_kind="NC"),)),
    ]
    fm = build_clause_kind_feature_matrix(psalms)
    assert fm.psalm_numbers == (5, 2)
