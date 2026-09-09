from __future__ import annotations

from tehillim_compare.corpus import Psalm, PsalmWord
from tehillim_compare.lexical_set import build_lexical_set_feature_matrix, lexical_set_words


def _word(pos: str = "subs", lexical_set: str = "", lexeme: str = "X", node: int = 0) -> PsalmWord:
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
        lexical_set=lexical_set,
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
        phrase_valence="",
        phrase_grammatical_role="",
        verb_sense="",
    )


def test_lexical_set_words_excludes_words_with_no_lexical_set():
    words = (_word(lexical_set="card"), _word())
    assert len(lexical_set_words(words)) == 1


def test_build_matrix_counts_lexical_set_tags():
    psalms = [
        Psalm(
            number=1,
            verse_count=1,
            incipit="",
            words=(_word(lexical_set="card"), _word(lexical_set="card"), _word(lexical_set="ordn")),
        ),
    ]
    fm = build_lexical_set_feature_matrix(psalms)
    assert set(fm.terms) == {"card", "ordn"}
    assert fm.counts[0, fm.terms.index("card")] == 2
    assert fm.counts[0, fm.terms.index("ordn")] == 1


def test_term_info_has_human_readable_labels():
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(_word(lexical_set="gntl"),))]
    fm = build_lexical_set_feature_matrix(psalms)
    assert fm.term_info["gntl"].label == "Gentilic"


def test_unknown_lexical_set_code_falls_back_to_the_raw_code():
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(_word(lexical_set="zzzz"),))]
    fm = build_lexical_set_feature_matrix(psalms)
    assert fm.term_info["zzzz"].label == "zzzz"


def test_build_matrix_handles_psalm_with_no_lexical_set_words():
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(_word(),))]
    fm = build_lexical_set_feature_matrix(psalms)
    assert fm.terms == ()
    assert fm.counts.shape == (1, 0)


def test_build_matrix_handles_empty_psalm_list():
    fm = build_lexical_set_feature_matrix([])
    assert fm.psalm_numbers == ()
    assert fm.counts.shape == (0, 0)
