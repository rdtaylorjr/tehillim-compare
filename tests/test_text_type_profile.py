from __future__ import annotations

from tehillim_compare.corpus import Psalm, PsalmWord
from tehillim_compare.text_type_profile import build_text_type_feature_matrix, text_type_words


def _word(pos: str = "subs", text_type: str = "", lexeme: str = "X", node: int = 0) -> PsalmWord:
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
        text_type=text_type,
        clause_relation="",
        clause_kind="",
        phrase_function="",
        phrase_determination="",
        phrase_type="",
        phrase_valence="",
        phrase_grammatical_role="",
        verb_sense="",
    )


def test_text_type_words_excludes_words_with_no_text_type():
    words = (_word(text_type="Q"), _word())
    assert len(text_type_words(words)) == 1


def test_build_matrix_counts_text_type_tags():
    psalms = [
        Psalm(
            number=1,
            verse_count=1,
            incipit="",
            words=(_word(text_type="Q"), _word(text_type="Q"), _word(text_type="N")),
        ),
    ]
    fm = build_text_type_feature_matrix(psalms)
    assert set(fm.terms) == {"Q", "N"}
    assert fm.counts[0, fm.terms.index("Q")] == 2
    assert fm.counts[0, fm.terms.index("N")] == 1


def test_single_character_code_gets_its_plain_label():
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(_word(text_type="Q"),))]
    fm = build_text_type_feature_matrix(psalms)
    assert fm.term_info["Q"].description == "Quotation"


def test_nested_code_gets_a_chained_description():
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(_word(text_type="QND"),))]
    fm = build_text_type_feature_matrix(psalms)
    assert fm.term_info["QND"].description == "Quotation within Narrative within Discursive"
    assert fm.term_info["QND"].label == "QND"


def test_build_matrix_handles_psalm_with_no_text_type_words():
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(_word(),))]
    fm = build_text_type_feature_matrix(psalms)
    assert fm.terms == ()
    assert fm.counts.shape == (1, 0)


def test_build_matrix_handles_empty_psalm_list():
    fm = build_text_type_feature_matrix([])
    assert fm.psalm_numbers == ()
    assert fm.counts.shape == (0, 0)


def test_build_matrix_preserves_psalm_order_and_numbers():
    psalms = [
        Psalm(number=5, verse_count=1, incipit="", words=(_word(text_type="Q"),)),
        Psalm(number=2, verse_count=1, incipit="", words=(_word(text_type="N"),)),
    ]
    fm = build_text_type_feature_matrix(psalms)
    assert fm.psalm_numbers == (5, 2)
