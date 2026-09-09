from __future__ import annotations

from tehillim_compare.corpus import Psalm, PsalmWord
from tehillim_compare.named_entity_profile import (
    build_named_entity_feature_matrix,
    name_type_words,
)


def _word(pos: str = "nmpr", name_type: str = "", lexeme: str = "X", node: int = 0) -> PsalmWord:
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
        name_type=name_type,
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


def test_name_type_words_excludes_words_with_no_name_type():
    words = (_word(name_type="pers"), _word())
    assert len(name_type_words(words)) == 1


def test_build_matrix_counts_name_type_tags():
    psalms = [
        Psalm(
            number=1,
            verse_count=1,
            incipit="",
            words=(_word(name_type="pers"), _word(name_type="pers"), _word(name_type="topo")),
        ),
    ]
    fm = build_named_entity_feature_matrix(psalms)
    assert set(fm.terms) == {"pers", "topo"}
    assert fm.counts[0, fm.terms.index("pers")] == 2
    assert fm.counts[0, fm.terms.index("topo")] == 1


def test_term_info_has_human_readable_labels():
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(_word(name_type="topo"),))]
    fm = build_named_entity_feature_matrix(psalms)
    assert fm.term_info["topo"].label == "Place"


def test_comma_joined_name_type_gets_a_combined_label():
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(_word(name_type="pers,gens"),))]
    fm = build_named_entity_feature_matrix(psalms)
    assert fm.terms == ("pers,gens",)
    assert fm.term_info["pers,gens"].label == "Person / People"


def test_unknown_name_type_code_falls_back_to_the_raw_code():
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(_word(name_type="zzzz"),))]
    fm = build_named_entity_feature_matrix(psalms)
    assert fm.term_info["zzzz"].label == "zzzz"


def test_build_matrix_handles_psalm_with_no_name_type_words():
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(_word(),))]
    fm = build_named_entity_feature_matrix(psalms)
    assert fm.terms == ()
    assert fm.counts.shape == (1, 0)


def test_build_matrix_handles_empty_psalm_list():
    fm = build_named_entity_feature_matrix([])
    assert fm.psalm_numbers == ()
    assert fm.counts.shape == (0, 0)
