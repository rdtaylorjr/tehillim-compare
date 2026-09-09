from __future__ import annotations

from tehillim_compare.corpus import Psalm, PsalmWord
from tehillim_compare.verb_sense_profile import build_verb_sense_feature_matrix, verb_sense_words


def _word(
    pos: str = "verb", verb_sense: str = "", lexeme: str = "X", gloss: str = "", node: int = 0
) -> PsalmWord:
    return PsalmWord(
        node=node,
        lexeme=lexeme,
        lemma=lexeme,
        surface=lexeme,
        part_of_speech=pos,
        gloss=gloss,
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
        phrase_valence="",
        phrase_grammatical_role="",
        verb_sense=verb_sense,
    )


def test_verb_sense_words_excludes_words_with_no_sense_code():
    words = (_word(verb_sense="d-"), _word())
    assert len(verb_sense_words(words)) == 1


def test_build_matrix_counts_sense_tags():
    psalms = [
        Psalm(
            number=1,
            verse_count=1,
            incipit="",
            words=(_word(verb_sense="d-"), _word(verb_sense="d-"), _word(verb_sense="-p")),
        ),
    ]
    fm = build_verb_sense_feature_matrix(psalms)
    assert set(fm.terms) == {"d-", "-p"}
    assert fm.counts[0, fm.terms.index("d-")] == 2
    assert fm.counts[0, fm.terms.index("-p")] == 1


def test_term_info_uses_the_code_itself_as_the_label():
    word = _word(verb_sense="d-", gloss="go")
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(word,))]
    fm = build_verb_sense_feature_matrix(psalms)
    assert fm.term_info["d-"].label == "d-"
    assert fm.term_info["d-"].description == "go"


def test_build_matrix_handles_psalm_with_no_verb_sense_words():
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(_word(),))]
    fm = build_verb_sense_feature_matrix(psalms)
    assert fm.terms == ()
    assert fm.counts.shape == (1, 0)


def test_build_matrix_handles_empty_psalm_list():
    fm = build_verb_sense_feature_matrix([])
    assert fm.psalm_numbers == ()
    assert fm.counts.shape == (0, 0)


def test_build_matrix_preserves_psalm_order_and_numbers():
    psalms = [
        Psalm(number=5, verse_count=1, incipit="", words=(_word(verb_sense="d-"),)),
        Psalm(number=2, verse_count=1, incipit="", words=(_word(verb_sense="-p"),)),
    ]
    fm = build_verb_sense_feature_matrix(psalms)
    assert fm.psalm_numbers == (5, 2)
