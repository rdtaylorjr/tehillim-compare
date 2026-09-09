from __future__ import annotations

from tehillim_compare.corpus import Psalm, PsalmWord
from tehillim_compare.phrase_function_profile import build_phrase_function_feature_matrix


def _word(
    pos: str = "subs", phrase_function: str | None = None, lexeme: str = "X", node: int = 0
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
        phrase_function=phrase_function if phrase_function is not None else "Subj",
        phrase_determination="",
        phrase_type="",
        phrase_valence="",
        phrase_grammatical_role="",
        verb_sense="",
    )


def test_build_matrix_counts_phrase_function_tags():
    psalms = [
        Psalm(
            number=1,
            verse_count=1,
            incipit="",
            words=(
                _word(phrase_function="Pred"),
                _word(phrase_function="Pred"),
                _word(phrase_function="Subj"),
            ),
        ),
    ]
    fm = build_phrase_function_feature_matrix(psalms)
    assert set(fm.terms) == {"Pred", "Subj"}
    assert fm.counts[0, fm.terms.index("Pred")] == 2
    assert fm.counts[0, fm.terms.index("Subj")] == 1


def test_term_info_has_human_readable_labels():
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(_word(phrase_function="Voct"),))]
    fm = build_phrase_function_feature_matrix(psalms)
    assert fm.term_info["Voct"].label == "Vocative"


def test_every_word_contributes_a_tag_no_filter_function():
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(_word(phrase_function="Time"),))]
    fm = build_phrase_function_feature_matrix(psalms)
    assert fm.terms == ("Time",)


def test_build_matrix_handles_empty_psalm_list():
    fm = build_phrase_function_feature_matrix([])
    assert fm.psalm_numbers == ()
    assert fm.counts.shape == (0, 0)


def test_build_matrix_preserves_psalm_order_and_numbers():
    psalms = [
        Psalm(number=5, verse_count=1, incipit="", words=(_word(phrase_function="Pred"),)),
        Psalm(number=2, verse_count=1, incipit="", words=(_word(phrase_function="Subj"),)),
    ]
    fm = build_phrase_function_feature_matrix(psalms)
    assert fm.psalm_numbers == (5, 2)
