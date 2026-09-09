from __future__ import annotations

from tehillim_compare.corpus import Psalm, PsalmWord
from tehillim_compare.phrase_type_profile import build_phrase_type_feature_matrix


def _word(
    pos: str = "subs", phrase_type: str | None = None, lexeme: str = "X", node: int = 0
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
        phrase_type=phrase_type if phrase_type is not None else "NP",
        phrase_valence="",
        phrase_grammatical_role="",
        verb_sense="",
    )


def test_build_matrix_counts_phrase_type_tags():
    psalms = [
        Psalm(
            number=1,
            verse_count=1,
            incipit="",
            words=(_word(phrase_type="VP"), _word(phrase_type="VP"), _word(phrase_type="PP")),
        ),
    ]
    fm = build_phrase_type_feature_matrix(psalms)
    assert set(fm.terms) == {"VP", "PP"}
    assert fm.counts[0, fm.terms.index("VP")] == 2
    assert fm.counts[0, fm.terms.index("PP")] == 1


def test_term_info_has_human_readable_labels():
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(_word(phrase_type="PrNP"),))]
    fm = build_phrase_type_feature_matrix(psalms)
    assert fm.term_info["PrNP"].label == "Proper-Noun Phrase"


def test_every_word_contributes_a_tag_no_filter_function():
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(_word(phrase_type="AdjP"),))]
    fm = build_phrase_type_feature_matrix(psalms)
    assert fm.terms == ("AdjP",)


def test_build_matrix_handles_empty_psalm_list():
    fm = build_phrase_type_feature_matrix([])
    assert fm.psalm_numbers == ()
    assert fm.counts.shape == (0, 0)


def test_build_matrix_preserves_psalm_order_and_numbers():
    psalms = [
        Psalm(number=5, verse_count=1, incipit="", words=(_word(phrase_type="VP"),)),
        Psalm(number=2, verse_count=1, incipit="", words=(_word(phrase_type="NP"),)),
    ]
    fm = build_phrase_type_feature_matrix(psalms)
    assert fm.psalm_numbers == (5, 2)
