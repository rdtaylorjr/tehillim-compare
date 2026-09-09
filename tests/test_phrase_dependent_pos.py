from __future__ import annotations

from tehillim_compare.corpus import Psalm, PsalmWord
from tehillim_compare.phrase_dependent_pos import build_phrase_dependent_pos_feature_matrix


def _word(pos: str = "subs", pdp: str | None = None, lexeme: str = "X", node: int = 0) -> PsalmWord:
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
        phrase_dependent_pos=pdp if pdp is not None else pos,
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


def test_build_matrix_counts_pdp_tags():
    psalms = [
        Psalm(
            number=1,
            verse_count=1,
            incipit="",
            words=(_word(pdp="subs"), _word(pdp="subs"), _word(pdp="verb")),
        ),
    ]
    fm = build_phrase_dependent_pos_feature_matrix(psalms)
    assert set(fm.terms) == {"subs", "verb"}
    assert fm.counts[0, fm.terms.index("subs")] == 2
    assert fm.counts[0, fm.terms.index("verb")] == 1


def test_pdp_can_differ_from_part_of_speech_for_a_substantivized_adjective():
    psalms = [
        Psalm(
            number=1,
            verse_count=1,
            incipit="",
            words=(_word(pos="adjv", pdp="subs"),),
        ),
    ]
    fm = build_phrase_dependent_pos_feature_matrix(psalms)
    assert fm.terms == ("subs",)


def test_term_info_has_human_readable_labels():
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(_word(pdp="nmpr"),))]
    fm = build_phrase_dependent_pos_feature_matrix(psalms)
    assert fm.term_info["nmpr"].label == "Proper Noun"


def test_unknown_pdp_code_falls_back_to_the_raw_code():
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(_word(pdp="zzzz"),))]
    fm = build_phrase_dependent_pos_feature_matrix(psalms)
    assert fm.term_info["zzzz"].label == "zzzz"


def test_every_word_contributes_a_tag_no_filter_function():
    # Unlike other extractors, pdp is always populated - no words are excluded.
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(_word(pdp="conj"),))]
    fm = build_phrase_dependent_pos_feature_matrix(psalms)
    assert fm.terms == ("conj",)


def test_build_matrix_handles_empty_psalm_list():
    fm = build_phrase_dependent_pos_feature_matrix([])
    assert fm.psalm_numbers == ()
    assert fm.counts.shape == (0, 0)


def test_build_matrix_preserves_psalm_order_and_numbers():
    psalms = [
        Psalm(number=5, verse_count=1, incipit="", words=(_word(pdp="subs"),)),
        Psalm(number=2, verse_count=1, incipit="", words=(_word(pdp="verb"),)),
    ]
    fm = build_phrase_dependent_pos_feature_matrix(psalms)
    assert fm.psalm_numbers == (5, 2)
