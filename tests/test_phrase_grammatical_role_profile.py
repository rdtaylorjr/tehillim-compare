from __future__ import annotations

from tehillim_compare.corpus import Psalm, PsalmWord
from tehillim_compare.phrase_grammatical_role_profile import (
    build_phrase_grammatical_role_feature_matrix,
    grammatical_role_words,
)


def _word(
    pos: str = "subs", phrase_grammatical_role: str = "", lexeme: str = "X", node: int = 0
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
        phrase_type="",
        phrase_valence="",
        phrase_grammatical_role=phrase_grammatical_role,
        verb_sense="",
    )


def test_grammatical_role_words_excludes_words_with_no_role():
    words = (_word(phrase_grammatical_role="subject"), _word())
    assert len(grammatical_role_words(words)) == 1


def test_build_matrix_counts_grammatical_role_tags():
    psalms = [
        Psalm(
            number=1,
            verse_count=1,
            incipit="",
            words=(
                _word(phrase_grammatical_role="direct_object"),
                _word(phrase_grammatical_role="direct_object"),
                _word(phrase_grammatical_role="subject"),
            ),
        ),
    ]
    fm = build_phrase_grammatical_role_feature_matrix(psalms)
    assert set(fm.terms) == {"direct_object", "subject"}
    assert fm.counts[0, fm.terms.index("direct_object")] == 2
    assert fm.counts[0, fm.terms.index("subject")] == 1


def test_term_info_label_replaces_underscores_with_spaces():
    psalms = [
        Psalm(
            number=1,
            verse_count=1,
            incipit="",
            words=(_word(phrase_grammatical_role="principal_direct_object"),),
        )
    ]
    fm = build_phrase_grammatical_role_feature_matrix(psalms)
    assert fm.term_info["principal_direct_object"].label == "principal direct object"


def test_build_matrix_handles_psalm_with_no_grammatical_role_words():
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(_word(),))]
    fm = build_phrase_grammatical_role_feature_matrix(psalms)
    assert fm.terms == ()
    assert fm.counts.shape == (1, 0)


def test_build_matrix_handles_empty_psalm_list():
    fm = build_phrase_grammatical_role_feature_matrix([])
    assert fm.psalm_numbers == ()
    assert fm.counts.shape == (0, 0)


def test_build_matrix_preserves_psalm_order_and_numbers():
    psalms = [
        Psalm(
            number=5,
            verse_count=1,
            incipit="",
            words=(_word(phrase_grammatical_role="subject"),),
        ),
        Psalm(
            number=2,
            verse_count=1,
            incipit="",
            words=(_word(phrase_grammatical_role="direct_object"),),
        ),
    ]
    fm = build_phrase_grammatical_role_feature_matrix(psalms)
    assert fm.psalm_numbers == (5, 2)
