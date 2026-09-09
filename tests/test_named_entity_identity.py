from __future__ import annotations

from tehillim_compare.corpus import Psalm, PsalmWord
from tehillim_compare.named_entity_identity import (
    build_named_entity_identity_feature_matrix,
    proper_noun_words,
)


def _word(
    pos: str = "nmpr", lexeme: str = "X", lemma: str | None = None, gloss: str = "", node: int = 0
) -> PsalmWord:
    return PsalmWord(
        node=node,
        lexeme=lexeme,
        lemma=lemma or lexeme,
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
        verb_sense="",
    )


def test_proper_noun_words_excludes_non_proper_nouns():
    words = (_word(pos="nmpr"), _word(pos="subs"))
    assert len(proper_noun_words(words)) == 1


def test_build_matrix_counts_proper_noun_lexeme_tags():
    psalms = [
        Psalm(
            number=1,
            verse_count=1,
            incipit="",
            words=(
                _word(lexeme="JHWH/"),
                _word(lexeme="JHWH/"),
                _word(lexeme="YJWN==/"),
                _word(lexeme="MLK/", pos="subs"),
            ),
        ),
    ]
    fm = build_named_entity_identity_feature_matrix(psalms)
    assert set(fm.terms) == {"JHWH/", "YJWN==/"}
    assert fm.counts[0, fm.terms.index("JHWH/")] == 2
    assert fm.counts[0, fm.terms.index("YJWN==/")] == 1


def test_term_info_uses_lemma_as_the_label_and_gloss_as_the_description():
    word = _word(lexeme="DWD==/", lemma="דָּוִד", gloss="David")
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(word,))]
    fm = build_named_entity_identity_feature_matrix(psalms)
    assert fm.term_info["DWD==/"].label == "דָּוִד"
    assert fm.term_info["DWD==/"].description == "David"


def test_build_matrix_handles_psalm_with_no_proper_nouns():
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(_word(pos="subs"),))]
    fm = build_named_entity_identity_feature_matrix(psalms)
    assert fm.terms == ()
    assert fm.counts.shape == (1, 0)


def test_build_matrix_handles_empty_psalm_list():
    fm = build_named_entity_identity_feature_matrix([])
    assert fm.psalm_numbers == ()
    assert fm.counts.shape == (0, 0)


def test_build_matrix_preserves_psalm_order_and_numbers():
    psalms = [
        Psalm(number=5, verse_count=1, incipit="", words=(_word(lexeme="JHWH/"),)),
        Psalm(number=2, verse_count=1, incipit="", words=(_word(lexeme="DWD==/"),)),
    ]
    fm = build_named_entity_identity_feature_matrix(psalms)
    assert fm.psalm_numbers == (5, 2)
