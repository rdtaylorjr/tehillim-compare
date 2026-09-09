from __future__ import annotations

from tehillim_compare.corpus import Psalm, PsalmWord
from tehillim_compare.root_similarity import build_root_feature_matrix, root_words


def _word(
    pos: str = "subs", root: str = "", lexeme: str = "X", gloss: str = "", node: int = 0
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
        root=root,
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


def test_root_words_excludes_words_with_no_known_root():
    words = (_word(root="HGH"), _word())
    assert len(root_words(words)) == 1


def test_build_matrix_counts_root_tags():
    psalms = [
        Psalm(
            number=1,
            verse_count=1,
            incipit="",
            words=(_word(root="HGH"), _word(root="HGH"), _word(root="DRK")),
        ),
    ]
    fm = build_root_feature_matrix(psalms)
    assert set(fm.terms) == {"HGH", "DRK"}
    assert fm.counts[0, fm.terms.index("HGH")] == 2
    assert fm.counts[0, fm.terms.index("DRK")] == 1


def test_root_collapses_distinct_lexemes_into_one_term():
    # A verb and its cognate noun have different lexemes but the same root.
    psalms = [
        Psalm(
            number=1,
            verse_count=1,
            incipit="",
            words=(
                _word(lexeme="HGH[", pos="verb", root="HGH"),
                _word(lexeme="HGH/", pos="subs", root="HGH"),
            ),
        ),
    ]
    fm = build_root_feature_matrix(psalms)
    assert fm.terms == ("HGH",)
    assert fm.counts[0, 0] == 2


def test_term_info_uses_the_root_itself_as_the_label():
    word = _word(root="HGH", gloss="meditate")
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(word,))]
    fm = build_root_feature_matrix(psalms)
    assert fm.term_info["HGH"].label == "HGH"
    assert fm.term_info["HGH"].description == "meditate"


def test_build_matrix_handles_psalm_with_no_root_words():
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(_word(),))]
    fm = build_root_feature_matrix(psalms)
    assert fm.terms == ()
    assert fm.counts.shape == (1, 0)


def test_build_matrix_handles_empty_psalm_list():
    fm = build_root_feature_matrix([])
    assert fm.psalm_numbers == ()
    assert fm.counts.shape == (0, 0)
