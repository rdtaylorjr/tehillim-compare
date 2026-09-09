from __future__ import annotations

from tehillim_compare.corpus import Psalm, PsalmWord
from tehillim_compare.gender_profile import build_gender_feature_matrix, gender_words


def _word(
    pos: str = "subs",
    gender: str = "",
    suffix_gender: str = "",
    lexeme: str = "X",
    node: int = 0,
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
        gender=gender,
        suffix_gender=suffix_gender,
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


def test_gender_words_excludes_words_with_no_gender_marking_at_all():
    words = (_word(gender="m"), _word())
    assert len(gender_words(words)) == 1


def test_gender_words_includes_words_with_only_a_suffix_gender():
    words = (_word(suffix_gender="f"),)
    assert len(gender_words(words)) == 1


def test_gender_words_includes_words_with_both_word_and_suffix_gender():
    words = (_word(gender="m", suffix_gender="f"),)
    assert len(gender_words(words)) == 1


def test_build_matrix_counts_word_gender_tags():
    psalms = [
        Psalm(
            number=1,
            verse_count=1,
            incipit="",
            words=(_word(gender="m"), _word(gender="m"), _word(gender="f")),
        ),
    ]
    fm = build_gender_feature_matrix(psalms)
    assert set(fm.terms) == {"word.m", "word.f"}
    assert fm.counts[0, fm.terms.index("word.m")] == 2
    assert fm.counts[0, fm.terms.index("word.f")] == 1


def test_build_matrix_counts_suffix_gender_tags_separately_from_word_gender():
    psalms = [
        Psalm(number=1, verse_count=1, incipit="", words=(_word(suffix_gender="f"),)),
    ]
    fm = build_gender_feature_matrix(psalms)
    assert fm.terms == ("suffix.f",)


def test_build_matrix_gives_a_word_and_a_suffix_tag_for_a_doubly_marked_word():
    psalms = [
        Psalm(number=1, verse_count=1, incipit="", words=(_word(gender="m", suffix_gender="f"),)),
    ]
    fm = build_gender_feature_matrix(psalms)
    assert set(fm.terms) == {"word.m", "suffix.f"}
    assert fm.counts[0, fm.terms.index("word.m")] == 1
    assert fm.counts[0, fm.terms.index("suffix.f")] == 1


def test_term_info_gives_distinct_human_readable_labels_for_word_vs_suffix():
    psalms = [
        Psalm(
            number=1,
            verse_count=1,
            incipit="",
            words=(_word(gender="f"), _word(suffix_gender="f")),
        ),
    ]
    fm = build_gender_feature_matrix(psalms)
    word_label = fm.term_info["word.f"].label
    suffix_label = fm.term_info["suffix.f"].label
    assert word_label != suffix_label
    assert "Feminine" in word_label
    assert "Suffix" in suffix_label


def test_unknown_gender_code_falls_back_to_the_raw_code():
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(_word(gender="zzzz"),))]
    fm = build_gender_feature_matrix(psalms)
    assert fm.term_info["word.zzzz"].label == "zzzz"


def test_build_matrix_handles_psalm_with_no_gender_marked_words():
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(_word(),))]
    fm = build_gender_feature_matrix(psalms)
    assert fm.terms == ()
    assert fm.counts.shape == (1, 0)


def test_build_matrix_handles_empty_psalm_list():
    fm = build_gender_feature_matrix([])
    assert fm.psalm_numbers == ()
    assert fm.counts.shape == (0, 0)


def test_build_matrix_preserves_psalm_order_and_numbers():
    psalms = [
        Psalm(number=5, verse_count=1, incipit="", words=(_word(gender="m"),)),
        Psalm(number=2, verse_count=1, incipit="", words=(_word(gender="m"),)),
    ]
    fm = build_gender_feature_matrix(psalms)
    assert fm.psalm_numbers == (5, 2)
