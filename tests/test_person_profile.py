from __future__ import annotations

from tehillim_compare.corpus import Psalm, PsalmWord
from tehillim_compare.person_profile import build_person_feature_matrix, person_words


def _word(
    pos: str = "verb",
    person: str = "",
    number: str = "",
    suffix_person: str = "",
    suffix_number: str = "",
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
        person=person,
        number=number,
        suffix_person=suffix_person,
        suffix_number=suffix_number,
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


def test_person_words_excludes_words_with_no_person_marking_at_all():
    words = (_word(person="p1", number="sg"), _word())
    assert len(person_words(words)) == 1


def test_person_words_includes_words_with_only_a_suffix_person():
    # e.g.
    words = (_word(suffix_person="p1", suffix_number="sg"),)
    assert len(person_words(words)) == 1


def test_person_words_includes_words_with_both_word_and_suffix_person():
    words = (_word(person="p3", number="sg", suffix_person="p1", suffix_number="sg"),)
    assert len(person_words(words)) == 1


def test_build_matrix_counts_word_person_tags():
    psalms = [
        Psalm(
            number=1,
            verse_count=1,
            incipit="",
            words=(
                _word(person="p1", number="sg"),
                _word(person="p1", number="sg"),
                _word(person="p3", number="pl"),
            ),
        ),
    ]
    fm = build_person_feature_matrix(psalms)
    assert set(fm.terms) == {"word.p1.sg", "word.p3.pl"}
    assert fm.counts[0, fm.terms.index("word.p1.sg")] == 2
    assert fm.counts[0, fm.terms.index("word.p3.pl")] == 1


def test_build_matrix_counts_suffix_person_tags_separately_from_word_person():
    psalms = [
        Psalm(
            number=1,
            verse_count=1,
            incipit="",
            words=(_word(suffix_person="p1", suffix_number="sg"),),
        ),
    ]
    fm = build_person_feature_matrix(psalms)
    assert fm.terms == ("suffix.p1.sg",)


def test_build_matrix_gives_a_word_and_a_suffix_tag_for_a_doubly_marked_word():
    psalms = [
        Psalm(
            number=1,
            verse_count=1,
            incipit="",
            words=(_word(person="p3", number="sg", suffix_person="p1", suffix_number="sg"),),
        ),
    ]
    fm = build_person_feature_matrix(psalms)
    assert set(fm.terms) == {"word.p3.sg", "suffix.p1.sg"}
    assert fm.counts[0, fm.terms.index("word.p3.sg")] == 1
    assert fm.counts[0, fm.terms.index("suffix.p1.sg")] == 1


def test_term_info_gives_distinct_human_readable_labels_for_word_vs_suffix():
    psalms = [
        Psalm(
            number=1,
            verse_count=1,
            incipit="",
            words=(_word(person="p1", number="sg"), _word(suffix_person="p1", suffix_number="sg")),
        ),
    ]
    fm = build_person_feature_matrix(psalms)
    word_label = fm.term_info["word.p1.sg"].label
    suffix_label = fm.term_info["suffix.p1.sg"].label
    assert word_label != suffix_label
    assert "1st" in word_label
    assert "Singular" in word_label
    assert "Suffix" in suffix_label


def test_term_info_falls_back_gracefully_for_missing_number():
    # Defensive: a word could in principle have person set without number.
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(_word(person="p2", number=""),))]
    fm = build_person_feature_matrix(psalms)
    assert len(fm.terms) == 1
    assert fm.term_info[fm.terms[0]].label


def test_build_matrix_handles_psalm_with_no_person_marked_words():
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(_word(),))]
    fm = build_person_feature_matrix(psalms)
    assert fm.terms == ()
    assert fm.counts.shape == (1, 0)


def test_build_matrix_handles_empty_psalm_list():
    fm = build_person_feature_matrix([])
    assert fm.psalm_numbers == ()
    assert fm.counts.shape == (0, 0)


def test_build_matrix_preserves_psalm_order_and_numbers():
    psalms = [
        Psalm(number=5, verse_count=1, incipit="", words=(_word(person="p1", number="sg"),)),
        Psalm(number=2, verse_count=1, incipit="", words=(_word(person="p1", number="sg"),)),
    ]
    fm = build_person_feature_matrix(psalms)
    assert fm.psalm_numbers == (5, 2)
