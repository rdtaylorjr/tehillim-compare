from __future__ import annotations

from tehillim_compare.corpus import Psalm, PsalmWord
from tehillim_compare.verb_morphology import build_verb_morphology_feature_matrix, verb_words


def _word(
    pos: str = "verb",
    verb_stem: str = "qal",
    verb_mood: str = "perf",
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
        verb_stem=verb_stem,
        verb_mood=verb_mood,
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


def test_verb_words_excludes_non_verbs():
    words = (_word(pos="verb"), _word(pos="subs", verb_stem="", verb_mood=""))
    assert len(verb_words(words)) == 1


def test_verb_words_excludes_verbs_missing_stem_or_mood():
    # Defensive: corpus.py guarantees verbs always carry both.
    words = (
        _word(pos="verb", verb_stem="", verb_mood="impv"),
        _word(pos="verb", verb_stem="piel", verb_mood=""),
        _word(pos="verb", verb_stem="piel", verb_mood="impv"),
    )
    assert len(verb_words(words)) == 1


def test_build_matrix_counts_stem_mood_tag_occurrences():
    psalms = [
        Psalm(
            number=150,
            verse_count=1,
            incipit="",
            words=(
                _word(verb_stem="piel", verb_mood="impv"),
                _word(verb_stem="piel", verb_mood="impv"),
                _word(verb_stem="qal", verb_mood="perf"),
            ),
        ),
    ]
    fm = build_verb_morphology_feature_matrix(psalms)

    assert set(fm.terms) == {"piel.impv", "qal.perf"}
    impv_col = fm.terms.index("piel.impv")
    perf_col = fm.terms.index("qal.perf")
    assert fm.counts[0, impv_col] == 2
    assert fm.counts[0, perf_col] == 1


def test_build_matrix_excludes_non_verb_words_from_vocabulary():
    psalms = [
        Psalm(
            number=1,
            verse_count=1,
            incipit="",
            words=(_word(pos="verb"), _word(pos="subs", verb_stem="", verb_mood="")),
        )
    ]
    fm = build_verb_morphology_feature_matrix(psalms)
    assert len(fm.terms) == 1


def test_term_info_gives_human_readable_label_for_known_tag():
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(_word("verb", "piel", "impv"),))]
    fm = build_verb_morphology_feature_matrix(psalms)
    info = fm.term_info["piel.impv"]
    assert info.label == "Piel Imperative"
    assert info.category == "verb-morphology"


def test_term_info_falls_back_to_raw_code_for_unrecognized_tag():
    # Forward-compatible: an unlisted BHSA stem/mood code shouldn't crash extraction.
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(_word("verb", "zzz", "yyy"),))]
    fm = build_verb_morphology_feature_matrix(psalms)
    info = fm.term_info["zzz.yyy"]
    assert "zzz" in info.label
    assert "yyy" in info.label


def test_build_matrix_handles_psalm_with_no_verbs():
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(_word(pos="subs"),))]
    fm = build_verb_morphology_feature_matrix(psalms)
    assert fm.terms == ()
    assert fm.counts.shape == (1, 0)


def test_build_matrix_handles_empty_psalm_list():
    fm = build_verb_morphology_feature_matrix([])
    assert fm.psalm_numbers == ()
    assert fm.counts.shape == (0, 0)


def test_build_matrix_preserves_psalm_order_and_numbers():
    psalms = [
        Psalm(number=5, verse_count=1, incipit="", words=(_word(),)),
        Psalm(number=2, verse_count=1, incipit="", words=(_word(),)),
    ]
    fm = build_verb_morphology_feature_matrix(psalms)
    assert fm.psalm_numbers == (5, 2)
