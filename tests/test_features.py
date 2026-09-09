from __future__ import annotations

from tehillim_compare.corpus import Psalm, PsalmWord
from tehillim_compare.features import (
    CONTENT_POS,
    FeatureInfo,
    build_lexical_feature_matrix,
    build_tag_count_feature_matrix,
    content_words,
)


def _word(
    lexeme: str,
    pos: str = "subs",
    lemma: str | None = None,
    gloss: str = "",
    node: int = 0,
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


def _psalm(number: int, words: list[PsalmWord]) -> Psalm:
    return Psalm(number=number, verse_count=1, incipit="", words=tuple(words))


def test_content_words_excludes_closed_class_pos():
    words = (_word("MLK/", pos="subs"), _word("W", pos="conj"), _word("B", pos="prep"))
    assert [w.lexeme for w in content_words(words)] == ["MLK/"]


def test_content_words_keeps_every_content_pos_category():
    words = tuple(_word(f"L{i}", pos=pos) for i, pos in enumerate(sorted(CONTENT_POS)))
    assert len(content_words(words)) == len(words)


def test_content_words_preserves_input_order():
    words = (_word("A"), _word("W", pos="conj"), _word("B"))
    assert [w.lexeme for w in content_words(words)] == ["A", "B"]


def test_build_lexical_feature_matrix_counts_lexeme_occurrences():
    psalms = [
        Psalm(
            number=1,
            verse_count=1,
            incipit="",
            words=(_word("MLK/"), _word("MLK/"), _word("YD/")),
        ),
        Psalm(number=2, verse_count=1, incipit="", words=(_word("MLK/"),)),
    ]
    fm = build_lexical_feature_matrix(psalms)

    assert fm.psalm_numbers == (1, 2)
    assert set(fm.terms) == {"MLK/", "YD/"}

    mlk_col = fm.terms.index("MLK/")
    yd_col = fm.terms.index("YD/")
    assert fm.counts[0, mlk_col] == 2
    assert fm.counts[0, yd_col] == 1
    assert fm.counts[1, mlk_col] == 1
    assert fm.counts[1, yd_col] == 0


def test_build_lexical_feature_matrix_excludes_function_words_from_vocabulary():
    psalms = [
        Psalm(number=1, verse_count=1, incipit="", words=(_word("MLK/"), _word("W", pos="conj")))
    ]
    fm = build_lexical_feature_matrix(psalms)
    assert "W" not in fm.terms


def test_build_lexical_feature_matrix_records_term_display_info():
    psalms = [
        Psalm(
            number=1,
            verse_count=1,
            incipit="",
            words=(_word("MLK/", lemma="מֶלֶךְ", gloss="king"),),
        )
    ]
    fm = build_lexical_feature_matrix(psalms)
    info = fm.term_info["MLK/"]
    assert info.label == "מֶלֶךְ"
    assert info.description == "king"
    assert info.category == "subs"


def test_build_lexical_feature_matrix_handles_psalm_with_no_content_words():
    psalms = [Psalm(number=1, verse_count=1, incipit="", words=(_word("W", pos="conj"),))]
    fm = build_lexical_feature_matrix(psalms)
    assert fm.terms == ()
    assert fm.counts.shape == (1, 0)


def test_build_lexical_feature_matrix_handles_empty_psalm_list():
    fm = build_lexical_feature_matrix([])
    assert fm.psalm_numbers == ()
    assert fm.terms == ()
    assert fm.counts.shape == (0, 0)


def test_build_lexical_feature_matrix_terms_are_sorted_and_deduplicated():
    psalms = [
        Psalm(number=1, verse_count=1, incipit="", words=(_word("Z/"), _word("A/"), _word("Z/")))
    ]
    fm = build_lexical_feature_matrix(psalms)
    assert fm.terms == ("A/", "Z/")


class TestBuildTagCountFeatureMatrix:
    """The one counting rule every profile builds its matrix with."""

    @staticmethod
    def _psalms() -> list[Psalm]:
        return [
            _psalm(1, [_word("a", pos="subs"), _word("b", pos="verb"), _word("a", pos="subs")]),
            _psalm(2, [_word("b", pos="verb")]),
        ]

    def test_counts_every_selected_words_tag_per_psalm(self) -> None:
        matrix = build_tag_count_feature_matrix(
            self._psalms(),
            lambda words: words,
            lambda word: [
                (word.lexeme, FeatureInfo(word.lexeme, word.gloss, "test")),
            ],
        )

        assert matrix.counts.tolist() == [[2, 1], [0, 1]]

    def test_columns_are_the_tags_in_sorted_order(self) -> None:
        matrix = build_tag_count_feature_matrix(
            self._psalms(),
            lambda words: words,
            lambda word: [
                (word.lexeme, FeatureInfo(word.lexeme, word.gloss, "test")),
            ],
        )

        assert matrix.terms == ("a", "b")

    def test_a_selector_that_drops_words_drops_their_tags_from_the_matrix(self) -> None:
        matrix = build_tag_count_feature_matrix(
            self._psalms(),
            lambda words: [w for w in words if w.part_of_speech == "verb"],
            lambda word: [(word.lexeme, FeatureInfo(word.lexeme, "", "test"))],
        )

        assert matrix.terms == ("b",)

    def test_a_tag_is_described_once_by_the_first_word_that_carries_it(self) -> None:
        """setdefault, so a later word with the same tag cannot overwrite its description."""
        psalms = [_psalm(1, [_word("a", gloss="first"), _word("a", gloss="second")])]

        matrix = build_tag_count_feature_matrix(
            psalms,
            lambda words: words,
            lambda word: [
                (word.lexeme, FeatureInfo(word.lexeme, word.gloss, "test")),
            ],
        )

        assert matrix.term_info["a"].description == "first"

    def test_a_psalm_with_no_selected_words_is_a_zero_row_rather_than_a_missing_one(self) -> None:
        """Every psalm keeps its row, so the matrix rows stay aligned with the psalm numbers."""
        psalms = [_psalm(1, [_word("a")]), _psalm(2, [])]

        matrix = build_tag_count_feature_matrix(
            psalms,
            lambda words: words,
            lambda word: [(word.lexeme, FeatureInfo(word.lexeme, "", "test"))],
        )

        assert matrix.psalm_numbers == (1, 2)
        assert matrix.counts[1].tolist() == [0]
