from __future__ import annotations

import numpy as np

from tehillim_compare.corpus import Psalm
from tehillim_compare.export import MethodComputation, build_similarity_payload
from tehillim_compare.features import FeatureInfo, FeatureMatrix
from tehillim_compare.similarity import SimilarityResult


def _psalms() -> list[Psalm]:
    return [
        Psalm(number=1, verse_count=2, incipit="אשרי", words=()),
        Psalm(number=2, verse_count=3, incipit="למה", words=()),
        Psalm(number=3, verse_count=1, incipit="יהוה", words=()),
    ]


def _lexical_computation() -> MethodComputation:
    terms = ("MLK/", "YD/")
    term_info = {
        "MLK/": FeatureInfo(label="מֶלֶךְ", description="king", category="subs"),
        "YD/": FeatureInfo(label="יָד", description="hand", category="subs"),
    }
    counts = np.array([[2, 1], [1, 0], [0, 3]], dtype=np.int32)
    features = FeatureMatrix(
        psalm_numbers=(1, 2, 3), terms=terms, counts=counts, term_info=term_info
    )
    weights = counts.astype(float)
    matrix = np.array(
        [
            [1.0, 0.6, 0.1],
            [0.6, 1.0, 0.05],
            [0.1, 0.05, 1.0],
        ]
    )
    result = SimilarityResult(
        method="lexical-tfidf-cosine",
        description="lexical test method",
        psalm_numbers=(1, 2, 3),
        matrix=matrix,
    )
    return MethodComputation(features=features, weights=weights, result=result)


def _verb_morphology_computation() -> MethodComputation:
    terms = ("piel.impv",)
    term_info = {
        "piel.impv": FeatureInfo(
            label="Piel Imperative",
            description="Piel Imperative verb form",
            category="verb-morphology",
        ),
    }
    counts = np.array([[0], [1], [1]], dtype=np.int32)
    features = FeatureMatrix(
        psalm_numbers=(1, 2, 3), terms=terms, counts=counts, term_info=term_info
    )
    weights = counts.astype(float)
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    result = SimilarityResult(
        method="verb-morphology-tfidf-cosine",
        description="verb morphology test method",
        psalm_numbers=(1, 2, 3),
        matrix=matrix,
    )
    return MethodComputation(features=features, weights=weights, result=result)


def _payload():
    return build_similarity_payload(
        psalms=_psalms(),
        computations=[_lexical_computation(), _verb_morphology_computation()],
        default_method="lexical-tfidf-cosine",
    )


def test_payload_has_expected_top_level_keys():
    payload = _payload()
    assert set(payload) == {"generatedAt", "corpus", "psalms", "methods", "defaultMethod"}


def test_corpus_and_default_method_are_recorded():
    payload = _payload()
    assert payload["corpus"] == {"name": "ETCBC/BHSA", "version": "2021"}
    assert payload["defaultMethod"] == "lexical-tfidf-cosine"


def test_psalms_are_method_independent_and_shared_across_methods():
    payload = _payload()
    assert len(payload["psalms"]) == 3
    p1 = payload["psalms"][0]
    assert p1 == {"number": 1, "verseCount": 2, "wordCount": 0, "incipit": "אשרי"}


def test_both_methods_are_present_with_correct_ids():
    payload = _payload()
    ids = [m["id"] for m in payload["methods"]]
    assert ids == ["lexical-tfidf-cosine", "verb-morphology-tfidf-cosine"]


def test_each_method_carries_its_own_description():
    payload = _payload()
    lexical = next(m for m in payload["methods"] if m["id"] == "lexical-tfidf-cosine")
    verb = next(m for m in payload["methods"] if m["id"] == "verb-morphology-tfidf-cosine")
    assert lexical["description"] == "lexical test method"
    assert verb["description"] == "verb morphology test method"


def test_each_method_has_its_own_square_matrix():
    payload = _payload()
    for method in payload["methods"]:
        assert len(method["matrix"]) == 3
        assert all(len(row) == 3 for row in method["matrix"])


def test_similar_psalms_exclude_self_within_each_method():
    payload = _payload()
    for method in payload["methods"]:
        entries = method["similar"]["1"]
        assert all(entry["psalm"] != 1 for entry in entries)


def test_similar_psalms_are_ranked_descending_by_score():
    payload = _payload()
    lexical = next(m for m in payload["methods"] if m["id"] == "lexical-tfidf-cosine")
    entries = lexical["similar"]["1"]
    scores = [entry["score"] for entry in entries]
    assert scores == sorted(scores, reverse=True)
    assert entries[0]["psalm"] == 2


def test_shared_terms_only_include_terms_present_in_both_psalms():
    payload = _payload()
    lexical = next(m for m in payload["methods"] if m["id"] == "lexical-tfidf-cosine")
    # Psalm 1 has MLK/+YD/; Psalm 3 (weakest match) has only YD/.
    weakest_match = lexical["similar"]["1"][-1]
    assert weakest_match["psalm"] == 3
    assert all(term["label"] == "יָד" for term in weakest_match["sharedTerms"])


def test_psalm_stats_include_expected_fields_per_method():
    payload = _payload()
    lexical = next(m for m in payload["methods"] if m["id"] == "lexical-tfidf-cosine")
    stats = lexical["psalmStats"][0]
    assert stats["number"] == 1
    assert stats["uniqueTermCount"] == 2
    assert stats["termCount"] == 3


def test_top_terms_exclude_zero_score_terms():
    payload = _payload()
    lexical = next(m for m in payload["methods"] if m["id"] == "lexical-tfidf-cosine")
    p2_stats = lexical["psalmStats"][1]  # counts [1, 0] -> only MLK/ has weight
    assert len(p2_stats["topTerms"]) == 1
    assert p2_stats["topTerms"][0]["label"] == "מֶלֶךְ"


def test_scores_are_rounded_floats():
    payload = _payload()
    for method in payload["methods"]:
        for entry in method["similar"]["1"]:
            assert isinstance(entry["score"], float)


def test_different_methods_can_disagree_on_ranking():
    # Sanity: the two methods in this fixture are deliberately built to disagree (lexical favors.
    payload = _payload()
    verb = next(m for m in payload["methods"] if m["id"] == "verb-morphology-tfidf-cosine")
    assert verb["similar"]["2"][0]["psalm"] == 3
