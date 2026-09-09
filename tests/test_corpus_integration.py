"""Integration tests against the real BHSA Text-Fabric dataset."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def test_extracts_all_150_psalms_in_order(psalms):
    assert len(psalms) == 150
    assert [p.number for p in psalms] == list(range(1, 151))


def test_psalm_1_has_six_verses(psalms):
    psalm_1 = next(p for p in psalms if p.number == 1)
    assert psalm_1.verse_count == 6


def test_psalm_119_is_the_longest_by_word_count(psalms):
    longest = max(psalms, key=lambda p: len(p.words))
    assert longest.number == 119


def test_every_psalm_has_words_and_an_incipit(psalms):
    assert all(p.words for p in psalms)
    assert all(p.incipit for p in psalms)


def test_divine_name_yhwh_is_tagged_as_proper_noun(psalms):
    all_words = [w for p in psalms for w in p.words]
    yhwh_words = [w for w in all_words if w.lexeme == "JHWH/"]
    assert yhwh_words
    assert all(w.part_of_speech == "nmpr" for w in yhwh_words)


def test_words_carry_nonempty_display_fields(psalms):
    sample = psalms[0].words[0]
    assert sample.lemma
    assert sample.surface
    assert sample.part_of_speech


def test_words_carry_a_unique_text_fabric_node_id(psalms):
    all_words = [w for p in psalms for w in p.words]
    node_ids = [w.node for w in all_words]
    assert all(isinstance(n, int) and n > 0 for n in node_ids)
    assert len(set(node_ids)) == len(node_ids)


def test_psalm_150_is_dominated_by_piel_imperative_praise_verbs(psalms):
    # "הַלְלוּ" (praise!) repeated throughout - the textbook example of the imperative-heavy hymnic.
    psalm_150 = next(p for p in psalms if p.number == 150)
    verbs = [w for w in psalm_150.words if w.part_of_speech == "verb"]
    assert verbs
    piel_imperatives = [w for w in verbs if w.verb_stem == "piel" and w.verb_mood == "impv"]
    assert len(piel_imperatives) / len(verbs) > 0.5


def test_non_verb_words_have_empty_verb_stem_and_mood(psalms):
    non_verbs = [w for p in psalms for w in p.words if w.part_of_speech != "verb"]
    assert non_verbs
    assert all(w.verb_stem == "" and w.verb_mood == "" for w in non_verbs)


def test_verb_words_have_nonempty_verb_stem_and_mood(psalms):
    verbs = [w for p in psalms for w in p.words if w.part_of_speech == "verb"]
    assert verbs
    assert all(w.verb_stem != "" and w.verb_mood != "" for w in verbs)


def test_person_field_covers_all_three_grammatical_persons(psalms):
    all_words = [w for p in psalms for w in p.words]
    persons = {w.person for w in all_words if w.person}
    assert {"p1", "p2", "p3"} <= persons


def test_suffix_person_field_covers_all_three_grammatical_persons(psalms):
    all_words = [w for p in psalms for w in p.words]
    suffix_persons = {w.suffix_person for w in all_words if w.suffix_person}
    assert {"p1", "p2", "p3"} <= suffix_persons


def test_number_field_distinguishes_singular_and_plural(psalms):
    all_words = [w for p in psalms for w in p.words]
    numbers = {w.number for w in all_words if w.number}
    assert {"sg", "pl"} <= numbers


def test_psalm_22_has_first_person_singular_words(psalms):
    # The classic "My God, my God.
    psalm_22 = next(p for p in psalms if p.number == 22)
    first_person_singular = [
        w for w in psalm_22.words if w.person == "p1" or w.suffix_person == "p1"
    ]
    assert len(first_person_singular) > 5


def test_words_without_a_pronominal_suffix_have_empty_suffix_fields(psalms):
    # suffix_person/suffix_number are only meaningful on words that actually carry a pronominal.
    all_words = [w for p in psalms for w in p.words]
    unsuffixed = [w for w in all_words if w.suffix_person == ""]
    assert unsuffixed
    assert all(w.suffix_number == "" for w in unsuffixed)


def test_number_is_populated_independently_of_person(psalms):
    # `number` is a general word-level feature (nouns/adjectives have it too).
    all_words = [w for p in psalms for w in p.words]
    plain_plural_nouns = [
        w for w in all_words if w.part_of_speech == "subs" and w.number == "pl" and w.person == ""
    ]
    assert plain_plural_nouns


def test_gender_field_distinguishes_masculine_and_feminine(psalms):
    all_words = [w for p in psalms for w in p.words]
    genders = {w.gender for w in all_words if w.gender}
    assert {"m", "f"} <= genders


def test_words_without_gender_marking_have_empty_gender_field(psalms):
    all_words = [w for p in psalms for w in p.words]
    ungendered = [w for w in all_words if w.gender == ""]
    assert ungendered


def test_suffix_gender_field_is_populated_for_some_suffixed_words(psalms):
    all_words = [w for p in psalms for w in p.words]
    suffix_genders = {w.suffix_gender for w in all_words if w.suffix_gender}
    assert suffix_genders
    assert suffix_genders <= {"m", "f", "unknown"}


def test_state_field_distinguishes_construct_and_absolute(psalms):
    # Construct-chain density (e.g.
    all_words = [w for p in psalms for w in p.words]
    states = {w.state for w in all_words if w.state}
    assert states == {"c", "a"}


def test_finite_verbs_have_empty_state(psalms):
    # State (construct/absolute) is a nominal category.
    finite_moods = {"perf", "impf", "wayq", "impv", "juss", "coho"}
    all_words = [w for p in psalms for w in p.words]
    finite_verbs = [
        w for w in all_words if w.part_of_speech == "verb" and w.verb_mood in finite_moods
    ]
    assert finite_verbs
    assert all(w.state == "" for w in finite_verbs)


def test_verbal_participles_and_infinitives_carry_state(psalms):
    # Unlike finite verb forms.
    nominal_moods = {"ptca", "ptcp", "infc", "infa"}
    all_words = [w for p in psalms for w in p.words]
    nominal_verbs = [
        w for w in all_words if w.part_of_speech == "verb" and w.verb_mood in nominal_moods
    ]
    assert nominal_verbs
    assert any(w.state != "" for w in nominal_verbs)


def test_lexical_set_field_is_populated_for_a_minority_of_words(psalms):
    # Most words have no lexical-set subcategory (BHSA's "none" sentinel.
    all_words = [w for p in psalms for w in p.words]
    with_lexical_set = [w for w in all_words if w.lexical_set]
    without = [w for w in all_words if w.lexical_set == ""]
    assert with_lexical_set
    assert without
    assert len(with_lexical_set) < len(without)
    assert "nmdi" in {w.lexical_set for w in with_lexical_set}


def test_phrase_dependent_pos_differs_from_part_of_speech_for_some_words(psalms):
    # pdp reclassifies a word's function within its specific phrase (e.g.
    all_words = [w for p in psalms for w in p.words]
    mismatches = [w for w in all_words if w.phrase_dependent_pos != w.part_of_speech]
    assert mismatches
    assert any(w.part_of_speech == "adjv" and w.phrase_dependent_pos == "subs" for w in mismatches)


def test_phrase_dependent_pos_is_always_populated(psalms):
    all_words = [w for p in psalms for w in p.words]
    assert all(w.phrase_dependent_pos != "" for w in all_words)


def test_name_type_marks_the_divine_name_as_a_personal_name(psalms):
    all_words = [w for p in psalms for w in p.words]
    yhwh_words = [w for w in all_words if w.lexeme == "JHWH/"]
    assert yhwh_words
    assert all(w.name_type == "pers" for w in yhwh_words)


def test_name_type_marks_zion_as_a_place_name(psalms):
    all_words = [w for p in psalms for w in p.words]
    zion_words = [w for w in all_words if w.lexeme == "YJWN==/"]
    assert zion_words
    assert all(w.name_type == "topo" for w in zion_words)


def test_most_words_have_no_name_type(psalms):
    all_words = [w for p in psalms for w in p.words]
    assert sum(1 for w in all_words if w.name_type == "") > sum(
        1 for w in all_words if w.name_type != ""
    )


def test_root_is_populated_for_a_subset_of_content_words(psalms):
    all_words = [w for p in psalms for w in p.words]
    with_root = [w for w in all_words if w.root]
    without_root = [w for w in all_words if w.root == ""]
    assert with_root
    assert without_root


def test_root_collapses_a_verb_and_its_cognate_noun(psalms):
    # HGH/ ("meditation", noun) and HGH[ ("to meditate".
    all_words = [w for p in psalms for w in p.words]
    roots = {w.root for w in all_words if w.lexeme in ("HGH/", "HGH[")}
    assert roots == {"HGH"}


def test_clause_type_is_always_populated_with_a_rich_vocabulary(psalms):
    all_words = [w for p in psalms for w in p.words]
    assert all(w.clause_type != "" for w in all_words)
    assert len({w.clause_type for w in all_words}) >= 30


def test_text_type_marks_quotation_narrative_and_discursive(psalms):
    all_words = [w for p in psalms for w in p.words]
    text_types = {w.text_type for w in all_words}
    assert any(t.startswith("Q") for t in text_types)
    assert any(t.startswith("N") for t in text_types)
    assert any(t.startswith("D") for t in text_types)


def test_clause_relation_is_populated_for_a_minority_of_words(psalms):
    all_words = [w for p in psalms for w in p.words]
    with_relation = [w for w in all_words if w.clause_relation]
    without = [w for w in all_words if w.clause_relation == ""]
    assert with_relation
    assert without
    assert "Coor" in {w.clause_relation for w in with_relation}


def test_clause_kind_has_exactly_the_three_documented_values(psalms):
    all_words = [w for p in psalms for w in p.words]
    assert {w.clause_kind for w in all_words} == {"VC", "NC", "WP"}


def test_psalm_150_clauses_are_entirely_verbal(psalms):
    # A pure imperative "praise the LORD" hymn - every clause should be kind VC (verbal).
    psalm_150 = next(p for p in psalms if p.number == 150)
    kinds = {w.clause_kind for w in psalm_150.words if w.clause_kind}
    assert kinds == {"VC"}


def test_phrase_function_is_always_populated_with_a_rich_vocabulary(psalms):
    all_words = [w for p in psalms for w in p.words]
    assert all(w.phrase_function != "" for w in all_words)
    assert len({w.phrase_function for w in all_words}) >= 20


def test_phrase_determination_has_only_det_and_und_values(psalms):
    # "NA" (not applicable) normalizes to "" like every other field; the remaining real values are.
    all_words = [w for p in psalms for w in p.words]
    non_empty = {w.phrase_determination for w in all_words if w.phrase_determination}
    assert non_empty == {"det", "und"}


def test_phrase_type_covers_verbal_nominal_and_prepositional_phrases(psalms):
    all_words = [w for p in psalms for w in p.words]
    phrase_types = {w.phrase_type for w in all_words}
    assert {"VP", "NP", "PP"} <= phrase_types


def test_phrase_valence_distinguishes_core_complement_and_adjunct(psalms):
    all_words = [w for p in psalms for w in p.words]
    valences = {w.phrase_valence for w in all_words if w.phrase_valence}
    assert valences == {"core", "complement", "adjunct"}


def test_phrase_grammatical_role_is_populated_for_a_minority_of_words(psalms):
    all_words = [w for p in psalms for w in p.words]
    with_role = [w for w in all_words if w.phrase_grammatical_role]
    without = [w for w in all_words if w.phrase_grammatical_role == ""]
    assert with_role
    assert without
    assert "direct_object" in {w.phrase_grammatical_role for w in with_role}


def test_verb_sense_is_only_populated_on_verb_occurrences(psalms):
    all_words = [w for p in psalms for w in p.words]
    with_sense = [w for w in all_words if w.verb_sense]
    assert with_sense
    assert all(w.part_of_speech == "verb" for w in with_sense)


def test_every_psalm_has_half_verses(psalms):
    assert all(p.half_verses for p in psalms)


def test_psalm_1_has_the_expected_half_verse_count(psalms):
    # BHSA's `half_verse` sectional division isn't strictly binary per verse - most verses split.
    psalm_1 = next(p for p in psalms if p.number == 1)
    assert len(psalm_1.half_verses) == 14
    assert len(psalm_1.half_verses) >= psalm_1.verse_count


def test_half_verses_are_nonempty_vocalized_hebrew_text(psalms):
    all_half_verses = [hv for p in psalms for hv in p.half_verses]
    assert all(hv.strip() for hv in all_half_verses)
    # Vocalized (has niqqud, not bare consonants) - a real Hebrew vowel point (e.g.
    assert any("ָ" in hv for hv in all_half_verses)


def test_half_verses_unvocalized_are_nonempty_consonantal_hebrew_text(psalms):
    all_half_verses = [hv for p in psalms for hv in p.half_verses_unvocalized]
    assert all(hv.strip() for hv in all_half_verses)
    # Consonantal only, so a true vowel point must be absent.
    assert not any("ָ" in hv for hv in all_half_verses)


def test_half_verses_unvocalized_count_matches_vocalized_count(psalms):
    # Both are rendered from the same underlying half_verse nodes.
    assert all(len(p.half_verses_unvocalized) == len(p.half_verses) for p in psalms)


def test_psalm_14_and_53_share_a_near_identical_opening_half_verse(psalms):
    # The twin-psalm pair already used elsewhere as ground truth.
    psalm_14 = next(p for p in psalms if p.number == 14)
    psalm_53 = next(p for p in psalms if p.number == 53)
    assert psalm_14.half_verses[1] == psalm_53.half_verses[1]


def test_half_verses_niqqud_only_are_nonempty_hebrew_text_with_niqqud_but_no_accents(psalms):
    all_half_verses = [hv for p in psalms for hv in p.half_verses_niqqud_only]
    assert all(hv.strip() for hv in all_half_verses)
    # Niqqud (vowel points) must survive - same QAMATS check as the fully vocalized field.
    assert any("ָ" in hv for hv in all_half_verses)
    # Cantillation marks (U+0591-U+05AF) must be absent, ETNAHTA being a common one.
    assert any("֑" in hv for hv in [hv for p in psalms for hv in p.half_verses])
    assert not any(any("֑" <= ch <= "֯" for ch in hv) for hv in all_half_verses)


def test_half_verses_niqqud_only_count_matches_vocalized_count(psalms):
    # Same underlying half_verse nodes, just accent-stripped - must line up one-to-one per psalm.
    assert all(len(p.half_verses_niqqud_only) == len(p.half_verses) for p in psalms)


def test_every_psalm_has_half_verse_nodes(psalms):
    assert all(p.half_verse_nodes for p in psalms)


def test_half_verse_nodes_count_matches_half_verses_count(psalms):
    # The same real BHSA half_verse nodes the other half-verse accessors return.
    assert all(len(p.half_verse_nodes) == len(p.half_verses) for p in psalms)


def test_half_verse_nodes_are_real_distinct_integers_across_the_whole_corpus(psalms):
    all_nodes = [node for p in psalms for node in p.half_verse_nodes]
    assert all(isinstance(node, int) for node in all_nodes)
    # Text-Fabric node ids are globally unique - no two half-verses, even in different psalms.
    assert len(all_nodes) == len(set(all_nodes))


def test_psalm_14_and_53_have_different_half_verse_node_ids_for_their_shared_text(psalms):
    # The twin-psalm pair shares near-identical text at this half-verse.
    psalm_14 = next(p for p in psalms if p.number == 14)
    psalm_53 = next(p for p in psalms if p.number == 53)
    assert psalm_14.half_verse_nodes[1] != psalm_53.half_verse_nodes[1]
