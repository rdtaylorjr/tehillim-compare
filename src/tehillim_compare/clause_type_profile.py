"""Builds a psalm-by-clause-type tag-count matrix from BHSA's `typ` pattern codes."""

from __future__ import annotations

from tehillim_compare.corpus import Psalm, PsalmWord
from tehillim_compare.features import FeatureInfo, FeatureMatrix, build_tag_count_feature_matrix

#: BHSA clause-type codes -> human-readable labels.
_CLAUSE_TYPE_LABELS: dict[str, str] = {
    "AjCl": "Adjective Clause",
    "CPen": "Casus Pendens",
    "Defc": "Defective Clause Atom",
    "Ellp": "Ellipsis",
    "InfA": "Infinitive Absolute Clause",
    "InfC": "Infinitive Construct Clause",
    "MSyn": "Macrosyntactic Sign",
    "NmCl": "Nominal Clause",
    "Ptcp": "Participle Clause",
    "Reop": "Reopening",
    "Unkn": "Unknown",
    "Voct": "Vocative Clause",
    "Way0": "Wayyiqtol-Null Clause",
    "WayX": "Wayyiqtol-X Clause",
    "WIm0": "We-Imperative-Null Clause",
    "WImX": "We-Imperative-X Clause",
    "WQt0": "We-Qatal-Null Clause",
    "WQtX": "We-Qatal-X Clause",
    "WxI0": "We-X-Imperative-Null Clause",
    "WXIm": "We-X-Imperative Clause",
    "WxIX": "We-X-Imperative-X Clause",
    "WxQ0": "We-X-Qatal-Null Clause",
    "WXQt": "We-X-Qatal Clause",
    "WxQX": "We-X-Qatal-X Clause",
    "WxY0": "We-X-Yiqtol-Null Clause",
    "WXYq": "We-X-Yiqtol Clause",
    "WxYX": "We-X-Yiqtol-X Clause",
    "WYq0": "We-Yiqtol-Null Clause",
    "WYqX": "We-Yiqtol-X Clause",
    "xIm0": "X-Imperative-Null Clause",
    "XImp": "X-Imperative Clause",
    "xImX": "X-Imperative-X Clause",
    "XPos": "Extraposition",
    "xQt0": "X-Qatal-Null Clause",
    "XQtl": "X-Qatal Clause",
    "xQtX": "X-Qatal-X Clause",
    "xYq0": "X-Yiqtol-Null Clause",
    "XYqt": "X-Yiqtol Clause",
    "xYqX": "X-Yiqtol-X Clause",
    "ZIm0": "Zero-Imperative-Null Clause",
    "ZImX": "Zero-Imperative-X Clause",
    "ZQt0": "Zero-Qatal-Null Clause",
    "ZQtX": "Zero-Qatal-X Clause",
    "ZYq0": "Zero-Yiqtol-Null Clause",
    "ZYqX": "Zero-Yiqtol-X Clause",
}


def clause_type_words(words: tuple[PsalmWord, ...]) -> list[PsalmWord]:
    """Return the subset of `words` carrying a clause-type tag."""
    return [w for w in words if w.clause_type]


def build_clause_type_feature_matrix(psalms: list[Psalm]) -> FeatureMatrix:
    """Build a psalm x clause-type tag-count matrix."""

    def tags_of(word: PsalmWord) -> list[tuple[str, FeatureInfo]]:
        """The tag this word contributes, described the way the profile displays it."""
        tag = word.clause_type
        label = _CLAUSE_TYPE_LABELS.get(tag, tag)
        return [
            (
                tag,
                FeatureInfo(
                    label=label, description=f"{label} (clause type)", category="clause-type"
                ),
            )
        ]

    return build_tag_count_feature_matrix(psalms, clause_type_words, tags_of)
