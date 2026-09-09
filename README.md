# tehillim-compare

## Overview

`tehillim-compare` computes psalm-by-psalm similarity matrices for the 150 Psalms. It is the upstream representation layer in the split Tehillim project. It derives lexical, morphological, and syntactic feature profiles from ETCBC/BHSA data, combines them with half-verse embedding representations, and publishes the matrices for `tehillim-cluster` to consume. Its output boundary is a documented similarity relation between every unordered pair of psalms. Partitioning, historical classification, and cluster evaluation belong to `tehillim-cluster`.

## Data

The repository reads the 2021 Text-Fabric representation of the ETCBC [Biblia Hebraica Stuttgartensia Amstelodamensis](https://github.com/ETCBC/bhsa), together with the ETCBC valence module. It selects the book `Psalmi` and treats each of its 150 chapters as one psalm. The extracted record preserves psalm number, verse count, incipit, and word occurrences. Word occurrences carry ETCBC values for lexeme, root, part of speech, verbal stem and conjugation, grammatical person, clause text type and relation, phrase type and function, and selected valence fields.

The same corpus provides 5,203 `half_verse` section nodes. BHSA defines this as a Masoretic verse-internal section with labels `A`, `B`, or `C`. It is the alignment unit for semantic input, not a linguistic definition of the colon. Cantillated, niqqud-only, and consonantal renderings are three representations of the same BHSA encoding of Codex Leningradensis.

The semantic artifacts reside in a local `tehillim-embeddings` checkout and join to this corpus through BHSA node identifiers. Each configured artifact must cover the extracted Psalter. The run stops when an artifact is absent. That rule preserves a stable representation inventory. BHSA and valence annotations encode prior analytical decisions, and the source contains no critical apparatus, alternative textual witnesses, or second linguistic database. These limits are properties of the input, not residual implementation detail.

## Methodology

The analytical unit is the whole psalm. Each representation yields a 150-row feature matrix and 11,175 unordered pairwise scores. Eleven corpus-derived methods preserve separate inventories for content-word lexemes, roots, lexical sets, named-entity identity and type, verb morphology, grammatical person, clause type, clause text type, clause relation, and valence-based verbal complementation. This design registers formal distributions before any statement about theme, genre, rhetorical function, or historical setting.

For the corpus-derived methods, the repository applies TF-IDF weighting and cosine similarity. TF-IDF records term frequency within a psalm and discounts terms shared widely across the Psalter. Cosine then measures proximity between the resulting vectors. The method identifier and description travel with every matrix, so a score remains tied to the representation that produced it.

The semantic family contains 80 configured aggregations over half-verse vectors. Mean pooling forms a float64 psalm mean before cosine comparison. Soft alignment compares both half-verse sets, averages each set's best matches in the other, and symmetrizes the two directions. Top-principal-component removal and whitening are retained as separately named transformations for three contextual encoders. Encoder choice, textual rendering, alignment unit, and aggregation rule are analytic decisions. Computation exposes their consequences. It does not adjudicate an interpretation.

## Results

No versioned production similarity dataset is committed to this repository. A complete run writes `analysis=compare/stage=raw/psalms.parquet` and, within each of the lexical, morphological, syntactic, and semantic domains, `similarity.parquet` and `methods.parquet`. The configuration therefore produces 91 named matrices, each with 11,175 pairwise scores. These are research artifacts for inspection and for the downstream clustering step.

The retained feature inventory rests on explicit development checks. Named-entity identity yields scores below 0.5 for 60 percent of psalm pairs, and clause type for 66.7 percent. Root values occur in approximately 20 percent of Psalter word occurrences. Valence verb-sense values cover 60.6 percent of verb occurrences. Lexical similarity ranks Psalm 53 first for Psalm 14 at 0.811 and identifies Psalms 57 and 60 as close to Psalm 108. These are checks on coverage and registered textual reuse. They are not validation of a general theory of similarity.

The repository also preserves negative results through its exclusions. Gender, nominal-state, clause-kind, phrase-type, phrase-function, phrase-determination, phrase-valence, and phrase-role profiles produce heavily compressed whole-psalm distributions. Their 0 to 4.9 percent rate of scores below 0.5 falls below the retained methods' 15 percent threshold. The result concerns this representation, this similarity measure, and this aggregation scale. It leaves their possible value for local or sequential analysis open.

## Limitations

Whole-psalm aggregation removes order, local parallelism, changes of speaker, and discourse transitions. The half-verse carrier has fixed corpus-wide alignment, although its Masoretic boundaries can be coarser than clauses and other linguistic units. Similarity scores inherit that granularity.

ETCBC feature values are an analysis of the text. Morphological, phrase, and clause layers make different theoretical commitments. The valence module covers only a documented subset of verbal occurrences. Pretrained embeddings add a further opaque representation with its own training data and tokenization. A matrix can describe regularities in these encodings. It cannot recover genre, authorship, date, theology, or literary meaning.

## Reproducibility

The package requires Python 3.12 or later with Text-Fabric, NumPy, PyArrow, and scikit-learn. An audited reconstruction requires the BHSA and valence release identifiers, the embedding artifact tree, the repository revision, command options, and hashes of the emitted Parquet tables. The default checkout is `latest`. Pass an explicit `--checkout` value or a local `TEHILLIM_BHSA_PATH` for a versioned run.

Provide the matching embedding tree through `--embeddings-dir` or `TEHILLIM_EMBEDDINGS_DIR`. Select the output root with `--data-root` or `TEHILLIM_DATA_DIR`. The cache fingerprints vector content, while it does not retain the BHSA release or code revision. Clear it before an audited rerun and preserve those identifiers with the outputs. `./check.sh` exercises formatting, static analysis, dependency checks, dead-code analysis, and unit tests under fixtures. It repeats the computation, not the scholarly judgment embedded in the source annotations.

## Installation

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Usage

```bash
.venv/bin/tehillim-compare \
  --checkout 2021 \
  --embeddings-dir /path/to/tehillim-embeddings/data/domain=semantic \
  --data-root /path/to/tehillim-data

./check.sh
```

The command writes its raw tables under `/path/to/tehillim-data/analysis=compare/`. Run `tehillim-cluster` against the same data root after the compare analysis completes.

## References

ETCBC. [*Biblia Hebraica Stuttgartensia Amstelodamensis*](https://github.com/ETCBC/bhsa), Text-Fabric data, release 2021.

Mu, Jiaqi, Suma Bhat, and Pramod Viswanath. [“All-but-the-Top: Simple and Effective Postprocessing for Word Representations.”](https://openreview.net/forum?id=HkuGJ3kCb) In *International Conference on Learning Representations*, 2018.

Roorda, Dirk. [“Text-Fabric: Handling Biblical Data with IKEA Logistics.”](https://doi.org/10.7146/hn.v5i2.142740) *HIPHIL Novum* 5, no. 2 (2019): 126–135.

Spärck Jones, Karen. [“A Statistical Interpretation of Term Specificity and Its Application in Retrieval.”](https://doi.org/10.1108/EB026526) *Journal of Documentation* 28, no. 1 (1972): 11–21.

## License

The source code is released under the MIT License. BHSA, the ETCBC valence module, and embedding artifacts remain subject to their own licences and access conditions.
