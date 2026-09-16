# tehillim-compare

## Overview

`tehillim-compare` computes psalm-by-psalm similarity matrices from every representation in
`tehillim-embeddings`. It is the comparison layer of the Tehillim project. It records a similarity
relation for every unordered pair of the 150 Psalms and supplies those matrices to
[`tehillim-cluster`](https://github.com/rdtaylorjr/tehillim-cluster).

## Data

The repository reads the 2021 Text-Fabric representation of the ETCBC
[Biblia Hebraica Stuttgartensia Amstelodamensis](https://github.com/ETCBC/bhsa). It treats the
150 chapters of `Psalmi` as psalms and uses its 5,203 `half_verse` nodes as the input unit.
`half_verse` is a Masoretic accentual section, labelled `A`, `B`, or `C`, and supplies a stable
address for the calculation. It does not define a poetic colon.

Representations are read only from the BHSA half-verse scope,
`corpus=bhsa/unit=half_verse/`, in the embeddings Parquet tree. This scope matches the corpus and
node identifiers of the psalm data. The loader discovers lexical, morphological, syntactic, and
semantic files within that scope, including dense and sparse vectors, and retains the description
recorded with each file. BHSA features and model embeddings encode prior analytical decisions.

## Methodology

Each representation produces a 150-row psalm matrix and 11,175 unordered pairwise scores when all
psalms have usable vectors. Mean pooling averages a psalm's half-verse vectors in float64 and
compares the resulting rows with cosine similarity. Soft alignment compares the two half-verse
sets, averages each set's best cosine matches in the other, and symmetrizes the result.

AlephBERT, BEREL, and NeoDictaBERT also receive top-principal-component removal and whitening
before soft alignment. Model, text form, aggregation, and correction remain in the method
identifier and output metadata. A matrix records the behavior of a representation under these
operations. It does not establish genre, authorship, date, theology, or literary meaning.

## Results

A run writes psalm facts to `analysis=compare/stage=raw/psalms.parquet` and writes
`similarity.parquet` and `methods.parquet` under each
`analysis=compare/domain={domain}/stage=raw/` partition. The matrices are generated artifacts for
inspection and for clustering. They are not committed to this source repository.

## Limitations

Whole-psalm comparison suppresses order, local parallelism, speaker shifts, and discourse
transitions. Scores inherit the accentual granularity of the half-verse carrier, the ETCBC
analysis, and the assumptions of each embedding model and tokenizer. A similarity matrix can
register regularities in those encodings. Interpretation requires philological and literary
judgment.

## Reproducibility

The package requires Python 3.12 or later, Text-Fabric, NumPy, SciPy, PyArrow, scikit-learn, and a
matching `tehillim-embeddings` checkout. An audited run requires the BHSA release, embeddings
revision, command arguments, and output files. Use `--checkout 2021` or `TEHILLIM_BHSA_PATH` for a
versioned BHSA source instead of the default `latest` checkout.

## Installation

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ../tehillim-embeddings -e ".[dev]"
```

## Usage

Write the psalm facts, then score one embeddings domain:

```bash
.venv/bin/tehillim-compare psalms --checkout 2021 --data-root /path/to/tehillim-data
.venv/bin/tehillim-compare domain --checkout 2021 --domain lexical \
  --data-root /path/to/tehillim-data \
  --embeddings-root /path/to/tehillim-embeddings/data
```

Run the domain command for each discovered representation domain before clustering.

## References

ETCBC. [*Biblia Hebraica Stuttgartensia Amstelodamensis*](https://github.com/ETCBC/bhsa),
Text-Fabric data, release 2021.

Mu, Jiaqi, and Pramod Viswanath. [“All-but-the-Top: Simple and Effective Postprocessing for Word
Representations.”](https://openreview.net/forum?id=HkuGJ3kCb) In *International Conference on
Learning Representations*, 2018.

Roorda, Dirk. [“Text-Fabric: Handling Biblical Data with IKEA
Logistics.”](https://doi.org/10.7146/hn.v5i2.142740) *HIPHIL Novum* 5, no. 2 (2019): 126-135.

Su, Jianlin, Jiarun Cao, Weijie Liu, and Yangyiwen Ou. [“Whitening Sentence Representations for
Better Semantics and Faster Retrieval.”](https://arxiv.org/abs/2103.15316) arXiv:2103.15316, 2021.

## License

MIT. BHSA and embedding artifacts have separate terms of use.
