# Amazon ML Challenge: Business Entity Resolution Across Three Sources

A multi-stage pipeline that matches business records across three data sources. The final submission scored a public leaderboard macro F0.5 of **0.9745**. This figure comes from the challenge submission and has not been independently verified here.

## Approach

1. **Cleaning:** text repair, transliteration, and name/address normalization, using word maps and state aliases learned from the training pairs only.
2. **Candidate generation:** blocking on compound name/address keys, plus reverse blocking so that every Source 2/3 record proposes its likeliest Source 1 shops. This avoids comparing every possible pair.
3. **Scoring:** two LightGBM stages, then MiniLM cross-encoders (L6 and L12) stacked on top.
4. **Recall add-ons:** union add-ons and a multilingual-e5 embedding search to recover matches that blocking missed.
5. **Decisions:** a one-to-one assignment layer with per-shop expected-F0.5 selection.

For the full implementation guide, run order, environment notes, and validation summary, see [`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md). The longer solution write-up is in [`Documentation_template.md`](Documentation_template.md).

## Repository Layout

```text
code/business_entity_resolution/
  src/                 Pipeline modules and runnable steps
  requirements.txt     Dependency file (currently empty)
  README.md            Detailed pipeline and execution notes
output/                Generated result files (not committed)
Documentation_template.md
```

## Running the Pipeline

Run the scripts in order from `code/business_entity_resolution/`. The complete list is in the project README. For example:

```powershell
cd code/business_entity_resolution
python src/steps/s01_prepare_data.py
```

The later cross-encoder and embedding steps need a CUDA-capable GPU, and embedding search also needs substantial memory. Provide the competition datasets yourself and set the data, working, and output paths as described in the project README and `src/common.py`.

> **Note:** `requirements.txt` is currently empty, so a fresh environment can't yet be reproduced from it alone. Check the imports and environment notes in the project README before running.

## Results

The reported final run produced `matching_results.tsv` (the matches) and `candidate_pairs.tsv` (the scored candidate pairs). These generated files are not included because of their size. The methodology document describes the reported validation and leaderboard results.

## Data and Usage

This repository contains the implementation and the accompanying write-up, not the challenge dataset. The competition data, model weights, and intermediate artifacts are intentionally excluded. Use challenge data only under the applicable competition terms. No license for reuse of this repository's code is granted by this README.
This repository contains the implementation and accompanying write-up, not the challenge dataset. Obtain and use challenge data only under the applicable competition terms. No license for reuse of this repository's code is granted by this README.
