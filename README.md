# Amazon ML Challenge: Business Entity Resolution

This repository contains a multi-stage system for matching business records across three data sources. The final submitted version reported a public leaderboard macro F0.5 score of **0.9745**. Results are reported from the challenge submission and have not been independently verified here.

## Approach

The pipeline combines learned text normalization, blocking-based candidate generation, reverse matching, LightGBM classifiers, MiniLM cross-encoders, multilingual embedding search, and a one-to-one decision layer. Candidate generation is designed to avoid comparing every possible record pair.

The detailed implementation guide, run order, environment notes, and validation summary are in [`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md). The longer solution write-up is in [`Documentation_template.md`](Documentation_template.md).

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

The pipeline scripts are run in order from `code/business_entity_resolution/`; the complete list is in the project README. For example:

```powershell
cd code/business_entity_resolution
python src/steps/s01_prepare_data.py
```

The later cross-encoder and embedding steps require a CUDA-capable GPU; embedding search also needs substantial memory. Provide the competition datasets separately and configure the data, working, and output paths as described in the project README and `src/common.py`. Competition data, model weights, intermediate artifacts, and generated TSV results are intentionally excluded from this repository.

> `requirements.txt` is currently empty, so a fresh environment is not yet reproducible from that file alone. Check the imports and environment notes in the project README before running the pipeline.

## Results

The reported final run produced `matching_results.tsv` and `candidate_pairs.tsv`. The former contains the match output; the latter contains scored candidate pairs. These files are generated artifacts and are not included because of their size. The methodology document describes the reported validation and leaderboard results.

## Data and Usage

This repository contains the implementation and accompanying write-up, not the challenge dataset. Obtain and use challenge data only under the applicable competition terms. No license for reuse of this repository's code is granted by this README.