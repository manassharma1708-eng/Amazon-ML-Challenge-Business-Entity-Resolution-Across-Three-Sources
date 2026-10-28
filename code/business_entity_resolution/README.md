# Business Entity Resolution — Amazon ML Challenge 2026

Matches every Source 1 business to its Source 2 / Source 3 records using only the provided data.
Final submission (v12b): public leaderboard macro F0.5 = 0.9745, 12.71 candidates per Source 1 entity.

## Pipeline

1. Cleaning: ftfy repair, anyascii transliteration, name/address normalisation, plus a word map and state aliases
   learned from training pairs only.
2. Blocking (S1 side): compound keys (rare name word x rare address word / house number / state, address-only,
   name-only for empty addresses), block cap 200, candidates ranked by sum of 1/log2(block size + 1), top 10 kept.
3. Reverse blocking: every S2/S3 record ranks its top-3 S1 shops with the same keys.
4. Stage 1: LightGBM on 40 language-independent pair features.
5. Stage 2: LightGBM on stage-1 probability + in-shop context + error-analysis + reverse features.
6. Stage 3: MiniLM-L6 cross-encoder on uncertain pairs + LightGBM stacker.
7. Stage 4: MiniLM-L12 cross-encoder (trained on 1.2M pairs) stacked on the stage-3 probability.
8. Union add-on 1: a record's reverse #1 shop, when missing from that shop's top 10.
9. Union add-on 2: an unmatched record's reverse #2/#3 shops.
10. Embedding search: an unmatched record's 3 nearest S1 shops (multilingual-e5-small), cheap pre-ranker,
    L12 cross-encoder + add-on 3; accepted for India/US at p >= 0.9.
11. Decisions: one-to-one (each record goes to at most one shop) + per-shop expected-F0.5 selection.

`candidate_pairs.tsv` = every pair the models ran inference on (top 10 + all scored union and embedding pairs).
Every match is a subset of the candidates.

## Environment

Python 3.13, packages pinned in `requirements.txt` (`pip install -r requirements.txt`).
Steps 9, 10, 12, 13, 14 need a CUDA GPU (a T4 is enough; we used Colab and Kaggle T4s). Step 14 needs ~30 GB RAM
(Kaggle). All other steps run on CPU with ~12.7 GB RAM.
Pretrained weights downloaded from Hugging Face: cross-encoder/ms-marco-MiniLM-L-6-v2 and -L-12-v2 (Apache 2.0),
intfloat/multilingual-e5-small (MIT). No external data.

## Data layout

Paths are set in `src/common.py` and can be overridden with environment variables:

| Variable | Default (Colab) | Contents |
|---|---|---|
| `ER_DATA` | `/content/drive/MyDrive/amazon_ml/student_resource/dataset` | `train/*.tsv`, `test/*.tsv` |
| `ER_WORK` | `/content/drive/MyDrive/amazon_ml/artifacts` | intermediate files and models |
| `ER_OUT`  | `/content/drive/MyDrive/amazon_ml/output` | the two output TSVs |

## Run order (from `code/business_entity_resolution/`)

| Step | Command | Approx. time | Hardware |
|---|---|---|---|
| 1 | `python src/steps/s01_prepare_data.py` | 5 min | CPU |
| 2 | `python src/steps/s02_learn_maps.py` | 10 min | CPU |
| 3 | `python src/steps/s03_clean.py` | 27 min | CPU |
| 4 | `python src/steps/s04_folds.py` | 1 min | CPU |
| 5 | `python src/steps/s05_blocking.py` | 65 min | CPU |
| 6 | `python src/steps/s06_reverse_blocking.py` | 105 min | CPU |
| 7 | `python src/steps/s07_stage1.py` | 110 min | CPU |
| 8 | `python src/steps/s08_stage2.py` | 60 min | CPU |
| 9 | `python src/steps/s09_cross_encoder.py` | 50 min | GPU |
| 10 | `python src/steps/s10_union_addon.py` | 25 min | GPU |
| 12 | `python src/steps/s12_ce_v3_stage4.py` | 90 min | GPU |
| 13 | `python src/steps/s13_union_addon2.py` | 40 min | GPU |
| 14 | `python src/steps/s14_embedding_search.py` | 100 min | GPU, ~30 GB RAM |
| 15 | `python src/steps/s15_write_outputs.py` | 5 min | CPU |

(There is no step 11; the final writer is step 15.) Heavy steps work in chunks and skip finished parts.

## Validation

Training S1 shops: 5 folds (stratified by country). Stage 1 trains on fold 1; later stages use fold 0 split into
A (60%) / B1 (20%) / B2 (20%) shops. B2 is never used for training or tuning.
B2 macro F0.5: stage 1 0.9413 -> stage 2 0.9561 -> cross-encoder 0.9750 -> union add-on 0.9774.
Later gains on B2: stage 4 +0.0005 (B2b), union add-on 2 +0.0013, embedding extras +0.0044.

## Settings chosen with public-leaderboard feedback

The embedding extras' final threshold (0.9 instead of the B1 optimum 0.6) and the exclusion of France (no French
training examples) were chosen by comparing public-leaderboard submissions (v12a-d, see the methodology document).

## Notes

- Seeds are fixed; GPU training can vary slightly between runs, so a re-run may differ marginally from the submitted file.
- `names.py` / `addresses.py` contain France-only cleaning rules that are switched off (`FRANCE_RULES = False`).
