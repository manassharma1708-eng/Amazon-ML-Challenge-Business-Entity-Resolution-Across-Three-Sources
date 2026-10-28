# Business Entity Resolution — Methodology

## 1. Overview

We match every Source 1 (S1) business to all Source 2 / Source 3 records of the same real business using only the
provided data. Final submission: **public leaderboard macro F0.5 = 0.9745**, with **12.71 candidates per S1 entity**
(reduction ratio 99.9997%).

The system: (1) text normalisation with maps learned from training pairs; (2) candidate generation from three
directions — S1-side compound keys, reverse keys (each record ranks S1 shops) and multilingual embedding search;
(3) a stacked matcher — LightGBM stages 1–2, MiniLM cross-encoders (L6, then L12) with LightGBM stackers, and three
add-on models for the extra candidates; (4) a decision layer that maximises expected F0.5 per shop under one-to-one.

### 2.1 Problem Understanding

| Set | S1 | S2 | S3 | Countries |
|---|---|---|---|---|
| Train | 2,206,821 | 5,034,616 | 5,285,603 | US 60%, India 40% |
| Test | 1,732,544 | 4,887,273 | 5,082,316 | India, US, France (France unseen in training) |

- **7,638,365 true pairs**; shops have mostly 2–6 matches (mean about 3.5, max 11); 5.58% are singletons.
- **Strictly one-to-one**: no S2/S3 record matches more than one S1 shop; all true pairs share the country.
- About **26% of S2/S3 records are distractors**, and many businesses share generic names, so a name identifies a
  business only together with a location.
- Noise: ALL CAPS, typos and digit-for-letter swaps (`gr0up`, `8lanks`), truncation, reordering, legal-suffix variants,
  filler words, domains/handles, acronyms, empty addresses (about 3.5%), number typos, city aliases,
  **regional-script names (about 28% of India S2 names)** and state names (about 23% of India addresses), broken encoding.

### 2.2 Solution Strategy

**Approach Type:** Hybrid — multi-source blocking (keys + reverse keys + embeddings) with stacked classifiers
(gradient boosting + fine-tuned transformer cross-encoders) and a set-level decision layer.

**Core Innovation:**
1. **Reverse direction.** S1 is deduplicated, so each record's own ranking of S1 shops is uncrowded: 98.2% of true pairs
   are the record's first choice (17.1% of non-matches). Used as features and to recover missed copies (union add-ons).
2. **Cross-encoders only where needed.** Only pairs with stage-1 probability in 0.01–0.99 (about 24%) are read by the
   cross-encoder — the largest single gain (+0.024 on the leaderboard).
3. **Embedding search for "keyless" copies.** 3.06% of true copies share no usable key. Multilingual embeddings put the
   true shop among the 3 nearest neighbours for 46.3% of them; a pre-ranker keeps 97% of true copies with 22% of the pairs.
4. **Expected-F0.5 decisions per shop** under a one-to-one constraint.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used** (country-prefixed 64-bit hashes):
  - *Name × address:* the 2 rarest name words and the first 5 letters of the space-free name, each combined with the
    3 rarest address words, the 2 longest house numbers and the state code.
  - *Address only:* number × rare address word, and pairs of rare address words.
  - *Name only:* name-word pairs and single name parts (S2/S3 side only for records with an empty address).
  - Blocks larger than 200 are dropped; candidates ranked by Σ 1/log2(block size + 1); **top 10 per shop kept**.
  - *Reverse:* each record's top-3 S1 shops by the same keys. #1 is added when missing from the top 10 (union add-on 1);
    #2/#3 for still-unmatched records after a pre-filter that keeps 99% of true copies (union add-on 2).
  - *Embeddings:* `intfloat/multilingual-e5-small` (mean-pooled, cosine) — each still-unmatched record's 3 nearest
    S1 shops in the same country, excluding pairs already considered, then a LightGBM pre-ranker.
  - Learned normalisation: 581 transliteration word mappings and 15 regional-script state aliases from training pairs.
- **Candidate pairs generated (test):** 17,287,081 top-10 pairs + 825,625 union-1 + 2,185,366 union-2 + 1,723,746
  embedding pairs = **about 22.0 million, 12.71 per S1 entity**. All-pairs within country would be about 6.7 × 10^12
  (**reduction ratio 99.9997%**). All joins use sorted hash indexes; nothing compares all pairs.
- **How we ensured true matches were not lost:** recall was measured on labelled folds at every step, and misses
  were inspected by hand.

| Candidate set (fold 0) | Pair recall | Candidates/shop | F0.5 ceiling |
|---|---|---|---|
| S1-side top 10 | 95.44% | 10.0 | 0.9840 |
| + reverse #1 | 96.34% | 10.25 | 0.9871 |
| + reverse #2/#3 (all) | 96.92% | 16.63 | 0.9892 |
| Embedding top 3 of the remaining misses | +46.3% of the 3.06% missed | – | – |

---

## 4. Matching Model

**Features used:**
- **Name features:** token-set/sort and Levenshtein ratios (core and full name), partial ratio and Jaro-Winkler
  (space-free), IDF-weighted Jaccard, first-word equality, acronym match, word counts, legal forms and
  **legal-form conflict**, digit-to-letter repaired similarity, exact-name flag and name frequency among S1 / S2-S3.
- **Address features:** token-set/partial ratios, IDF-weighted Jaccard, shared numbers, number Jaccard, main-number
  equality, "one digit off" numbers, state equality, empty-address flag, landmark flags.
- **Other:** blocking score, shared keys and rank; in-shop context (rank/gap/share of similarities and probabilities);
  reverse rank/score/margin; cross-encoder scores on raw "name | address" text; embedding similarity, rank, gap and
  shop crowding; shop-level context (best probability, number of accepted matches). No country feature.

**Model type:**
- Stage 1: LightGBM, 40 features (AUC 0.9972). Stage 2: LightGBM with context + reverse features (AUC 0.9986).
- Stage 3: `cross-encoder/ms-marco-MiniLM-L-6-v2` (Apache 2.0, 22.7M parameters), 2 epochs on 473k uncertain pairs,
  + LightGBM stacker. Stage 4: `ms-marco-MiniLM-L-12-v2` (33M), 1 epoch on 1.2M pairs, + LightGBM stacker.
- Add-ons 1–3: LightGBM on cross-encoder score, reverse/embedding features and shop context (AUC 0.993 / 0.996 / 0.998).

**Threshold selection method:** per-shop expected-F0.5 selection (top-k including k = 0) after a one-to-one rule;
add-on thresholds tuned on B1 shops, results reported on untouched B2 shops. For the embedding add-on, the final
threshold (0.9) and the exclusion of France were chosen with public-leaderboard feedback (Appendix B).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9774** on untouched validation shops (B2) for the v8 system; later additions measured
  +0.0005 (stage 4), +0.0013 (union add-on 2) and +0.0044 (embedding extras) on B2. **Public leaderboard: 0.9745.**

| Version | Change | Public leaderboard |
|---|---|---|
| v1 | Stage 1 + decision rule | 0.9284 |
| v3 | + stage 2 + error-analysis features | 0.9370 |
| v5 | + reverse features | 0.9381 |
| v6 | + cross-encoder L6 | 0.9623 |
| v7 | + wider band, 2 epochs | 0.9650 |
| v8 | + union add-on 1 | 0.9684 |
| v9 | + cross-encoder L12, stage 4 | 0.9691 |
| v10 | + union add-on 2 | 0.9706 |
| v11 | + embedding extras, all countries, p ≥ 0.6 | 0.9703 |
| **v12b** | **embedding extras, India + US, p ≥ 0.9** | **0.9745** |

- **Common false positives (wrong merges):** same-name shops with empty-address copies (e.g. one name in two cities);
  deliberate look-alikes (same street, changed house number); generic names at shared buildings; and, for embedding
  extras, missed copies of *other* shops with near-identical names — the reason the stricter threshold helped.
- **Common false negatives (missed matches):** copies with garbled or transliterated names *and* rewritten addresses
  (no shared key); empty-address copies of shared-name shops (left unassigned on purpose, since F0.5 punishes wrong
  merges more); French copies, which we do not add through the embedding route.
- **Lesson:** our validation queries left out records owned by non-validation shops, so the embedding add-on never saw
  the hardest negatives and validation overstated its precision (test accepted 0.074 extras/shop vs 0.047 on B2).
  Diagnosing by country revealed France (no training data) as the main source of errors.
- **Tried and not kept:** France-only cleaning rules; near-duplicate expansion (12% precision); a global strictness
  shift; adaptive candidate cut-offs; larger K; French embedding extras at any threshold.

---

## 6. Conclusion

Learned normalisation, candidates from three complementary directions (keys, reverse keys, embeddings) and a stacked
matcher that sends only uncertain pairs to cross-encoders reach 0.9745 with about 13 candidates per shop. The key
lessons: measure every idea on untouched shops, make validation as hard as the test (hard negatives), and inspect errors
by segment — the country breakdown turned a regression (v11) into our best result (v12b).

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`: `src/` (modules + `steps/`), `README.md` (run order, timings, hardware),
`requirements.txt`. Run from `code/business_entity_resolution/`, in order:
`s01_prepare_data` → `s02_learn_maps` → `s03_clean` → `s04_folds` → `s05_blocking` → `s06_reverse_blocking` →
`s07_stage1` → `s08_stage2` → `s09_cross_encoder` (GPU) → `s10_union_addon` (GPU) → `s12_ce_v3_stage4` (GPU) →
`s13_union_addon2` (GPU) → `s14_embedding_search` (GPU) → `s15_write_outputs`, which writes
`output/matching_results.tsv` and `output/candidate_pairs.tsv` and runs the validator.

### B. Additional Results

**Embedding extras on the test set, by threshold and country (public leaderboard):**

| Variant | Extras used | Leaderboard |
|---|---|---|
| v11 | all countries, p ≥ 0.6 (128,152) | 0.9703 |
| v12a | India + US, p ≥ 0.6 (87,402) | 0.9735 |
| **v12b** | **India + US, p ≥ 0.9 (63,892)** | **0.9745** |
| v12c | India + US, p ≥ 0.95 | 0.9744 |
| v12d | v12b + France p ≥ 0.98 | 0.9742 |

Accepted extras per shop at p ≥ 0.6: France 0.157, India 0.094, US 0.017 (validation average 0.047).

**Top-K trade-off (fold 0):** K = 10: recall 95.44%, ceiling 0.9840; K = 30: 97.16%, 0.9897; K = 100: 98.07%, 0.9929.

**Cleaning impact:** name similarity ≥ 80 for true pairs 78.3% → 85.2%; address 84.5% → 94.1% (random pairs < 0.1%).
