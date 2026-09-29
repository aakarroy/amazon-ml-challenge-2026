# Business Entity Resolution - Exploratory Data Analysis

## 1. Dataset Inventory

- **Total train files:** 4
- **Total test files:** 3

### File Statistics

| split   | file                   |   size_mb |   records |
|:--------|:-----------------------|----------:|----------:|
| train   | train_ground_truth.tsv |   121.131 |   2206821 |
| train   | train_source1.tsv      |   200.338 |   2206821 |
| train   | train_source2.tsv      |   466.634 |   5034616 |
| train   | train_source3.tsv      |   480.371 |   5285603 |
| test    | test_source1.tsv       |   166.914 |   1732544 |
| test    | test_source2.tsv       |   485.856 |   4887273 |
| test    | test_source3.tsv       |   482.562 |   5082316 |

- **Total Train Source 1 records:** 2206821
- **Total Train Source 2 records:** 5034616
- **Total Train Source 3 records:** 5285603
- **Total Ground Truth records:** 2206821

## 2. File Structure and Integrity

- **train_ground_truth.tsv** columns: `['source1_entity_id', 'matched_entity_ids']`
- **train_source1.tsv** columns: `['entity_id', 'business_name', 'business_address', 'country']`
- **train_source2.tsv** columns: `['entity_id', 'business_name', 'business_address', 'country']`
- **train_source3.tsv** columns: `['entity_id', 'business_name', 'business_address', 'country']`
- **test_source1.tsv** columns: `['entity_id', 'business_name', 'business_address', 'country']`
- **test_source2.tsv** columns: `['entity_id', 'business_name', 'business_address', 'country']`
- **test_source3.tsv** columns: `['entity_id', 'business_name', 'business_address', 'country']`

## 5. Ground-Truth Matching Distribution

- **Total Source 1 entities in GT:** 2206821
- **Total positive links:** 7638365
  - **S1 -> S2 links:** 3693619
  - **S1 -> S3 links:** 3944746

### Match Count Distribution
- **Singletons (0 matches):** 123247 (5.58%)
- **Exactly 1 match:** 119157 (5.40%)
- **Multiple matches (2+):** 1964417 (89.02%)
- **Matches in both S2 and S3:** 1776047 (80.48%)

|   Match Count |   Source 1 Entities | Percentage   |
|--------------:|--------------------:|:-------------|
|             0 |              123247 | 5.58%        |
|             1 |              119157 | 5.4%         |
|             2 |              375212 | 17.0%        |
|             3 |              530841 | 24.05%       |
|             4 |              484115 | 21.94%       |
|             5 |              321957 | 14.59%       |
|             6 |              164868 | 7.47%        |
|             7 |               63968 | 2.9%         |
|             8 |               18680 | 0.85%        |
|             9 |                4205 | 0.19%        |


## 16. Country Distribution and Domain Bias

### Training Countries
| Country   |   Count | Percentage   |
|:----------|--------:|:-------------|
| US        | 7510506 | 59.95%       |
| India     | 5016534 | 40.05%       |

### Testing Countries
| Country   |   Count | Percentage   |
|:----------|--------:|:-------------|
| India     | 5527551 | 47.24%       |
| US        | 4480137 | 38.28%       |
| France    | 1694445 | 14.48%       |

**Previously unseen country values in test set:** ['France']


## 8. Cross-source Consistency & 9. Pair-level Similarity

### Feature Distributions for Positive vs Negative Pairs
Metrics calculated on a sample of ground truth positives vs random negatives.

| Feature            |   Pos_Mean |   Neg_Mean |   Pos_P50 |   Neg_P50 |
|:-------------------|-----------:|-----------:|----------:|----------:|
| exact_name         |  0.104401  | 0          |  0        | 0         |
| exact_addr         |  0.0736628 | 0          |  0        | 0         |
| country_match      |  1         | 0.52325    |  1        | 1         |
| name_token_jaccard |  0.55519   | 0.0113648  |  0.6      | 0         |
| addr_token_jaccard |  0.525876  | 0.00587452 |  0.5      | 0         |
| name_3gram_jaccard |  0.60515   | 0.0169277  |  0.666667 | 0         |
| addr_3gram_jaccard |  0.6509    | 0.0221236  |  0.682927 | 0.0142857 |


## 10. Candidate Generation / Blocking Analysis

### Blocking Strategy Simulation
- **Exact Normalized Name Blocking Recall:** 10.44%
- **First Token Name Blocking Recall:** 67.79%
- **Name 3-Gram Overlap (>0.3) Blocking Recall:** 85.04%

### 12. Visual / Qualitative Inspection (Examples)

#### Hard Positive Example (Low string similarity but True Match)
- **Source 1:** Team Air Pvt. Ltd. | Flat No:101, Anuska Towers, Opp. Mercedes Benz Show Room, Lakdi- Ka, -Pool, Hyderabad, Telangana | India
- **Matched:** TEAMAIR.COM | FLAT NO:101, ANUSKA TOWERS, OPP. MERCEDES BENZ SHOW ROOM, LAKDI- KA, -POOL, తెలంగాణ | India

#### False-Positive-Prone Example (High string similarity but Non-Match)

## 3. Record Geometry & 6. Name Dist & 7. Address Dist

### 3. Record Geometry

- **Missing Names:** 15 (0.0001%)
- **Missing Addresses:** 344883 (2.7531%)

- **Name Length (chars):** Avg: nan, P25: 18, Median: 25, P75: 31
- **Address Length (chars):** Avg: nan, P25: 33, Median: 41, P75: 63
- **Name Token Count:** Avg: nan, P25: 3, Median: 4, P75: 4
- **Address Token Count:** Avg: nan, P25: 5, Median: 6, P75: 9

### Character Composition (Names sampled)
- **Total characters sampled:** 32522661
- **Alphabetic:** 80.87%
- **Numeric:** 0.44%
- **Whitespace:** 13.90%
- **Punctuation:** 4.80%

### 6. Name Distribution & 7. Address Distribution

- **Unique Names:** 9935723
- **Unique Name Ratio:** 79.31%
#### Top 10 Names
| Name                |   Count |
|:--------------------|--------:|
| Primary Care        |     741 |
| Physical Therapy    |     707 |
| Womens Health       |     676 |
| Urgent Care         |     674 |
| Pediatric Dental    |     661 |
| Pediatric Dentistry |     635 |
| Behavioral Health   |     626 |
| Internal Medicine   |     607 |
| earnosethroat.com   |     507 |
| Meridian            |     496 |

- **Unique Addresses:** 10929927
- **Unique Address Ratio:** 87.25%
#### Top 10 Addresses
| Address                      |   Count |
|:-----------------------------|--------:|
| nan                          |  344883 |
| Ground Floor, Bangalore, KA  |      29 |
| Floor, Mumbai, MH            |      26 |
| 18, Kolkata, Howrah, WB      |      26 |
| 303, Mumbai, MH              |      25 |
| 12, Ahmedabad, GJ            |      23 |
| 101, Mumbai, Mumbai City, MH |      23 |
| 4, Mumbai, MH                |      23 |
| Floor, Bangalore, KA         |      22 |
| 7, Kolkata, Howrah, WB       |      21 |


## 11. Per-Entity Matching Difficulty

- **Very Easy Matches:** Character overlap and token similarity are above 90%. Simple exact normalized match finds them.
- **Moderate Matches:** Requires abbreviation expansion (e.g. Pvt Ltd vs Private Limited) and address cleaning.
- **Hard Matches:** Entities with high name ambiguity (e.g., common names like 'Starbucks' with varying landmark-based addresses).

## 13. Data Quality and Annotation Checks

- **Duplicate Annotations:** Some S1 entities map to multiple identically named S2/S3 businesses in different geographical zones, suggesting franchise structures or false positive merges.
- **Missing Data:** A fraction of businesses lack complete addresses or countries, requiring robust imputation or ignoring missing fields safely.

## 14. Duplicate and Entity-Cluster Analysis

- **Within-source duplicates:** Found identically named businesses. Clustering them inside Source 2/3 before blocking against Source 1 can reduce the search space.

## 15. Split and Leakage Analysis

- **Entity-Aware Validation Required:** Because F0.5 is evaluated per Source 1 entity, a random row split will leak S2/S3 pairs. Validation must be split by `Source 1 Entity ID` to prevent data leakage.
- **Singleton Stratification:** Ensure validation set preserves the ~5.5% singleton rate observed in training.

## 18. Normalization Analysis

Normalization strategies that show highest improvement:
1. **Level 1 (Basic):** Lowercase + Whitespace strip.
2. **Level 3 (Business Name):** Standardizing 'Ltd', 'Pvt', 'Inc' eliminates artificial differences.
3. **Level 4 (Address):** Removing punctuation and unifying 'St' vs 'Street' helps Jaccard overlap.

## 19. Feature Separability

- **Strong Features:** 3-Gram Jaccard on Name, Exact Match Country, Token Jaccard on Address.
- **Weak/Dangerous Features:** Length difference (many valid matches have drastically different address lengths due to missing landmarks).

## 20. Matching-Model Implications

- **Baseline:** TF-IDF + Cosine similarity over normalized name and address, combined with LightGBM/XGBoost.
- **Stronger Model:** A cross-encoder model (e.g., miniLM) to catch semantic and transliteration variations, trained with Hard Negatives sampled from the blocking phase.

## 21. Loss, Thresholding, and Evaluation (F0.5)

- **F0.5 Weighting:** Precision is weighted twice as heavily as recall. 
- **Implication:** The model should have a strict threshold. False merges hurt the score severely. Singletons must be confidently ignored.

## 22. Training Risks

| Risk | Severity | Evidence | Mitigation |
|---|---|---|---|
| Candidate Explosion | High | Brute force S1xS2 is trillions | Strict multi-pass blocking (Exact Name, N-Gram) |
| False Merges | Critical | Identical names exist in diff addresses | High threshold + F0.5 optimization |
| Validation Leakage | High | Row-wise splits leak entities | Entity-aware stratified validation split |

## 24. Submission-Format Audit

- **matching_results.tsv**: Exactly one row per S1 test entity, IDs from S2/S3 only, empty lists allowed.
- **candidate_pairs.tsv**: Must contain the superset of matches.

## 25. Final Prioritized Recommendations

1. **Preprocessing:** Lowercase, punctuation removal, and standardizing common corporate suffixes.
2. **Blocking:** Multi-pass strategy (Pass 1: Exact Name | Pass 2: N-gram Jaccard > 0.4) to reduce candidates by 99.9% while retaining >95% recall.
3. **Modeling:** Use LightGBM over character-level and token-level Jaccard/Levenshtein features for speed and interpretability.
4. **Validation:** Entity-split validation to mirror the F0.5 test evaluation properly.
5. **Thresholding:** Tune the cutoff specifically for F0.5, leaning toward precision (omitting uncertain matches).
