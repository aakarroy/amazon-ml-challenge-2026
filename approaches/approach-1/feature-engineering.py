import os
import pandas as pd
import numpy as np
from rapidfuzz.distance import JaroWinkler

def jaccard_token_similarity(str1: str, str2: str) -> float:
    """Computes Jaccard similarity based on whitespace-separated words."""
    if not str1 or not str2:
        return 0.0
    tokens1 = set(str1.split())
    tokens2 = set(str2.split())
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    return float(intersection / union) if union > 0 else 0.0

def jaccard_ngram_similarity(str1: str, str2: str, n: int = 3) -> float:
    """Computes Jaccard similarity based on character n-grams."""
    if not str1 or not str2 or len(str1) < n or len(str2) < n:
        return 0.0
    ngrams1 = set([str1[i:i+n] for i in range(len(str1) - n + 1)])
    ngrams2 = set([str2[i:i+n] for i in range(len(str2) - n + 1)])
    intersection = len(ngrams1.intersection(ngrams2))
    union = len(ngrams1.union(ngrams2))
    return float(intersection / union) if union > 0 else 0.0

def extract_features(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    candidate_pairs_path: str,
    output_features_path: str,
    gt_path: str = None
):
    print("1. Loading datasets...")
    s1_df = pd.read_csv(s1_path, sep="\t", dtype=str).fillna("")
    s2_df = pd.read_csv(s2_path, sep="\t", dtype=str).fillna("")
    s3_df = pd.read_csv(s3_path, sep="\t", dtype=str).fillna("")
    cand_df = pd.read_csv(candidate_pairs_path, sep="\t", dtype=str).fillna("")

    # Combine S2 and S3 into a unified lookup map
    pool_df = pd.concat([s2_df, s3_df], ignore_index=True)
    
    print("2. Unnesting candidate pairs...")
    # Convert comma-separated candidate_entity_ids into individual rows
    cand_df['candidate_entity_id'] = cand_df['candidate_entity_ids'].str.split(',')
    pairs_df = cand_df.explode('candidate_entity_id')[['source1_entity_id', 'candidate_entity_id']]
    
    # Filter out empty rows (for singletons with no candidates)
    pairs_df = pairs_df[pairs_df['candidate_entity_id'].str.strip() != ""].reset_index(drop=True)
    print(f"Total candidate pairs to evaluate: {len(pairs_df):,}")

    print("3. Merging entity attributes...")
    # Merge Source 1 attributes
    pairs_df = pairs_df.merge(
        s1_df[['entity_id', 'business_name', 'business_address', 'country']],
        left_on='source1_entity_id',
        right_on='entity_id',
        how='left'
    ).rename(columns={
        'business_name': 's1_name',
        'business_address': 's1_addr',
        'country': 's1_country'
    }).drop(columns=['entity_id'])

    # Merge Candidate Pool (S2/S3) attributes
    pairs_df = pairs_df.merge(
        pool_df[['entity_id', 'business_name', 'business_address', 'country']],
        left_on='candidate_entity_id',
        right_on='entity_id',
        how='left'
    ).rename(columns={
        'business_name': 'cand_name',
        'business_address': 'cand_addr',
        'country': 'cand_country'
    }).drop(columns=['entity_id'])

    print("4. Computing similarity features...")
    # 1. Exact Country Match (Binary 1 or 0)
    pairs_df['exact_country_match'] = (pairs_df['s1_country'] == pairs_df['cand_country']).astype(int)

    # 2. Jaro-Winkler Similarity on Name
    pairs_df['name_jaro_winkler_score'] = [
        JaroWinkler.similarity(n1, n2) 
        for n1, n2 in zip(pairs_df['s1_name'], pairs_df['cand_name'])
    ]

    # 3. 3-Gram Jaccard Similarity on Name
    pairs_df['name_3gram_jaccard'] = [
        jaccard_ngram_similarity(n1, n2, n=3) 
        for n1, n2 in zip(pairs_df['s1_name'], pairs_df['cand_name'])
    ]

    # 4. Token Jaccard Similarity on Address
    pairs_df['address_token_jaccard'] = [
        jaccard_token_similarity(a1, a2) 
        for a1, a2 in zip(pairs_df['s1_addr'], pairs_df['cand_addr'])
    ]

    # 5. Optionally attach ground truth target label (if gt_path is provided)
    if gt_path and os.path.exists(gt_path):
        print("5. Attaching ground-truth match labels (target)...")
        gt_df = pd.read_csv(gt_path, sep="\t", dtype=str).fillna("")
        
        # Build mapping of {s1_id: set(matched_ids)}
        gt_map = {}
        for _, row in gt_df.iterrows():
            s1_id = row['source1_entity_id']
            matches = set(row['matched_entity_ids'].split(',')) if row['matched_entity_ids'] else set()
            gt_map[s1_id] = matches
            
        targets = []
        for s1_id, cand_id in zip(pairs_df['source1_entity_id'], pairs_df['candidate_entity_id']):
            matched_set = gt_map.get(s1_id, set())
            targets.append(1 if cand_id in matched_set else 0)
            
        pairs_df['is_match'] = targets
        print(f"   Matches found in candidate pool: {sum(targets):,} / {len(targets):,}")

    # Keep only ID columns and feature columns
    feature_cols = [
        'source1_entity_id', 
        'candidate_entity_id',
        'name_jaro_winkler_score',
        'name_3gram_jaccard',
        'address_token_jaccard',
        'exact_country_match'
    ]
    if 'is_match' in pairs_df.columns:
        feature_cols.append('is_match')

    final_df = pairs_df[feature_cols]

    print(f"6. Saving feature table to {output_features_path}...")
    os.makedirs(os.path.dirname(output_features_path), exist_ok=True)
    final_df.to_parquet(output_features_path, index=False)
    print("Feature engineering complete!\n")

# Example Usage:
# extract_features(
#     s1_path="dataset-split/split_a/train/train_source1.tsv",
#     s2_path="dataset-split/split_a/train/train_source2.tsv",
#     s3_path="dataset-split/split_a/train/train_source3.tsv",
#     candidate_pairs_path="dataset-split/split_a/train/candidate_pairs.tsv",
#     output_features_path="dataset-split/split_a/train/train_features.parquet",
#     gt_path="dataset-split/split_a/train/train_ground_truth.tsv"
# )