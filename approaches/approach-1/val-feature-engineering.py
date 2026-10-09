import os
import gc
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz.distance import JaroWinkler
from tqdm.auto import tqdm

def jaccard_token_similarity(str1: str, str2: str) -> float:
    if not str1 or not str2: return 0.0
    tokens1, tokens2 = set(str1.split()), set(str2.split())
    union = len(tokens1.union(tokens2))
    return float(len(tokens1.intersection(tokens2)) / union) if union > 0 else 0.0

def jaccard_ngram_similarity(str1: str, str2: str, n: int = 3) -> float:
    if not str1 or not str2 or len(str1) < n or len(str2) < n: return 0.0
    ngrams1 = set([str1[i:i+n] for i in range(len(str1) - n + 1)])
    ngrams2 = set([str2[i:i+n] for i in range(len(str2) - n + 1)])
    union = len(ngrams1.union(ngrams2))
    return float(len(ngrams1.intersection(ngrams2)) / union) if union > 0 else 0.0

def extract_features_memory_safe(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    candidate_pairs_path: str,
    output_features_path: str,
    gt_path: str = None
):
    print("1. Loading datasets into fast-lookup dictionaries...")
    # Load Source 1 and convert to dictionary for O(1) lookup
    s1_df = pd.read_csv(s1_path, sep="\t", dtype=str).fillna("")
    s1_map = s1_df.set_index('entity_id')[['business_name', 'business_address', 'country']].to_dict('index')
    del s1_df
    gc.collect()

    # Load S2/S3, combine, and convert to dictionary
    s2_df = pd.read_csv(s2_path, sep="\t", dtype=str).fillna("")
    s3_df = pd.read_csv(s3_path, sep="\t", dtype=str).fillna("")
    pool_df = pd.concat([s2_df, s3_df], ignore_index=True)
    del s2_df, s3_df
    gc.collect()
    
    pool_map = pool_df.set_index('entity_id')[['business_name', 'business_address', 'country']].to_dict('index')
    del pool_df
    gc.collect()

    # Load Ground Truth
    gt_map = {}
    if gt_path and os.path.exists(gt_path):
        print("2. Loading Ground Truth targets...")
        gt_df = pd.read_csv(gt_path, sep="\t", dtype=str).fillna("")
        for _, row in gt_df.iterrows():
            s1_id = row['source1_entity_id']
            matches = set(row['matched_entity_ids'].split(',')) if row['matched_entity_ids'] else set()
            gt_map[s1_id] = matches
        del gt_df
        gc.collect()

    print("3. Processing candidate pairs in chunks...")
    chunksize = 20000  # Process 20,000 Source 1 records at a time
    writer = None
    
    # Get total chunks for the progress bar
    total_lines = sum(1 for _ in open(candidate_pairs_path)) - 1
    total_chunks = (total_lines // chunksize) + 1

    # Read candidates dynamically in chunks
    candidate_chunks = pd.read_csv(candidate_pairs_path, sep="\t", dtype=str, chunksize=chunksize)
    
    for chunk in tqdm(candidate_chunks, total=total_chunks, desc="Extracting Features"):
        # Explode the comma-separated strings
        chunk['candidate_entity_id'] = chunk['candidate_entity_ids'].str.split(',')
        pairs_df = chunk.explode('candidate_entity_id')[['source1_entity_id', 'candidate_entity_id']]
        pairs_df = pairs_df[pairs_df['candidate_entity_id'].str.strip() != ""].reset_index(drop=True)
        
        if pairs_df.empty:
            continue

        # Fast Dictionary Lookup (Replaces pd.merge)
        empty_attr = {'business_name': '', 'business_address': '', 'country': ''}
        s1_attrs = [s1_map.get(i, empty_attr) for i in pairs_df['source1_entity_id']]
        cand_attrs = [pool_map.get(i, empty_attr) for i in pairs_df['candidate_entity_id']]

        # Compute Features via list comprehensions (Highly optimized in Python)
        pairs_df['exact_country_match'] = [
            1 if s1['country'] == c['country'] and s1['country'] else 0 
            for s1, c in zip(s1_attrs, cand_attrs)
        ]

        pairs_df['name_jaro_winkler_score'] = [
            JaroWinkler.similarity(s1['business_name'], c['business_name']) 
            for s1, c in zip(s1_attrs, cand_attrs)
        ]

        pairs_df['name_3gram_jaccard'] = [
            jaccard_ngram_similarity(s1['business_name'], c['business_name'], n=3) 
            for s1, c in zip(s1_attrs, cand_attrs)
        ]

        pairs_df['address_token_jaccard'] = [
            jaccard_token_similarity(s1['business_address'], c['business_address']) 
            for s1, c in zip(s1_attrs, cand_attrs)
        ]

        if gt_map:
            pairs_df['is_match'] = [
                1 if cand_id in gt_map.get(s1_id, set()) else 0
                for s1_id, cand_id in zip(pairs_df['source1_entity_id'], pairs_df['candidate_entity_id'])
            ]

        # Disk Spilling: Append chunk directly to Parquet file
        table = pa.Table.from_pandas(pairs_df)
        if writer is None:
            os.makedirs(os.path.dirname(output_features_path), exist_ok=True)
            writer = pq.ParquetWriter(output_features_path, table.schema)
        
        writer.write_table(table)
        
        # Destroy chunk to free RAM
        del pairs_df, s1_attrs, cand_attrs, table, chunk
        gc.collect()

    if writer:
        writer.close()
        
    print(f"\nFeature engineering complete! Saved safely to {output_features_path}")

# Example Usage:
extract_features_memory_safe(
    s1_path="/kaggle/input/datasets/aakarroy17/amazon-2026-ml-preprocessed-20/dataset-split-20-percent/split_a/val/val_source1.tsv",
    s2_path="/kaggle/input/datasets/aakarroy17/amazon-2026-ml-preprocessed-20/dataset-split-20-percent/split_a/val/val_source2.tsv",
    s3_path="/kaggle/input/datasets/aakarroy17/amazon-2026-ml-preprocessed-20/dataset-split-20-percent/split_a/val/val_source3.tsv",
    candidate_pairs_path="/kaggle/input/notebooks/aakarroy17/approach-1-blocking-val/candidate_pairs-val.tsv",
    output_features_path="/kaggle/working/val_features.parquet",
    gt_path="/kaggle/input/datasets/aakarroy17/amazon-2026-ml-preprocessed-20/dataset-split-20-percent/split_a/val/val_ground_truth.tsv"
)