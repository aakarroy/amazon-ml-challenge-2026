import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def run_blocking(s1_path, s2_path, s3_path, output_path):
    print("Loading preprocessed datasets...")
    s1_df = pd.read_csv(s1_path, sep="\t", dtype=str).fillna("")
    s2_df = pd.read_csv(s2_path, sep="\t", dtype=str).fillna("")
    s3_df = pd.read_csv(s3_path, sep="\t", dtype=str).fillna("")
    
    # Combine S2 and S3 into a single searchable candidate pool
    pool_df = pd.concat([s2_df, s3_df], ignore_index=True)
    
    candidate_dict = {s1_id: set() for s1_id in s1_df['entity_id']}
    countries = s1_df['country'].unique()

    for country in countries:
        if not country or country == "UNKNOWN":
            continue
            
        print(f"\nProcessing Country Partition: {country}")
        s1_sub = s1_df[s1_df['country'] == country].reset_index(drop=True)
        pool_sub = pool_df[pool_df['country'] == country].reset_index(drop=True)
        
        if s1_sub.empty or pool_sub.empty:
            continue

        # ==========================================
        # PASS 1: Exact Name Match
        # ==========================================
        print("  -> Running Pass 1: Exact Name Match...")
        exact_merged = pd.merge(
            s1_sub[['entity_id', 'business_name']], 
            pool_sub[['entity_id', 'business_name']], 
            on='business_name', 
            suffixes=('_s1', '_pool')
        )
        for _, row in exact_merged.iterrows():
            if row['business_name'].strip() != "":
                candidate_dict[row['entity_id_s1']].add(row['entity_id_pool'])

        # ==========================================
        # PASS 2: N-Gram (TF-IDF) Similarity > 0.4
        # ==========================================
        print("  -> Running Pass 2: Sparse 3-Gram Similarity...")
        # Use character 3-grams to catch typos and transliterations
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), min_df=2)
        
        # Fit on pool, transform both
        pool_matrix = vectorizer.fit_transform(pool_sub['business_name'])
        s1_matrix = vectorizer.transform(s1_sub['business_name'])
        
        # Process in chunks to prevent RAM overflow
        chunk_size = 5000 
        for start_idx in range(0, s1_matrix.shape[0], chunk_size):
            end_idx = min(start_idx + chunk_size, s1_matrix.shape[0])
            s1_chunk = s1_matrix[start_idx:end_idx]
            
            # Compute similarity dot product
            sim_matrix = cosine_similarity(s1_chunk, pool_matrix)
            
            # Find indices where similarity > 0.4
            s1_indices, pool_indices = np.where(sim_matrix > 0.4)
            
            for s1_i, pool_i in zip(s1_indices, pool_indices):
                s1_id = s1_sub.iloc[start_idx + s1_i]['entity_id']
                pool_id = pool_sub.iloc[pool_i]['entity_id']
                candidate_dict[s1_id].add(pool_id)

    # ==========================================
    # FORMAT FOR SUBMISSION
    # ==========================================
    print("\nFormatting candidate_pairs.tsv...")
    results = []
    for s1_id, matches in candidate_dict.items():
        # Remove empty strings if any, and sort for consistency
        clean_matches = sorted([m for m in matches if m])
        results.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(clean_matches)
        })
        
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, sep="\t", index=False)
    print(f"Saved {len(out_df)} S1 entities to {output_path}")

# Example execution on Split A Training Set
# run_blocking(
#     "dataset-split/split_a/train/train_source1.tsv",
#     "dataset-split/split_a/train/train_source2.tsv",
#     "dataset-split/split_a/train/train_source3.tsv",
#     "dataset-split/split_a/train/candidate_pairs.tsv"
# )