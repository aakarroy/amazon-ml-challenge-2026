import os
import gc
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

def run_blocking_to_disk(s1_path, s2_path, s3_path, output_path):
    print("1. Loading preprocessed datasets...")
    s1_df = pd.read_csv(s1_path, sep="\t", dtype=str).fillna("")
    s2_df = pd.read_csv(s2_path, sep="\t", dtype=str).fillna("")
    s3_df = pd.read_csv(s3_path, sep="\t", dtype=str).fillna("")
    
    all_s1_ids = s1_df['entity_id'].unique()
    countries = s1_df['country'].unique()
    
    pool_df = pd.concat([s2_df, s3_df], ignore_index=True)
    del s2_df, s3_df
    gc.collect()
    
    # ---------------------------------------------------------
    # DISK SPILLING SETUP
    # ---------------------------------------------------------
    temp_csv_path = "temp_matches.csv"
    if os.path.exists(temp_csv_path):
        os.remove(temp_csv_path) # Clean up old runs
        
    def write_to_disk(df_chunk):
        """Helper function to append a chunk to the hard drive."""
        # Write header only if the file doesn't exist yet
        write_header = not os.path.exists(temp_csv_path)
        df_chunk.to_csv(temp_csv_path, mode='a', index=False, header=write_header)

    for country in countries:
        if not country or country == "UNKNOWN":
            continue
            
        print(f"\n--- Processing Country Partition: {country} ---")
        s1_sub = s1_df[s1_df['country'] == country].reset_index(drop=True)
        pool_sub = pool_df[pool_df['country'] == country].reset_index(drop=True)
        
        if s1_sub.empty or pool_sub.empty:
            continue

        s1_ids_arr = s1_sub['entity_id'].values
        pool_ids_arr = pool_sub['entity_id'].values

        # ==========================================
        # PASS 1: Exact Name Match
        # ==========================================
        print("  -> Pass 1: Exact Name Match...")
        exact_merged = pd.merge(
            s1_sub[['entity_id', 'business_name']], 
            pool_sub[['entity_id', 'business_name']], 
            on='business_name', 
            suffixes=('_s1', '_pool')
        )
        
        if not exact_merged.empty:
            match_chunk = pd.DataFrame({
                's1_id': exact_merged['entity_id_s1'].values,
                'pool_id': exact_merged['entity_id_pool'].values
            })
            write_to_disk(match_chunk) # SAVE TO HDD

        del exact_merged, match_chunk
        gc.collect()

        # ==========================================
        # PASS 2: N-Gram (TF-IDF) Similarity > 0.4
        # ==========================================
        print("  -> Pass 2: Fitting TF-IDF Vectorizer...")
        vectorizer = TfidfVectorizer(
            analyzer='char_wb', 
            ngram_range=(3, 3), 
            min_df=2, 
            max_df=0.03,  
            max_features=150000, 
            dtype=np.float32 
        )
        
        pool_matrix = vectorizer.fit_transform(pool_sub['business_name'])
        s1_matrix = vectorizer.transform(s1_sub['business_name'])
        
        chunk_size = 2000  
        num_chunks = int(np.ceil(s1_matrix.shape[0] / chunk_size))
        
        print("  -> Pass 2: Computing Sparse Dot Products...")
        for start_idx in tqdm(range(0, s1_matrix.shape[0], chunk_size), total=num_chunks, desc=f"Scoring {country}"):
            end_idx = min(start_idx + chunk_size, s1_matrix.shape[0])
            s1_chunk = s1_matrix[start_idx:end_idx]
            
            sparse_sim_matrix = s1_chunk.dot(pool_matrix.T)
            
            sparse_sim_matrix.data[sparse_sim_matrix.data <= 0.65] = 0
            sparse_sim_matrix.eliminate_zeros()
            sparse_sim_matrix = sparse_sim_matrix.tocoo()
            
            if sparse_sim_matrix.nnz > 0:
                s1_indices = sparse_sim_matrix.row
                pool_indices = sparse_sim_matrix.col
                
                chunk_df = pd.DataFrame({
                    's1_id': s1_ids_arr[start_idx + s1_indices],
                    'pool_id': pool_ids_arr[pool_indices]
                })
                write_to_disk(chunk_df) # SAVE TO HDD
                del chunk_df
                
            del sparse_sim_matrix, s1_chunk
            gc.collect()
            
        # Free heavy matrices before moving to the next country
        del pool_matrix, s1_matrix, vectorizer, s1_sub, pool_sub
        gc.collect()

    # ==========================================
    # LOAD FROM DISK AND FORMAT
    # ==========================================
    print("\n2. Loading all matches from disk for final formatting...")
    # Read the tiny 2-column CSV from the hard drive (takes very little RAM)
    if os.path.exists(temp_csv_path):
        final_matches_df = pd.read_csv(temp_csv_path)
        final_matches_df.drop_duplicates(inplace=True)
        final_matches_df.sort_values(['s1_id', 'pool_id'], inplace=True)
        
        print("3. Grouping candidates...")
        grouped = final_matches_df.groupby('s1_id')['pool_id'].apply(lambda x: ','.join(x)).reset_index()
        grouped.columns = ['source1_entity_id', 'candidate_entity_ids']
        
        # Clean up temp file
        os.remove(temp_csv_path)
    else:
        grouped = pd.DataFrame(columns=['source1_entity_id', 'candidate_entity_ids'])

    print("4. Formatting final candidate_pairs.tsv...")
    master_df = pd.DataFrame({'source1_entity_id': all_s1_ids})
    out_df = master_df.merge(grouped, on='source1_entity_id', how='left')
    out_df['candidate_entity_ids'] = out_df['candidate_entity_ids'].fillna("")
    
    out_df.to_csv(output_path, sep="\t", index=False)
    print(f"Saved {len(out_df)} S1 entities to {output_path}")

# Example execution
run_blocking_to_disk(
    "/kaggle/input/datasets/aakarroy17/amazon-2026-ml-preprocessed-20/dataset-split-20-percent/split_a/test/test_source1.tsv",
    "/kaggle/input/datasets/aakarroy17/amazon-2026-ml-preprocessed-20/dataset-split-20-percent/split_a/test/test_source2.tsv",
    "/kaggle/input/datasets/aakarroy17/amazon-2026-ml-preprocessed-20/dataset-split-20-percent/split_a/test/test_source3.tsv",
    "/kaggle/working/candidate_pairs-test.tsv"
)