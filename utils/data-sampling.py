import os
import pandas as pd
import glob

def preserve_and_sample_pool(pool_df, required_ids, fraction, random_state=42):
    """Keeps required ground-truth IDs, then samples a fraction of the remaining noise."""
    required_df = pool_df[pool_df['entity_id'].isin(required_ids)]
    remaining_df = pool_df[~pool_df['entity_id'].isin(required_ids)]
    
    sampled_df = remaining_df.sample(frac=fraction, random_state=random_state)
    return pd.concat([required_df, sampled_df]).sample(frac=1, random_state=random_state).reset_index(drop=True)

def process_directory(src_dir, dest_dir, fraction=0.20):
    """Downsamples a single split directory safely."""
    os.makedirs(dest_dir, exist_ok=True)
    
    # Find files dynamically to handle any naming conventions
    s1_file = glob.glob(os.path.join(src_dir, "*source1.tsv"))[0]
    s2_file = glob.glob(os.path.join(src_dir, "*source2.tsv"))[0]
    s3_file = glob.glob(os.path.join(src_dir, "*source3.tsv"))[0]
    gt_files = glob.glob(os.path.join(src_dir, "*ground_truth.tsv"))
    
    print(f"Processing: {src_dir}")
    s1_df = pd.read_csv(s1_file, sep="\t", dtype=str).fillna("")
    s2_df = pd.read_csv(s2_file, sep="\t", dtype=str).fillna("")
    s3_df = pd.read_csv(s3_file, sep="\t", dtype=str).fillna("")
    
    # 1. Sample 20% of Source 1 Entities
    s1_ids = pd.Series(s1_df['entity_id'].unique())
    sampled_s1_ids = s1_ids.sample(frac=fraction, random_state=42).values
    
    s1_out = s1_df[s1_df['entity_id'].isin(sampled_s1_ids)]
    
    required_pool_ids = set()
    
    # 2. Filter Ground Truth and extract guaranteed matches
    if gt_files:
        gt_file = gt_files[0]
        gt_df = pd.read_csv(gt_file, sep="\t", dtype=str).fillna("")
        gt_out = gt_df[gt_df['source1_entity_id'].isin(sampled_s1_ids)]
        
        # Extract all S2/S3 IDs that are true matches for the sampled S1 entities
        for matches in gt_out['matched_entity_ids']:
            if matches:
                required_pool_ids.update(matches.split(','))
                
        gt_out.to_csv(os.path.join(dest_dir, os.path.basename(gt_file)), sep="\t", index=False)
    
    # 3. Downsample S2 and S3 safely (guaranteed matches + 20% random noise)
    s2_out = preserve_and_sample_pool(s2_df, required_pool_ids, fraction)
    s3_out = preserve_and_sample_pool(s3_df, required_pool_ids, fraction)
    
    # 4. Save
    s1_out.to_csv(os.path.join(dest_dir, os.path.basename(s1_file)), sep="\t", index=False)
    s2_out.to_csv(os.path.join(dest_dir, os.path.basename(s2_file)), sep="\t", index=False)
    s3_out.to_csv(os.path.join(dest_dir, os.path.basename(s3_file)), sep="\t", index=False)
    
    print(f"  -> S1 Reduced: {len(s1_df)} -> {len(s1_out)}")
    print(f"  -> S2 Reduced: {len(s2_df)} -> {len(s2_out)}")
    print(f"  -> S3 Reduced: {len(s3_df)} -> {len(s3_out)}\n")

def run_20_percent_downsample(src_base="dataset-split", dest_base="dataset-split-20-percent"):
    print(f"Starting 20% structural downsample from {src_base} to {dest_base}...\n")
    
    # Walk through the original dataset-split directory
    for root, dirs, files in os.walk(src_base):
        # Only process directories that actually contain .tsv files
        if any(f.endswith('.tsv') for f in files):
            # Recreate the exact subfolder path
            rel_path = os.path.relpath(root, src_base)
            dest_dir = os.path.join(dest_base, rel_path)
            
            process_directory(root, dest_dir, fraction=0.20)
            
    print("=== 20% Downsampling Complete! ===")

if __name__ == "__main__":
    run_20_percent_downsample()