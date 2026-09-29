import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_base_data(train_dir=r"student_resource\dataset\train"):
    """Loads the base training files."""
    print(f"Loading datasets from {train_dir}...")
    s1_df = pd.read_csv(os.path.join(train_dir, "train_source1.tsv"), sep="\t", dtype=str).fillna("")
    s2_df = pd.read_csv(os.path.join(train_dir, "train_source2.tsv"), sep="\t", dtype=str).fillna("")
    s3_df = pd.read_csv(os.path.join(train_dir, "train_source3.tsv"), sep="\t", dtype=str).fillna("")
    gt_df = pd.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), sep="\t", dtype=str).fillna("")
    return s1_df, s2_df, s3_df, gt_df

def save_split(s1_ids, s1_df, s2_df, s3_df, gt_df, out_dir):
    """Helper to filter by Source 1 IDs and save to a specific directory."""
    os.makedirs(out_dir, exist_ok=True)
    
    # Filter S1 using 'entity_id' and Ground Truth using 'source1_entity_id'
    split_s1 = s1_df[s1_df['entity_id'].isin(s1_ids)]
    split_gt = gt_df[gt_df['source1_entity_id'].isin(s1_ids)]
    
    # Save files to output directory
    split_s1.to_csv(os.path.join(out_dir, "train_source1.tsv"), sep="\t", index=False)
    split_gt.to_csv(os.path.join(out_dir, "train_ground_truth.tsv"), sep="\t", index=False)
    s2_df.to_csv(os.path.join(out_dir, "train_source2.tsv"), sep="\t", index=False)
    s3_df.to_csv(os.path.join(out_dir, "train_source3.tsv"), sep="\t", index=False)
    return len(split_s1)

def create_split_a_70_20_10(s1_df, s2_df, s3_df, gt_df):
    """Creates a random 70/20/10 split across all data."""
    print("\n--- Generating Split A (70% Train, 20% Val, 10% Test) ---")
    unique_s1_ids = s1_df['entity_id'].unique()
    
    # First split off 10% for the test set
    train_val_ids, test_ids = train_test_split(unique_s1_ids, test_size=0.10, random_state=42)
    
    # From the remaining 90%, split into 70% train and 20% val
    # (0.20 / 0.90) = 0.2222 is the fraction needed for Val
    train_ids, val_ids = train_test_split(train_val_ids, test_size=(0.20 / 0.90), random_state=42)
    
    base_out = "dataset-split/split_a"
    n_train = save_split(train_ids, s1_df, s2_df, s3_df, gt_df, f"{base_out}/train")
    n_val = save_split(val_ids, s1_df, s2_df, s3_df, gt_df, f"{base_out}/val")
    n_test = save_split(test_ids, s1_df, s2_df, s3_df, gt_df, f"{base_out}/test")
    
    print(f"Split A Complete -> Train: {n_train} | Val: {n_val} | Test: {n_test}")

def create_split_b_domain_shift(s1_df, s2_df, s3_df, gt_df):
    """Creates a country-holdout split: Train/Val on US, Test on India."""
    print("\n--- Generating Split B (Train/Val on US, Test on India) ---")
    
    # Filter S1 entities by country
    us_s1_df = s1_df[s1_df['country'].str.upper() == 'US']
    india_s1_df = s1_df[s1_df['country'].str.upper() == 'INDIA']
    
    us_ids = us_s1_df['entity_id'].unique()
    test_india_ids = india_s1_df['entity_id'].unique()
    
    # Split US data into Train (80%) and Val (20%)
    train_us_ids, val_us_ids = train_test_split(us_ids, test_size=0.20, random_state=42)
    
    base_out = "dataset-split/split_b"
    n_train = save_split(train_us_ids, s1_df, s2_df, s3_df, gt_df, f"{base_out}/train_us")
    n_val = save_split(val_us_ids, s1_df, s2_df, s3_df, gt_df, f"{base_out}/val_us")
    n_test = save_split(test_india_ids, s1_df, s2_df, s3_df, gt_df, f"{base_out}/test_india")
    
    print(f"Split B Complete -> Train (US): {n_train} | Val (US): {n_val} | Test (India): {n_test}")

if __name__ == "__main__":
    s1, s2, s3, gt = load_base_data()
    create_split_a_70_20_10(s1, s2, s3, gt)
    create_split_b_domain_shift(s1, s2, s3, gt)
    print("\nAll splits generated successfully in 'dataset-split/'")