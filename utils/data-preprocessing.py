import os
import pandas as pd
import re
from glob import glob

def clean_and_normalize(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Applies standard normalization to a text column.
    """
    df[text_column] = df[text_column].fillna("")
    df[text_column] = df[text_column].str.lower()
    df[text_column] = df[text_column].apply(lambda x: re.sub(r'[^\w\s]', ' ', x))
    
    suffix_map = {
        r'\bprivate\b': 'pvt',
        r'\blimited\b': 'ltd',
        r'\bcorporation\b': 'corp',
        r'\bincorporated\b': 'inc',
        r'\bcompany\b': 'co'
    }
    for pattern, replacement in suffix_map.items():
        df[text_column] = df[text_column].apply(lambda x: re.sub(pattern, replacement, x))
        
    address_map = {
        r'\bstreet\b': 'st',
        r'\broad\b': 'rd',
        r'\bavenue\b': 'ave',
        r'\bboulevard\b': 'blvd'
    }
    for pattern, replacement in address_map.items():
        df[text_column] = df[text_column].apply(lambda x: re.sub(pattern, replacement, x))
        
    df[text_column] = df[text_column].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    
    return df

def process_file_in_place(file_path: str):
    """Loads, preprocesses, and overwrites a specific TSV dataset."""
    print(f"Reading: {file_path}...")
    df = pd.read_csv(file_path, sep="\t", dtype=str)
    
    print(f"  -> Normalizing text columns...")
    df = clean_and_normalize(df, 'business_name')
    df = clean_and_normalize(df, 'business_address')
    
    df['country'] = df['country'].fillna("UNKNOWN").str.upper().str.strip()
    
    print(f"  -> Saving in-place to {file_path}...\n")
    df.to_csv(file_path, sep="\t", index=False)

def run_batch_preprocessing(base_dir="dataset-split"):
    """Recursively finds all source files in the split directories and processes them."""
    # Find all .tsv files recursively
    search_pattern = os.path.join(base_dir, "**", "*.tsv")
    all_tsv_files = glob(search_pattern, recursive=True)
    
    target_files = []
    
    # Filter out ground_truth files because they don't have name/address columns
    for filepath in all_tsv_files:
        filename = os.path.basename(filepath)
        if "ground_truth" not in filename:
            target_files.append(filepath)
            
    print(f"Found {len(target_files)} source files to process.\n")
    
    for filepath in target_files:
        process_file_in_place(filepath)
        
    print("=== All files successfully preprocessed in-place! ===")

if __name__ == "__main__":
    # Point this to your root split folder shown in your file explorer
    run_batch_preprocessing(base_dir="dataset-split")