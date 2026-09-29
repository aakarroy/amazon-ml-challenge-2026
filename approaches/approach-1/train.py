import pandas as pd
import numpy as np
import lightgbm as lgb
import os

def train_and_predict(
    train_features_path: str,
    val_features_path: str,
    val_s1_path: str,
    output_results_path: str,
    threshold: float = 0.85
):
    print("1. Loading feature datasets...")
    train_df = pd.read_parquet(train_features_path)
    val_df = pd.read_parquet(val_features_path)
    
    # Define the exact features used for training
    feature_cols = [
        'name_jaro_winkler_score',
        'name_3gram_jaccard',
        'address_token_jaccard',
        'exact_country_match'
    ]
    target_col = 'is_match'
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    
    X_val = val_df[feature_cols]
    
    print("2. Training LightGBM Baseline Model...")
    # Standard binary classifier using Binary Logloss
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        objective='binary',
        class_weight='balanced' # Helps if true matches are heavily outnumbered by false candidates
    )
    
    model.fit(X_train, y_train)
    print("   Training complete. Feature Importances:")
    for name, imp in zip(feature_cols, model.feature_importances_):
        print(f"   - {name}: {imp}")

    print(f"\n3. Predicting Probabilities & Applying Strict Threshold (tau > {threshold})...")
    # predict_proba returns [P(class=0), P(class=1)]
    val_probs = model.predict_proba(X_val)[:, 1]
    
    # Only declare a match if the model is highly confident (favoring Precision for F0.5)
    val_df['match_probability'] = val_probs
    val_df['is_predicted_match'] = (val_df['match_probability'] > threshold).astype(int)
    
    # Filter down to ONLY the pairs that crossed the threshold
    matched_pairs = val_df[val_df['is_predicted_match'] == 1]
    
    print("\n4. Formatting matching_results.tsv...")
    # Group the surviving candidates by Source 1 Entity ID
    # This turns multiple S2/S3 matches into a single comma-separated string
    grouped_matches = matched_pairs.groupby('source1_entity_id')['candidate_entity_id'].apply(
        lambda x: ",".join(sorted(list(set(x)))) # Ensure no duplicates in the string
    ).reset_index(name='matched_entity_ids')

    # IMPORTANT: The rules state EVERY Source 1 entity from the test/val set must be present.
    # We must load the original Source 1 file to capture the Singletons (entities with 0 matches).
    original_val_s1 = pd.read_csv(val_s1_path, sep="\t", dtype=str)
    
    # Create the master dataframe with all S1 IDs
    final_submission = pd.DataFrame({
        'source1_entity_id': original_val_s1['entity_id'] # Use the actual S1 IDs from the source file
    })
    
    # Merge the predicted matches onto the master list
    final_submission = final_submission.merge(
        grouped_matches, 
        on='source1_entity_id', 
        how='left'
    )
    
    # Fill NaN values with empty strings for Singletons
    final_submission['matched_entity_ids'] = final_submission['matched_entity_ids'].fillna("")
    
    print(f"5. Saving formatted results to {output_results_path}...")
    os.makedirs(os.path.dirname(output_results_path), exist_ok=True)
    final_submission.to_csv(output_results_path, sep="\t", index=False)
    
    # Quick sanity check printouts
    num_total = len(final_submission)
    num_singletons = len(final_submission[final_submission['matched_entity_ids'] == ""])
    print(f"\n--- Validation Summary ---")
    print(f"Total Source 1 Entities Evaluated: {num_total:,}")
    print(f"Predicted Singletons (No Matches): {num_singletons:,} ({(num_singletons/num_total)*100:.1f}%)")
    print(f"Predicted Matched Entities:        {num_total - num_singletons:,}")

# Example Execution for Local Validation Split A
# (Assuming you already ran the feature engineering script for both the Train and Val folders)
if __name__ == "__main__":
    # train_and_predict(
    #     train_features_path="dataset-split/split_a/train/train_features.parquet",
    #     val_features_path="dataset-split/split_a/val/val_features.parquet",
    #     val_s1_path="dataset-split/split_a/val/val_source1.tsv",
    #     output_results_path="dataset-split/split_a/val/matching_results.tsv",
    #     threshold=0.85 
    # )
    pass