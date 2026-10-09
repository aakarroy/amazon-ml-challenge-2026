import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from tqdm.auto import tqdm

# Custom callback to link LightGBM iterations to a tqdm progress bar
def tqdm_callback(pbar):
    def callback(env):
        pbar.update(1)
        # Display the validation logloss metric in the progress bar
        if env.evaluation_result_list:
            val_loss = env.evaluation_result_list[0][2]
            pbar.set_postfix({'val_logloss': f"{val_loss:.4f}"})
    return callback

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
    
    feature_cols = [
        'name_jaro_winkler_score',
        'name_3gram_jaccard',
        'address_token_jaccard',
        'exact_country_match'
    ]
    target_col = 'is_match'
    
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    
    # We must have a target column in the validation set to track progress metrics
    X_val, y_val = val_df[feature_cols], val_df[target_col]
    
    print("2. Training LightGBM Baseline Model...")
    n_trees = 100
    model = lgb.LGBMClassifier(
        n_estimators=n_trees,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        objective='binary',
        class_weight='balanced' 
    )
    
    # Initialize the progress bar
    with tqdm(total=n_trees, desc="Building Trees") as pbar:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)], # Required to trigger the callback
            eval_metric='binary_logloss',
            callbacks=[tqdm_callback(pbar)]
        )
        
    print("\n   Training complete. Feature Importances:")
    for name, imp in zip(feature_cols, model.feature_importances_):
        print(f"   - {name}: {imp}")

    print(f"\n3. Predicting Probabilities & Applying Strict Threshold (tau > {threshold})...")
    val_probs = model.predict_proba(X_val)[:, 1]
    
    val_df['match_probability'] = val_probs
    val_df['is_predicted_match'] = (val_df['match_probability'] > threshold).astype(int)
    matched_pairs = val_df[val_df['is_predicted_match'] == 1]
    
    print("4. Formatting matching_results.tsv...")
    grouped_matches = matched_pairs.groupby('source1_entity_id')['candidate_entity_id'].apply(
        lambda x: ",".join(sorted(list(set(x)))) 
    ).reset_index(name='matched_entity_ids')

    original_val_s1 = pd.read_csv(val_s1_path, sep="\t", dtype=str)
    
    final_submission = pd.DataFrame({
        'source1_entity_id': original_val_s1['entity_id'] 
    })
    
    final_submission = final_submission.merge(
        grouped_matches, 
        on='source1_entity_id', 
        how='left'
    )
    final_submission['matched_entity_ids'] = final_submission['matched_entity_ids'].fillna("")
    
    print(f"5. Saving formatted results to {output_results_path}...")
    os.makedirs(os.path.dirname(output_results_path), exist_ok=True)
    final_submission.to_csv(output_results_path, sep="\t", index=False)
    
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