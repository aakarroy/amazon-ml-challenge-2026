import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from tqdm.auto import tqdm

# Updated callback to handle multiple evaluation sets (train and val)
def tqdm_callback(pbar):
    def callback(env):
        pbar.update(1)
        if env.evaluation_result_list:
            # Look for the validation logloss to display on the progress bar
            for data_name, metric_name, value, _ in env.evaluation_result_list:
                if data_name == 'val' and metric_name == 'binary_logloss':
                    pbar.set_postfix({'val_logloss': f"{value:.4f}"})
    return callback

def train_and_predict(
    train_features_path: str,
    val_features_path: str,
    val_s1_path: str,
    output_results_base_path: str,
    thresholds: list = [0.75, 0.80, 0.85, 0.90, 0.95]
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
    
    with tqdm(total=n_trees, desc="Building Trees") as pbar:
        model.fit(
            X_train, y_train,
            # Pass both train and val sets to log metrics for both
            eval_set=[(X_train, y_train), (X_val, y_val)], 
            eval_names=['train', 'val'], 
            eval_metric='binary_logloss',
            callbacks=[tqdm_callback(pbar)]
        )
        
    print("\n   Training complete. Feature Importances:")
    for name, imp in zip(feature_cols, model.feature_importances_):
        print(f"   - {name}: {imp}")

    # ==========================================
    # LOGGING TRAINING HISTORY TO CSV
    # ==========================================
    print("\n3. Exporting training logs...")
    evals_result = model.evals_result_
    log_df = pd.DataFrame({
        'iteration': range(1, n_trees + 1),
        'train_logloss': evals_result['train']['binary_logloss'],
        'val_logloss': evals_result['val']['binary_logloss']
    })
    
    output_dir = os.path.dirname(output_results_base_path)
    os.makedirs(output_dir, exist_ok=True)
    log_csv_path = os.path.join(output_dir, "training_logs.csv")
    log_df.to_csv(log_csv_path, index=False)
    print(f"   Saved training logs to: {log_csv_path}")

    # ==========================================
    # MULTIPLE THRESHOLD PREDICTION LOOP
    # ==========================================
    print("\n4. Predicting Probabilities...")
    # We only need to predict probabilities once
    val_probs = model.predict_proba(X_val)[:, 1]
    original_val_s1 = pd.read_csv(val_s1_path, sep="\t", dtype=str)
    
    for threshold in thresholds:
        print(f"\n--- Processing Threshold: {threshold} ---")
        
        val_df['match_probability'] = val_probs
        val_df['is_predicted_match'] = (val_df['match_probability'] > threshold).astype(int)
        
        matched_pairs = val_df[val_df['is_predicted_match'] == 1]
        
        grouped_matches = matched_pairs.groupby('source1_entity_id')['candidate_entity_id'].apply(
            lambda x: ",".join(sorted(list(set(x)))) 
        ).reset_index(name='matched_entity_ids')

        final_submission = pd.DataFrame({
            'source1_entity_id': original_val_s1['entity_id'] 
        })
        
        final_submission = final_submission.merge(
            grouped_matches, 
            on='source1_entity_id', 
            how='left'
        )
        final_submission['matched_entity_ids'] = final_submission['matched_entity_ids'].fillna("")
        
        # Insert the threshold value into the output filename (e.g., matching_results_0.85.tsv)
        file_name, file_ext = os.path.splitext(output_results_base_path)
        current_output_path = f"{file_name}_{threshold}{file_ext}"
        
        final_submission.to_csv(current_output_path, sep="\t", index=False)
        
        num_total = len(final_submission)
        num_singletons = len(final_submission[final_submission['matched_entity_ids'] == ""])
        print(f"Saved: {current_output_path}")
        print(f"Singletons: {num_singletons:,} ({(num_singletons/num_total)*100:.1f}%) | Matches: {num_total - num_singletons:,}")

# Example Execution
if __name__ == "__main__":
    train_and_predict(
        train_features_path="/kaggle/input/notebooks/aakarroy17/approach-1-feature-engineering-train/train_features.parquet",
        val_features_path="/kaggle/input/notebooks/aakarroy17/approach-1-feature-engineering-val/val_features.parquet",
        val_s1_path="/kaggle/input/datasets/aakarroy17/amazon-2026-ml-preprocessed-20/dataset-split-20-percent/split_a/val/val_source1.tsv",
        output_results_base_path="/kaggle/working/matching_results.tsv",
        thresholds=[0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    )