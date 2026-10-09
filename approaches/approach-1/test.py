import pandas as pd
import numpy as np
import os

def compute_s1_f05(gt_set: set, pred_set: set) -> float:
    """
    Computes F_0.5 score for a single Source 1 entity.
    Formula: F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    """
    # Singleton logic: No true matches exist
    if len(gt_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0
    
    # Matches exist, but model predicted empty list
    if len(pred_set) == 0:
        return 0.0
    
    # Calculate True Positives
    true_positives = len(gt_set.intersection(pred_set))
    
    precision = true_positives / len(pred_set)
    recall = true_positives / len(gt_set)
    
    if (0.25 * precision + recall) == 0:
        return 0.0
        
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
    return f05

def evaluate_predictions(gt_filepath: str, pred_filepath: str):
    """
    Reads ground truth and prediction TSVs, calculates Macro F_0.5.
    """
    print(f"Evaluating {pred_filepath} against {gt_filepath}...")
    
    gt_df = pd.read_csv(gt_filepath, sep="\t", dtype=str).fillna("")
    pred_df = pd.read_csv(pred_filepath, sep="\t", dtype=str).fillna("")
    
    # Build Ground Truth Dictionary {s1_id: set(matches)}
    gt_map = {}
    for _, row in gt_df.iterrows():
        # Ground truth file header is 'source1_entity_id'
        s1_id = row['source1_entity_id']
        matches = set(row['matched_entity_ids'].split(',')) if row['matched_entity_ids'] else set()
        gt_map[s1_id] = matches
        
    # Build Prediction Dictionary {s1_id: set(matches)}
    pred_map = {}
    for _, row in pred_df.iterrows():
        s1_id = row['source1_entity_id']
        matches = set(row['matched_entity_ids'].split(',')) if row['matched_entity_ids'] else set()
        pred_map[s1_id] = matches

    scores = []
    
    # Evaluate per Source 1 entity in the ground truth
    for s1_id, gt_set in gt_map.items():
        pred_set = pred_map.get(s1_id, set())
        f05 = compute_s1_f05(gt_set, pred_set)
        scores.append(f05)
        
    macro_f05 = np.mean(scores)
    
    print(f"\n==============================")
    print(f"  MACRO F_0.5 SCORE: {macro_f05:.5f}")
    print(f"==============================")
    print(f"Total entities evaluated: {len(scores):,}")
    
    return macro_f05

# Example Execution
if __name__ == "__main__":
    evaluate_predictions(
        gt_filepath=r"dataset-split-20-percent\split_a\test\test_ground_truth.tsv",
        pred_filepath=r"approaches\approach-1\results-on-test\test_matching_results_0.95.tsv"
    )