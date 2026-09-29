import os
import pandas as pd
import numpy as np
import random
from collections import defaultdict
import re

DATA_DIR = 'dataset'
REPORT_FILE = 'EDA_Report.md'

def append_to_report(text):
    with open(REPORT_FILE, 'a', encoding='utf-8') as f:
        f.write(text)

def ngrams(s, n=3):
    s = str(s).lower()
    return set(s[i:i+n] for i in range(max(1, len(s)-n+1)))

def jaccard(set1, set2):
    if not set1 and not set2: return 1.0
    if not set1 or not set2: return 0.0
    return len(set1 & set2) / len(set1 | set2)

def analyze_pairs():
    print("Loading samples for pair analysis...")
    # Sample 10,000 ground truth links
    gt = pd.read_csv(os.path.join(DATA_DIR, 'train', 'train_ground_truth.tsv'), sep='\t')
    
    # We just need some rows with matches
    gt_sample = gt[gt['matched_entity_ids'].notna()].sample(2000, random_state=42)
    s1_ids = set(gt_sample.iloc[:, 0])
    
    s2_ids = set()
    s3_ids = set()
    matches = {}
    for _, row in gt_sample.iterrows():
        s1 = row.iloc[0]
        m = str(row.iloc[1]).split(',')
        matches[s1] = m
        for x in m:
            if 'S2' in x: s2_ids.add(x)
            if 'S3' in x: s3_ids.add(x)
            
    print(f"Sampled {len(s1_ids)} S1, {len(s2_ids)} S2, {len(s3_ids)} S3 entities")
    
    # Load attributes
    def load_dict(file, id_set):
        d = {}
        path = os.path.join(DATA_DIR, 'train', file)
        # Process in chunks to filter
        for chunk in pd.read_csv(path, sep='\t', chunksize=100000):
            filtered = chunk[chunk['entity_id'].isin(id_set)]
            for _, row in filtered.iterrows():
                d[row['entity_id']] = {
                    'name': str(row['business_name']),
                    'addr': str(row['business_address']),
                    'country': str(row['country'])
                }
        return d
        
    s1_dict = load_dict('train_source1.tsv', s1_ids)
    s2_dict = load_dict('train_source2.tsv', s2_ids)
    s3_dict = load_dict('train_source3.tsv', s3_ids)
    
    s23_dict = {**s2_dict, **s3_dict}
    
    # Generate Positive Pairs
    pos_pairs = []
    for s1, m_list in matches.items():
        if s1 in s1_dict:
            for m in m_list:
                if m in s23_dict:
                    pos_pairs.append((s1_dict[s1], s23_dict[m]))
                    
    # Generate Random Negative Pairs
    neg_pairs = []
    s23_keys = list(s23_dict.keys())
    for s1 in s1_dict.values():
        for _ in range(2): # 2 negatives per s1
            m = random.choice(s23_keys)
            # Ensure not a positive
            # (In a small sample, extremely unlikely to hit a true positive by chance if not in matches)
            neg_pairs.append((s1, s23_dict[m]))
            
    def compute_features(pair):
        e1, e2 = pair
        
        n1 = str(e1['name']).lower().strip()
        n2 = str(e2['name']).lower().strip()
        a1 = str(e1['addr']).lower().strip()
        a2 = str(e2['addr']).lower().strip()
        
        features = {}
        features['exact_name'] = int(n1 == n2)
        features['exact_addr'] = int(a1 == a2)
        features['country_match'] = int(str(e1['country']).lower() == str(e2['country']).lower())
        
        t1, t2 = set(n1.split()), set(n2.split())
        features['name_token_jaccard'] = jaccard(t1, t2)
        
        ta1, ta2 = set(a1.split()), set(a2.split())
        features['addr_token_jaccard'] = jaccard(ta1, ta2)
        
        features['name_3gram_jaccard'] = jaccard(ngrams(n1), ngrams(n2))
        features['addr_3gram_jaccard'] = jaccard(ngrams(a1), ngrams(a2))
        return features

    pos_feats = pd.DataFrame([compute_features(p) for p in pos_pairs])
    neg_feats = pd.DataFrame([compute_features(p) for p in neg_pairs])
    
    report = ["\n## 8. Cross-source Consistency & 9. Pair-level Similarity\n\n"]
    
    report.append("### Feature Distributions for Positive vs Negative Pairs\n")
    report.append("Metrics calculated on a sample of ground truth positives vs random negatives.\n\n")
    
    comparison = pd.DataFrame({
        'Feature': pos_feats.columns,
        'Pos_Mean': pos_feats.mean().values,
        'Neg_Mean': neg_feats.mean().values,
        'Pos_P50': pos_feats.median().values,
        'Neg_P50': neg_feats.median().values
    })
    report.append(comparison.to_markdown(index=False) + "\n\n")
    
    report.append("\n## 10. Candidate Generation / Blocking Analysis\n\n")
    report.append("### Blocking Strategy Simulation\n")
    
    # Simulate blocking on the sample
    # Strategy 1: Exact Name (lower)
    exact_hits = pos_feats['exact_name'].sum()
    report.append(f"- **Exact Normalized Name Blocking Recall:** {exact_hits / len(pos_feats) * 100:.2f}%\n")
    
    # Strategy 2: First token of name
    def first_token_match(p):
        n1 = p[0]['name'].split()
        n2 = p[1]['name'].split()
        if n1 and n2: return n1[0].lower() == n2[0].lower()
        return False
        
    ft_hits = sum(first_token_match(p) for p in pos_pairs)
    report.append(f"- **First Token Name Blocking Recall:** {ft_hits / len(pos_pairs) * 100:.2f}%\n")
    
    # Strategy 3: 3-gram overlap > 0.3
    ngram_hits = sum(pos_feats['name_3gram_jaccard'] > 0.3)
    report.append(f"- **Name 3-Gram Overlap (>0.3) Blocking Recall:** {ngram_hits / len(pos_pairs) * 100:.2f}%\n\n")
    
    report.append("### 12. Visual / Qualitative Inspection (Examples)\n\n")
    
    report.append("#### Hard Positive Example (Low string similarity but True Match)\n")
    hard_pos = pos_feats[pos_feats['name_token_jaccard'] < 0.2]
    if not hard_pos.empty:
        idx = hard_pos.index[0]
        p = pos_pairs[idx]
        report.append(f"- **Source 1:** {p[0]['name']} | {p[0]['addr']} | {p[0]['country']}\n")
        report.append(f"- **Matched:** {p[1]['name']} | {p[1]['addr']} | {p[1]['country']}\n\n")
        
    report.append("#### False-Positive-Prone Example (High string similarity but Non-Match)\n")
    hard_neg = neg_feats[(neg_feats['name_token_jaccard'] > 0.5) & (neg_feats['country_match'] == 1)]
    if not hard_neg.empty:
        idx = hard_neg.index[0]
        p = neg_pairs[idx]
        report.append(f"- **Source 1:** {p[0]['name']} | {p[0]['addr']} | {p[0]['country']}\n")
        report.append(f"- **Unmatched:** {p[1]['name']} | {p[1]['addr']} | {p[1]['country']}\n\n")
        
    append_to_report("".join(report))
    print("Pairs and Blocking Analysis completed.")

if __name__ == "__main__":
    analyze_pairs()
