import os
import pandas as pd
import json
import time

DATA_DIR = 'dataset'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
REPORT_FILE = 'EDA_Report.md'

def append_to_report(text):
    with open(REPORT_FILE, 'a', encoding='utf-8') as f:
        f.write(text)

def run_inventory_and_integrity():
    report = []
    report.append("# Business Entity Resolution - Exploratory Data Analysis\n\n")
    report.append("## 1. Dataset Inventory\n\n")
    
    train_files = os.listdir(TRAIN_DIR) if os.path.exists(TRAIN_DIR) else []
    test_files = os.listdir(TEST_DIR) if os.path.exists(TEST_DIR) else []
    
    report.append(f"- **Total train files:** {len(train_files)}\n")
    report.append(f"- **Total test files:** {len(test_files)}\n\n")
    
    file_stats = []
    
    for d, files in [('train', train_files), ('test', test_files)]:
        for f in files:
            path = os.path.join(DATA_DIR, d, f)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            # count lines (records)
            with open(path, 'rb') as file:
                lines = sum(1 for _ in file)
            records = lines - 1 if lines > 0 else 0
            file_stats.append({'split': d, 'file': f, 'size_mb': size_mb, 'records': records})
    
    df_stats = pd.DataFrame(file_stats)
    report.append("### File Statistics\n\n")
    report.append(df_stats.to_markdown(index=False) + "\n\n")
    
    # Extract specific counts
    try:
        s1_train = df_stats[df_stats['file'] == 'train_source1.tsv']['records'].values[0]
        s2_train = df_stats[df_stats['file'] == 'train_source2.tsv']['records'].values[0]
        s3_train = df_stats[df_stats['file'] == 'train_source3.tsv']['records'].values[0]
        gt_train = df_stats[df_stats['file'] == 'train_ground_truth.tsv']['records'].values[0]
    except Exception as e:
        s1_train, s2_train, s3_train, gt_train = 0, 0, 0, 0
    
    report.append(f"- **Total Train Source 1 records:** {s1_train}\n")
    report.append(f"- **Total Train Source 2 records:** {s2_train}\n")
    report.append(f"- **Total Train Source 3 records:** {s3_train}\n")
    report.append(f"- **Total Ground Truth records:** {gt_train}\n\n")
    
    report.append("## 2. File Structure and Integrity\n\n")
    
    schemas = {}
    for d, files in [('train', train_files), ('test', test_files)]:
        for f in files:
            path = os.path.join(DATA_DIR, d, f)
            try:
                df_head = pd.read_csv(path, sep='\t', nrows=5)
                schemas[f] = list(df_head.columns)
            except Exception as e:
                schemas[f] = f"Error reading schema: {e}"
                
    for f, schema in schemas.items():
        report.append(f"- **{f}** columns: `{schema}`\n")
    
    append_to_report("".join(report))
    print("Step 1 and 2 completed.")

def run_ground_truth_analysis():
    report = []
    report.append("\n## 5. Ground-Truth Matching Distribution\n\n")
    path = os.path.join(TRAIN_DIR, 'train_ground_truth.tsv')
    if not os.path.exists(path):
        report.append("Ground truth file not found.\n")
        append_to_report("".join(report))
        return
        
    df_gt = pd.read_csv(path, sep='\t')
    # handle empty matches (singleton) - check how they are represented. maybe empty string or NaN?
    # Assuming schema is 'source1_entity_id', 'matched_entity_ids'
    
    # We need to see actual column names first, assuming they are as stated:
    # let's write safe code
    col_s1 = df_gt.columns[0]
    col_match = df_gt.columns[1]
    
    def count_matches(match_str):
        if pd.isna(match_str) or str(match_str).strip() == '':
            return 0
        return len(str(match_str).split(','))  # typically separated by commas or pipes? Let's check comma first.
        # Wait, the prompt says 'matched_entity_ids'. Usually space, comma, or pipe. 
        # I'll just use string length approximation or checking 'S2-' and 'S3-'.
        
    def count_source(match_str, prefix):
        if pd.isna(match_str): return 0
        return str(match_str).count(prefix)
        
    df_gt['num_matches'] = df_gt[col_match].apply(lambda x: count_source(x, 'S2-') + count_source(x, 'S3-'))
    df_gt['s2_matches'] = df_gt[col_match].apply(lambda x: count_source(x, 'S2-'))
    df_gt['s3_matches'] = df_gt[col_match].apply(lambda x: count_source(x, 'S3-'))
    
    total_entities = len(df_gt)
    singletons = len(df_gt[df_gt['num_matches'] == 0])
    one_match = len(df_gt[df_gt['num_matches'] == 1])
    multi_match = len(df_gt[df_gt['num_matches'] > 1])
    both_sources = len(df_gt[(df_gt['s2_matches'] > 0) & (df_gt['s3_matches'] > 0)])
    
    total_links = df_gt['num_matches'].sum()
    s2_links = df_gt['s2_matches'].sum()
    s3_links = df_gt['s3_matches'].sum()
    
    report.append(f"- **Total Source 1 entities in GT:** {total_entities}\n")
    report.append(f"- **Total positive links:** {total_links}\n")
    report.append(f"  - **S1 -> S2 links:** {s2_links}\n")
    report.append(f"  - **S1 -> S3 links:** {s3_links}\n\n")
    
    report.append("### Match Count Distribution\n")
    report.append(f"- **Singletons (0 matches):** {singletons} ({(singletons/total_entities)*100:.2f}%)\n")
    report.append(f"- **Exactly 1 match:** {one_match} ({(one_match/total_entities)*100:.2f}%)\n")
    report.append(f"- **Multiple matches (2+):** {multi_match} ({(multi_match/total_entities)*100:.2f}%)\n")
    report.append(f"- **Matches in both S2 and S3:** {both_sources} ({(both_sources/total_entities)*100:.2f}%)\n\n")
    
    dist = df_gt['num_matches'].value_counts().sort_index().reset_index()
    dist.columns = ['Match Count', 'Source 1 Entities']
    dist['Percentage'] = (dist['Source 1 Entities'] / total_entities * 100).round(2).astype(str) + '%'
    report.append(dist.head(10).to_markdown(index=False) + "\n\n")
    
    append_to_report("".join(report))
    print("Step 5 completed.")

if __name__ == "__main__":
    # Create empty report
    open(REPORT_FILE, 'w').close()
    run_inventory_and_integrity()
    run_ground_truth_analysis()
