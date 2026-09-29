import os
import pandas as pd
import json
import numpy as np
from collections import Counter
import re

DATA_DIR = 'dataset'
REPORT_FILE = 'EDA_Report.md'

def append_to_report(text):
    with open(REPORT_FILE, 'a', encoding='utf-8') as f:
        f.write(text)

def analyze_countries():
    print("Analyzing countries...")
    report = ["\n## 16. Country Distribution and Domain Bias\n\n"]
    
    country_counts = {'train': Counter(), 'test': Counter()}
    source_country_counts = {}
    
    for split in ['train', 'test']:
        for source in [1, 2, 3]:
            file = f"{split}_source{source}.tsv"
            path = os.path.join(DATA_DIR, split, file)
            print(f"Reading {file}...")
            source_country_counts[file] = Counter()
            try:
                for chunk in pd.read_csv(path, sep='\t', usecols=['country'], chunksize=500000):
                    counts = chunk['country'].fillna('MISSING').value_counts().to_dict()
                    country_counts[split].update(counts)
                    source_country_counts[file].update(counts)
            except Exception as e:
                print(f"Error {e}")
                
    report.append("### Training Countries\n")
    df_train_c = pd.DataFrame(country_counts['train'].most_common(), columns=['Country', 'Count'])
    df_train_c['Percentage'] = (df_train_c['Count'] / df_train_c['Count'].sum() * 100).round(2).astype(str) + '%'
    report.append(df_train_c.to_markdown(index=False) + "\n\n")
    
    report.append("### Testing Countries\n")
    df_test_c = pd.DataFrame(country_counts['test'].most_common(), columns=['Country', 'Count'])
    df_test_c['Percentage'] = (df_test_c['Count'] / df_test_c['Count'].sum() * 100).round(2).astype(str) + '%'
    report.append(df_test_c.to_markdown(index=False) + "\n\n")
    
    train_c = set(country_counts['train'].keys())
    test_c = set(country_counts['test'].keys())
    new_in_test = test_c - train_c
    
    report.append(f"**Previously unseen country values in test set:** {list(new_in_test) if new_in_test else 'None'}\n\n")
    
    append_to_report("".join(report))

def analyze_geometry_and_distributions():
    print("Analyzing geometry and distributions...")
    # To keep memory bounded, we sample 1 million records per source
    report = ["\n## 3. Record Geometry & 6. Name Dist & 7. Address Dist\n\n"]
    
    name_lengths = Counter()
    addr_lengths = Counter()
    name_tokens = Counter()
    addr_tokens = Counter()
    
    total_names = 0
    missing_names = 0
    total_addrs = 0
    missing_addrs = 0
    
    # We will sample from train_source1, 2, and 3
    sources = ['train_source1.tsv', 'train_source2.tsv', 'train_source3.tsv']
    
    # regexes
    num_re = re.compile(r'\d')
    punct_re = re.compile(r'[^\w\s]')
    ws_re = re.compile(r'\s')
    
    stats = {
        'digits': 0, 'punct': 0, 'ws': 0, 'total_chars': 0, 'alpha': 0
    }
    
    # A reservoir or simple sample to find top duplicates
    name_counts = Counter()
    addr_counts = Counter()
    
    for f in sources:
        path = os.path.join(DATA_DIR, 'train', f)
        print(f"Reading {f}...")
        for chunk in pd.read_csv(path, sep='\t', chunksize=100000):
            names = chunk['business_name'].astype(str)
            addrs = chunk['business_address'].astype(str)
            
            missing_names += chunk['business_name'].isna().sum()
            missing_addrs += chunk['business_address'].isna().sum()
            total_names += len(chunk)
            total_addrs += len(chunk)
            
            name_counts.update(names)
            addr_counts.update(addrs)
            
            n_len = names.str.len()
            a_len = addrs.str.len()
            
            name_lengths.update(n_len)
            addr_lengths.update(a_len)
            
            name_tokens.update(names.str.split().str.len())
            addr_tokens.update(addrs.str.split().str.len())
            
            # Sub-sample for character stats to save compute (10% of each chunk)
            sample_names = names.sample(frac=0.1)
            sample_text = " ".join(sample_names)
            
            stats['total_chars'] += len(sample_text)
            stats['digits'] += len(num_re.findall(sample_text))
            stats['punct'] += len(punct_re.findall(sample_text))
            stats['ws'] += len(ws_re.findall(sample_text))
            stats['alpha'] += sum(c.isalpha() for c in sample_text)
            
    # Calculate quantiles from counter
    def get_percentiles(counter, total):
        if total == 0: return 0,0,0
        sorted_keys = sorted(counter.keys())
        p25_idx = total * 0.25
        p50_idx = total * 0.50
        p75_idx = total * 0.75
        cum = 0
        p25 = p50 = p75 = 0
        for k in sorted_keys:
            prev_cum = cum
            cum += counter[k]
            if prev_cum < p25_idx <= cum: p25 = k
            if prev_cum < p50_idx <= cum: p50 = k
            if prev_cum < p75_idx <= cum: p75 = k
        return p25, p50, p75
        
    def avg_from_counter(counter, total):
        return sum(k * v for k, v in counter.items()) / total if total > 0 else 0
        
    report.append("### 3. Record Geometry\n\n")
    report.append(f"- **Missing Names:** {missing_names} ({missing_names/total_names*100:.4f}%)\n")
    report.append(f"- **Missing Addresses:** {missing_addrs} ({missing_addrs/total_addrs*100:.4f}%)\n\n")
    
    n_p25, n_p50, n_p75 = get_percentiles(name_lengths, total_names)
    report.append(f"- **Name Length (chars):** Avg: {avg_from_counter(name_lengths, total_names):.2f}, P25: {n_p25}, Median: {n_p50}, P75: {n_p75}\n")
    
    a_p25, a_p50, a_p75 = get_percentiles(addr_lengths, total_addrs)
    report.append(f"- **Address Length (chars):** Avg: {avg_from_counter(addr_lengths, total_addrs):.2f}, P25: {a_p25}, Median: {a_p50}, P75: {a_p75}\n")
    
    nt_p25, nt_p50, nt_p75 = get_percentiles(name_tokens, total_names)
    report.append(f"- **Name Token Count:** Avg: {avg_from_counter(name_tokens, total_names):.2f}, P25: {nt_p25}, Median: {nt_p50}, P75: {nt_p75}\n")
    
    at_p25, at_p50, at_p75 = get_percentiles(addr_tokens, total_addrs)
    report.append(f"- **Address Token Count:** Avg: {avg_from_counter(addr_tokens, total_addrs):.2f}, P25: {at_p25}, Median: {at_p50}, P75: {at_p75}\n\n")
    
    if stats['total_chars'] > 0:
        report.append("### Character Composition (Names sampled)\n")
        report.append(f"- **Total characters sampled:** {stats['total_chars']}\n")
        report.append(f"- **Alphabetic:** {stats['alpha']/stats['total_chars']*100:.2f}%\n")
        report.append(f"- **Numeric:** {stats['digits']/stats['total_chars']*100:.2f}%\n")
        report.append(f"- **Whitespace:** {stats['ws']/stats['total_chars']*100:.2f}%\n")
        report.append(f"- **Punctuation:** {stats['punct']/stats['total_chars']*100:.2f}%\n\n")
        
    report.append("### 6. Name Distribution & 7. Address Distribution\n\n")
    
    unique_names = len(name_counts)
    report.append(f"- **Unique Names:** {unique_names}\n")
    report.append(f"- **Unique Name Ratio:** {unique_names/total_names*100:.2f}%\n")
    report.append("#### Top 10 Names\n")
    df_names = pd.DataFrame(name_counts.most_common(10), columns=['Name', 'Count'])
    report.append(df_names.to_markdown(index=False) + "\n\n")
    
    unique_addrs = len(addr_counts)
    report.append(f"- **Unique Addresses:** {unique_addrs}\n")
    report.append(f"- **Unique Address Ratio:** {unique_addrs/total_addrs*100:.2f}%\n")
    report.append("#### Top 10 Addresses\n")
    df_addrs = pd.DataFrame(addr_counts.most_common(10), columns=['Address', 'Count'])
    report.append(df_addrs.to_markdown(index=False) + "\n\n")

    append_to_report("".join(report))
    print("Geometry and Distribution completed.")

if __name__ == "__main__":
    analyze_countries()
    analyze_geometry_and_distributions()
