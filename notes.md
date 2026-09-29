Features : entity_id	business_name	business_address	country
Data-Separation:
    1. Split A: train:val:test = 70:20:10 (randomly selected data points of US and INDIA) this verison of data is used most of the time for training.
    2. Split B: train:test = 80:20 (train on US data and test on INDIA data) this version of data is used for testing on France.
Data-Preprocessing:
    - Lowercases text
    - Removes punctuation
    - Standardizes common corporate and street abbreviations
    - Strips excess whitespace 
make all changes in place to curated a preprocessed data set Split A and Split B.

Tips: 
- keep the data in parquet format for faster read/write operations. Use the same preprocessing steps for both splits to ensure consistency.
- 

Approach: 
    - Blocking Statergy: Pass 1: Exact Name Match | Pass 2: N-gram Jaccard > 0.4:
    - Feature Engineering (The Input)
        For every pair generated in candidate_pairs.tsv, you will calculate numeric similarity scores. The EDA report explicitly highlights strong features: 3-Gram Jaccard on the name and Token Jaccard on the address. Your feature table for the model will look like this: name_jaro_winkler_score, name_3gram_jaccard, address_token_jaccard, exact_country_match (Binary 1 or 0)
    - Model Architecture: LightGBM, Binary Logloss, 