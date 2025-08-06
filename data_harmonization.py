import pandas as pd
import os

# Set your data directory path
DATA_DIR = "ForParticipants/csv_data_extracted"

# Input file names (USA and China for 2023 and 2024)
FILES = {
    "CHN": ["trade_s_chn_m_hs_2023.csv", "trade_s_chn_m_hs_2024.csv"],
    "USA": ["trade_s_usa_state_m_hs_2023.csv", "trade_s_usa_state_m_hs_2024.csv"]
}

# Output Parquet file (highly compressed and efficient)
OUTPUT_PARQUET = os.path.join(DATA_DIR, "harmonized_trade_data.parquet")

# Unified list to store all dataframes
harmonized_chunks = []

# Helper function to process each chunk
def harmonize_chunk(chunk, origin_country):
    df = pd.DataFrame()
    df["origin"] = origin_country
    df["destination"] = chunk["country_id"].str.upper()
    df["hs6"] = chunk["product_id"].astype(str).str.zfill(6)
    df["hs4"] = df["hs6"].str[:4]
    df["trade_flow"] = chunk["trade_flow_name"].str.capitalize()
    df["month"] = pd.to_datetime(chunk["month_id"].astype(str), format="%Y%m", errors='coerce')
    df["value"] = pd.to_numeric(chunk["trade_value"], errors="coerce").fillna(0).astype(int)
    return df[["origin", "destination", "hs6", "hs4", "trade_flow", "month", "value"]]

# Iterate through each country and its files
for origin_country, file_list in FILES.items():
    print("Origin country is: ", origin_country)
    for file in file_list:
        full_path = os.path.join(DATA_DIR, file)
        print(f"Processing {full_path} ...")
        
        for chunk in pd.read_csv(full_path, chunksize=500000, dtype=str, low_memory=False, on_bad_lines='skip'):
            origin = origin_country  # <- explicitly bind in scope
            chunk_harmonized = harmonize_chunk(chunk.copy(), origin_country=origin)
            harmonized_chunks.append(chunk_harmonized)

# Concatenate all harmonized dataframes
df_all = pd.concat(harmonized_chunks, ignore_index=True)

# Save to Parquet for future fast loading
df_all.to_parquet(OUTPUT_PARQUET, index=False)
print(f"Saved harmonized dataset to: {OUTPUT_PARQUET}")