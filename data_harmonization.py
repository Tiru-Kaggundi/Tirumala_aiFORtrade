import pandas as pd
import os

# Set your data directory path
DATA_DIR = "ForParticipants/csv_data_extracted"

FILES = {
    "CHN": ["trade_s_chn_m_hs_2023.csv", "trade_s_chn_m_hs_2024.csv"],
    "USA": ["trade_s_usa_state_m_hs_2023.csv", "trade_s_usa_state_m_hs_2024.csv"]
}

OUTPUT_PARQUET = os.path.join(DATA_DIR, "harmonized_trade_data.parquet")
harmonized_chunks = []

def harmonize_chunk(chunk, origin_country, aggregate=True):
    print(f"Inside harmonize_chunk, origin_country = {origin_country}")

    # Drop invalid rows
    chunk = chunk.dropna(subset=["country_id", "product_id", "month_id", "trade_value", "trade_flow_name"])

    # Aggregate across subnational units (states/provinces)
    if aggregate:
        chunk["trade_value"] = pd.to_numeric(chunk["trade_value"], errors="coerce").fillna(0)
        grouped = chunk.groupby(["month_id", "country_id", "product_id", "trade_flow_name"], as_index=False)["trade_value"].sum()
    else:
        grouped = chunk.copy()

    df = grouped.copy()
    df["origin"] = origin_country
    df["destination"] = df["country_id"].str.upper()
    df["hs6"] = df["product_id"].astype(str).str.zfill(6)
    df["hs4"] = df["hs6"].str[:4]
    df["trade_flow"] = df["trade_flow_name"].str.capitalize()
    df["month"] = pd.to_datetime(df["month_id"].astype(str), format="%Y%m", errors='coerce')
    df["value"] = pd.to_numeric(df["trade_value"], errors="coerce").fillna(0).astype(int)

    return df[["origin", "destination", "hs6", "hs4", "trade_flow", "month", "value"]]

for origin_country, file_list in FILES.items():
    print("Origin country is:", origin_country)
    for file in file_list:
        full_path = os.path.join(DATA_DIR, file)
        print(f"Processing {full_path} ...")
        for chunk in pd.read_csv(full_path, chunksize=500000, dtype=str, low_memory=False, on_bad_lines='skip'):
            chunk_harmonized = harmonize_chunk(chunk.copy(), origin_country=origin_country, aggregate=True)
            harmonized_chunks.append(chunk_harmonized)

# Combine and save
df_all = pd.concat(harmonized_chunks, ignore_index=True)
df_all.to_parquet(OUTPUT_PARQUET, index=False)
print(f"✅ Saved harmonized dataset to: {OUTPUT_PARQUET}")