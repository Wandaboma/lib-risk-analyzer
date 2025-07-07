import pandas as pd
import json
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
dump_dir = os.path.join(base_dir, "..", "data")
crate_file = os.path.join(dump_dir, "crates.csv")
keyword_file = os.path.join(dump_dir, "keywords.csv")
crate_keywords_file = os.path.join(dump_dir, "crates_keywords.csv")

# Load the crates, keywords, and crate-keyword mapping data
crates_df = pd.read_csv(crate_file)
keywords_df = pd.read_csv(keyword_file)
crate_keywords_df = pd.read_csv(crate_keywords_file)

# Merge crate_keywords with keywords to get keyword names for each crate
merged_df = pd.merge(crate_keywords_df, keywords_df, left_on="keyword_id", right_on="id", how="inner")

# Group by crate_id and aggregate keywords into lists
keywords_grouped = merged_df.groupby("crate_id")["keyword"].apply(list).reset_index()

# Merge the aggregated keywords with crate information
crates_with_keywords = pd.merge(crates_df, keywords_grouped, left_on="id", right_on="crate_id", how="left")

# Select and rename the relevant columns
final_df = crates_with_keywords[["id", "name", "description", "readme", "keyword"]].rename(
    columns={"id": "crate_id", "keyword": "keywords"}
)

# Replace missing keywords with empty lists
final_df["keywords"] = final_df["keywords"].apply(lambda x: x if isinstance(x, list) else [])

# Export the result to a JSON file
output_path = os.path.join(dump_dir, "crates_texutal.json")
final_df.to_json(output_path, orient="records", indent=2, force_ascii=False)

output_path
