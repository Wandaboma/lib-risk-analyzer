import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm 

# Directory path to scan files
directory = "C:/Users/Admin/Downloads/db-dump/data/results"  

# Prepare lists to store data
data_entries = []

# Attempt to read and extract relevant information
try:
    files = [f for f in os.listdir(directory) if f.endswith(".json")]
    for filename in tqdm(files, desc="Processing JSON files"):
        filepath = os.path.join(directory, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            package_name = data.get("package_name")
            summary_embedding = data.get("summary_embedding", [])
            keywords_embedding = data.get("keywords_embedding", [])
            if package_name and summary_embedding and keywords_embedding:
                # combined_embedding = summary_embedding + keywords_embedding
                data_entries.append({
                    "package_name": package_name,
                    "summary_embedding": summary_embedding,
                    "keywords_embedding": keywords_embedding
                })
except FileNotFoundError:
    raise FileNotFoundError("Directory not found. Please upload the data folder.")

# Convert to DataFrame
df_embeddings = pd.DataFrame(data_entries)

# Save to a single JSON file
output_path = "C:/Users/Admin/Downloads/db-dump/data/package_embeddings.json"
df_embeddings.to_json(output_path, orient="records", lines=True, force_ascii=False)

output_path
