import pandas as pd
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Set data path
base_dir = os.path.dirname(os.path.abspath(__file__))
path = base_dir + '/data/'
# Load necessary CSV files
analysis_df_24 = pd.read_csv(path + "rust_22_24.csv").fillna(0.0)
print(len(analysis_df_24))
analysis_df_25 = pd.read_csv(path + "rust_25.csv").fillna(0.0)

analysis_df_all = pd.read_csv(path + "rust_22_25.csv").fillna(0.0)

if not os.path.exists(path + "training_data_with_updated_label.csv"):
    versions_df = pd.read_csv(path + "versions.csv")
    crates_df = pd.read_csv(path + "crates.csv")
    # Clean data: drop rows where all values are 0 except proj_id
    def drop_all_zero_rows(df):
        cols_except_proj = df.columns.difference(['proj_id'])
        return df[~(df[cols_except_proj] == 0).all(axis=1)].reset_index(drop=True)


    # Extract repo from proj_id and match with crates.csv
    crates_df["repository"] = crates_df["repository"].fillna("")
    repo_map = {
        row["repository"].strip().lower(): (row["id"], row["name"])
        for _, row in crates_df.iterrows()
    }

    def extract_repo(proj_id):
        if isinstance(proj_id, str) and ":" in proj_id:
            return "https://github.com/" + proj_id.split(":")[1].lower()
        return ""


    def find_crate_info(repo):
        return repo_map.get(repo, (-1, "UNKNOWN"))

    def add_crate_info(df):
        df["repo"] = df["proj_id"].apply(extract_repo)
        crate_info = df["repo"].apply(lambda x: pd.Series(find_crate_info(x)))
        crate_info.columns = ["crate_id", "crate_name"]
        df = pd.concat([df, crate_info], axis=1)
        df = df[df["crate_name"] != "UNKNOWN"].reset_index(drop=True)
        return df

    analysis_df_24 = add_crate_info(analysis_df_24)
    analysis_df_25 = add_crate_info(analysis_df_25)
    analysis_df_all = add_crate_info(analysis_df_all)
    print(analysis_df_all)

    # Label based on updated_at from crates.csv
    crates_df["updated_at"] = pd.to_datetime(crates_df["updated_at"], errors="coerce", utc=True)
    update_map = dict(zip(crates_df["name"], crates_df["updated_at"]))
    cutoff_date = pd.to_datetime("2025-01-01", utc=True)

    active_names = set(analysis_df_25["crate_name"])
    print(len(active_names))
    def is_active(crate_name):
        updated = update_map.get(crate_name)
        if crate_name not in active_names:
            return 0
        else: return 1

    analysis_df_24["label"] = analysis_df_24["crate_name"].apply(is_active)


    # Save final feature dataset
    feature_cols = analysis_df_24.columns.difference(["proj_id", "repo", "crate_name"])
    analysis_df_24[feature_cols].to_csv(path + "training_data_with_updated_label.csv", index=False)

    label_counts = analysis_df_24["label"].value_counts()

    num_positive = label_counts.get(1, 0)
    num_negative = label_counts.get(0, 0)

    print(f"Number of positive labels (active): {num_positive}")
    print(f"Number of negative labels (inactive): {num_negative}")
else:
    df = pd.read_csv(path + "training_data_with_updated_label.csv")

    X = df.drop(columns=["label", "crate_id"])
    y = df["label"]
    scaler = StandardScaler()

    # SMOTE + RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    pipeline = Pipeline([
        ('scaler', scaler),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    y_pred = cross_val_predict(pipeline, X, y, cv=skf)

    print("Classification Report:\n")
    print(classification_report(y, y_pred, digits=4))

    pipeline.fit(X, y)

    feature_cols = X.columns.tolist()

    X_all = analysis_df_all[feature_cols]

    y_all_proba = pipeline.predict_proba(X_all)[:, 1]
    analysis_df_all["predicted_score"] = y_all_proba

    analysis_df_all[["crate_name", "crate_id", "predicted_score"]].to_csv(path + "maintenance_activity.csv", index=False)


