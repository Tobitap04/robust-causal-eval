from collections import defaultdict
import pandas as pd
import shutil
import nlp
import os


raw_dir = "data/raw"  # Directory where original split of the Webis-CausalQA dataset is stored

def run_data_setup():
    """Sets up the raw data by downloading the ELI5 dataset, merging corresponding CSV files, and deleting unnecessary columns."""
    print("Starting data setup...")
    create_eli5_dataset()
    merge_datasets()
    delete_unnecessary_columns()
    print("Data setup completed.")

def delete_unnecessary_columns():
    """Deletes all columns except "id", "question", and "answer" from all CSV files in the raw directory."""
    print("Deleting unnecessary columns from CSV files in the raw directory...")
    for filename in os.listdir(raw_dir):
        if filename.endswith(".csv"):
            path = os.path.join(raw_dir, filename)
            df = pd.read_csv(path)
            cols = [col for col in ["id", "question", "answer"] if col in df.columns]
            df = df[cols]
            df.to_csv(path, index=False)

def merge_datasets():
    """Merges all CSV files in the raw directory that share the same prefix before the first "_" into a single file."""
    print("Merging CSV files in the raw directory...")
    csv_files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]

    # Group files by prefix before the first "_"
    groups = defaultdict(list)
    for filename in csv_files:
        prefix = filename.split("_")[0]
        groups[prefix].append(filename)

    # Merge files in each group and save to a new file
    for prefix, files in groups.items():
        dfs = []
        for f in files:
            path = os.path.join(raw_dir, f)
            dfs.append(pd.read_csv(path))
        merged_df = pd.concat(dfs, ignore_index=True)
        merged_path = os.path.join(raw_dir, f"{prefix}.csv")
        merged_df.to_csv(merged_path, index=False)
        # Delete the original files after merging
        for f in files:
            file_path = os.path.join(raw_dir, f)
            if os.path.exists(file_path):
                os.remove(file_path)


def create_eli5_dataset():
    """Downloads the ELI5 dataset and processes it to keep only the questions and answers belonging to the Webis-CausalQA dataset."""
    # Paths to the CSV files of the original splits (which only contain the IDs of the questions)
    train_ids_path = raw_dir + "/eli5_train_original_split.csv"
    valid_ids_path = raw_dir + "/eli5_valid_original_split.csv"

    if os.path.isfile(train_ids_path) or os.path.isfile(valid_ids_path):
        print("Downloading ELI5 dataset...")
        # Load the ELI5 dataset into the local cache folder "data/"
        eli5 = nlp.load_dataset("eli5", cache_dir="data/")

        # Access the train and validation splits
        train_set = eli5['train_eli5']
        val_set = eli5['validation_eli5']

        print("Processing ELI5 dataset...")
        # Convert splits to pandas DataFrames
        train_df = train_set.data.to_pandas()
        val_df = val_set.data.to_pandas()

        def process_df(df, ids_path):
            if os.path.isfile(ids_path):
                original_df = pd.read_csv(ids_path)
                original_ids = original_df["id"].astype(str).tolist()
                df = df[df["q_id"].isin(original_ids)].copy()

                # Build "question"
                def build_question(row):
                    if pd.isna(row["selftext"]) or row["selftext"] in ["[deleted]", "[removed]"]:
                        return row["title"]
                    else:
                        return f"{row['title']} {row['selftext']}"

                df["question"] = df.apply(build_question, axis=1)
                df["answer"] = df["answers"].apply(lambda ans: ans["text"][0])

                # Save only id, question, and answer
                df = df[["q_id", "question", "answer"]].copy()
                df.columns = ["id", "question", "answer"]
                df.to_csv(ids_path, index=False)

        process_df(train_df, train_ids_path)
        process_df(val_df, valid_ids_path)

        # Delete cache directory
        eli5_dir = os.path.join("data", "eli5")
        if os.path.isdir(eli5_dir):
            shutil.rmtree(eli5_dir)