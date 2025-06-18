from services.llm_service import LLMService
import os
import random
import pandas as pd
from services.command_line_service import print_progress_bar
from preprocessing.prompt_builder import build_prompt

class Preprocessing:
    """
    Handles the preprocessing of question datasets by sampling, filtering, and categorizing questions.
    Stores the results in a final CSV file and tracks progress to avoid duplicates.
    """

    def create_sample(self, target_size: int, output_path: str) -> None:
        """
        Creates a sample of questions from the raw datasets.
        Args:
            target_size (int): Target number of questions to collect.
            output_path (str): Path to save the output file.
        """
        data_dir = "data/raw"  # Directory containing raw question datasets
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found at {data_dir}")

        print("Loading datasets")
        # Load already existing IDs from the output file, if available
        if os.path.exists(output_path):
            sample_df = pd.read_csv(output_path)
            included_ids = set(sample_df["id"].astype(str))
        else:
            included_ids = set()
            sample_df = pd.DataFrame(columns=["id", "question_processed", "answer_processed", "dataset"])

        # Read all CSV files from the data directory and keep a DataFrame per file
        csv_files = [f for f in os.listdir(self.DATA_DIR) if f.endswith(".csv")]
        file_dfs = {}
        for file in csv_files:
            df = pd.read_csv(os.path.join(self.DATA_DIR, file))
            df["dataset"] = os.path.splitext(file)[0]
            file_dfs[file] = df[~df["id"].astype(str).isin(included_ids)]
        print(f"Loaded {len(file_dfs)} datasets with {sum(len(df) for df in file_dfs.values())} questions.")

        print_progress_bar(len(sample_df), target_size)
        while len(sample_df) < target_size:
            # Only consider files that are not empty
            non_empty_files = [f for f, df in file_dfs.items() if not df.empty]
            if not non_empty_files:
                break
            chosen_file = random.choice(non_empty_files)
            df = file_dfs[chosen_file]
            row = df.sample(1).iloc[0]
            q_id = str(row["id"])

            # Prepare new row for the final DataFrame
            row_out = row[["id", "question_processed", "answer_processed", "dataset"]].copy()
            new_row_df = pd.DataFrame([row_out])

            # Add row and update ID set
            sample_df = pd.concat([sample_df, new_row_df])
            included_ids.add(q_id)
            # Remove the selected question from the DataFrame of the file
            file_dfs[chosen_file] = df[df["id"].astype(str) != q_id]
            print_progress_bar(len(sample_df), target_size)

        sample_df = sample_df.sample(frac=1)  # Shuffles the DataFrame
        sample_df.to_csv(output_path, index=False)
        print("\nSample creation finished. Output saved to:", output_path)


    def categorize_question(self, question: str) -> str:
        """
        Uses the LLM service to determine if a question is causal, clear, and sensible.
        Args:
            question (str): The question to be categorized.
        Returns:
            str: "1" if the question is causal, clear, and sensible, otherwise "0".
        """
        # For R1 and qwq models, allow more tokens for reasoning
        if "R1" in self.llm_service.llm_name or "qwq" in self.llm_service.llm_name: max_tokens = 500
        else: max_tokens = 1  # For other models, limit to 1 token for efficiency

        response = self.llm_service.get_llm_response(
            prompt=build_prompt(question, 'overall'),
            temperature=0,
            max_tokens=max_tokens
        )
        return response

    def filter_questions(self, target_size: int, llm: str, input_path: str, output_path: str) -> None:
        """
        Executes the preprocessing pipeline to collect a specified number of causal questions.
        Samples questions from raw datasets, checks them with the LLM, and saves valid entries.
        Args:
            target_size (int): The target number of causal questions to collect.
            llm (str): The name of the LLM to use for categorization.
            input_path (str): Path to the input dataset file (should end with .csv).
            output_path (str): Path to save the output dataset file (should end with .csv).
        """
        # Load already processed IDs from progress file
        if os.path.exists(input_path):
            progress_df = pd.read_csv(self.PROGRESS_PATH)
            processed_ids = set(progress_df["id"].astype(str))
        else:
            raise FileNotFoundError(f"Input file not found at {self.PROGRESS_PATH}")

        # Load current length of final file
        if os.path.exists(self.FINAL_PATH):
            final_df = pd.read_csv(self.FINAL_PATH)
            final_count = len(final_df)
        else:
            raise FileNotFoundError(f"Final file not found at {self.FINAL_PATH}")

        csv_files = [f for f in os.listdir(self.DATA_DIR) if f.endswith(".csv")] # Load all CSV files from the data directory
        print_progress_bar(final_count, target_size)
        while final_count < target_size:
            # Sample one random question from the CSV files
            file = random.choice(csv_files)
            df = pd.read_csv(os.path.join(self.DATA_DIR, file))
            df = df[~df["id"].astype(str).isin(processed_ids)]  # Only keep questions that have not been processed yet
            if df.empty:
                continue
            row = df.sample(1).iloc[0]
            q_id = str(row["id"])
            question_proc = str(row["question_processed"])

            # Save the ID to the progress file
            processed_ids.add(q_id)
            progress_df = pd.concat([progress_df, pd.DataFrame([{"id": q_id}])], ignore_index=True)
            progress_df.to_csv(self.PROGRESS_PATH, index=False)

            # Check if the question is causal and sensible
            result = self.categorize_question(question_proc)
            if result == "1":
                # Add question and answer to final DataFrame
                dataset_name = os.path.splitext(file)[0]
                row_out = row[["id", "question_processed", "answer_processed"]].copy()
                row_out["dataset"] = dataset_name
                new_row_df = pd.DataFrame([row_out])
                final_df = pd.concat([final_df, new_row_df], ignore_index=True)

                # Save updated final_df to CSV
                final_df.to_csv(self.FINAL_PATH, index=False)
                final_count += 1
                print_progress_bar(final_count, target_size)

        print("\nFinished preprocessing")