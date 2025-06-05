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
    DATA_DIR = "data/raw" # Directory containing raw question datasets
    FINAL_PATH = "data/final.csv" # Path to the final output CSV file
    PROGRESS_PATH = "data/progress.csv" # Path to track processed question IDs

    def __init__(self, llm_service: LLMService) -> None:
        """
       Initializes the Preprocessing class with a given LLMService instance.
       Args:
           llm_service (LLMService): The language model service used for question categorization.
       """
        self.llm_service = llm_service

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

    def run(self, target_size: int) -> None:
        """
        Executes the preprocessing pipeline to collect a specified number of causal questions.
        Samples questions from raw datasets, checks them with the LLM, and saves valid entries.
        Args:
            target_size (int): The target number of causal questions to collect.
        """
        # Load already processed IDs from progress file
        if os.path.exists(self.PROGRESS_PATH):
            progress_df = pd.read_csv(self.PROGRESS_PATH)
            processed_ids = set(progress_df["id"].astype(str))
        else:
            raise FileNotFoundError(f"Progress file not found at {self.PROGRESS_PATH}")

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