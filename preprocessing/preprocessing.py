from services.llm_service import LLMService
import os
import re
import random
import logging
import pandas as pd
from services.command_line_service import print_progress_bar
from preprocessing.filter_funcs import build_prompt
from preprocessing.perturbation_funcs import perturbation_func

class Preprocessing:
    """
    Handles the preprocessing of question datasets by sampling, filtering, and categorizing questions.
    Stores the results in a final CSV file and tracks progress to avoid duplicates.
    """

    def __init__(self, llm_service: LLMService):
        """
        Initializes the Preprocessing class with LLM service.
        Args:
            llm_service (LLMService): An instance of the LLMService to use for question categorization.
        """
        self.llm_service = llm_service

    @staticmethod
    def create_sample(nq: int, output_path: str, exclude: list[str]) -> None:
        """
        Creates a sample of questions from the raw datasets.
        Args:
            nq (int): Number of questions to sample.
            output_path (str): Path to save the output file.
            exclude (list[str]): List of datasets to exclude from sampling.
        Raises:
            FileNotFoundError: If the raw data directory does not exist.
        """
        raw_dir = "data/raw"  # Directory containing raw question of the  Webis-CausalQA dataset
        if not os.path.exists(raw_dir):
            raise FileNotFoundError(f"Data directory not found at {raw_dir}")

        print("Loading datasets")
        # Load already existing IDs from the output file, if available
        if os.path.exists(output_path):
            sample_df = pd.read_csv(output_path)
            included_ids = set(sample_df["id"].astype(str))
        else:
            included_ids = set()
            sample_df = pd.DataFrame(columns=["id", "question_processed", "answer_processed", "dataset"])

        # Read all CSV files from the data directory which are not in the exclude list and keep a DataFrame per file
        csv_files = [f for f in os.listdir(raw_dir) if f.endswith(".csv") and os.path.splitext(f)[0] not in exclude]
        file_dfs = {}
        for file in csv_files:
            df = pd.read_csv(os.path.join(raw_dir, file))
            df["dataset"] = os.path.splitext(file)[0]
            file_dfs[file] = df[~df["id"].astype(str).isin(included_ids)]
        print(f"Loaded {len(file_dfs)} datasets with {sum(len(df) for df in file_dfs.values())} questions.")

        print_progress_bar(len(sample_df), nq)
        while len(sample_df) < nq:
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
            print_progress_bar(len(sample_df), nq)

        sample_df = sample_df.sample(frac=1)  # Shuffles the DataFrame
        sample_df.to_csv(output_path, index=False)
        print("\nSample creation finished. Output saved to:", output_path)

    @staticmethod
    def sample_lookup(input_path: str, nq: int) -> None:
       """
       Reads the CSV file from `input_path`, samples `nq` questions, and prints question, answer, and dataset to the console.
       Args:
           input_path (str): Path to the input CSV file.
           nq (int): Number of questions to sample and display.
       Raises:
           FileNotFoundError: If the input file does not exist.
       """
       if not os.path.exists(input_path):
           raise FileNotFoundError(f"File not found: {input_path}")
       df = pd.read_csv(input_path)
       if df.empty:
           print("The file contains no questions.")
           return
       sample_df = df.sample(n=min(nq, len(df)))
       print(f"Sampled {len(sample_df)} questions from {input_path}:\n")
       for _, row in sample_df.iterrows():
           print(f"Question: {row['question_processed']}\nAnswer: {row['answer_processed']}\n{'-'*40}")

    @staticmethod
    def sample_stats(input_path: str) -> None:
        """
        Prints statistics for each dataset in the given CSV file, showing how many questions are present per dataset.
        Args:
            input_path (str): Path to the input CSV file.
        Raises:
            FileNotFoundError: If the input file does not exist.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File not found: {input_path}")
        df = pd.read_csv(input_path)
        if df.empty:
            print("The file contains no questions.")
            return
        dataset_counts = df["dataset"].value_counts()
        print(f"Statistics of questions per dataset in file '{input_path}':")
        for ds, count in dataset_counts.items():
            print(f"{ds}: {count} questions")
        print(f"Total questions: {len(df)}")

    def create_perturbs(self, input_path: str, output_path: str, intensity: int) -> None:
        """
        Creates perturbations of questions from the input file with defined intensity and saves them to the output file.
        The function is robust for long runtimes and can resume if interrupted.
        Args:
            input_path (str): Path to the input CSV file with questions.
            output_path (str): Path to the output file.
            intensity (int): Intensity of the perturbations.
        Raises:
            FileNotFoundError: If the input file does not exist.
            ValueError: If the file does not have the expected format.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File not found: {input_path}")
        in_df = pd.read_csv(input_path)

        if in_df.empty:
            print("The input file contains no questions.")
            return

        if list(in_df.columns) != ["id", "question_processed", "answer_processed", "dataset"]:
            raise ValueError(f"Input File {input_path} not in expected format.")

        expected_columns = [
            "id", "question_none_perturb", "question_char_perturb", "question_synonym_perturb",
            "question_language_perturb", "question_paraphrase_perturb", "question_sentence_inj_perturb",
            "question_bias_perturb", "answer", "dataset"
        ]

        # Load output file if it exists, otherwise create a new DataFrame
        if os.path.exists(output_path):
            out_df = pd.read_csv(output_path)
            if list(out_df.columns) != expected_columns:
                raise ValueError(f"Output File {output_path} not in expected format.")
        else:
            out_df = pd.DataFrame()
            out_df["id"] = in_df["id"].astype(str)
            out_df["question_none_perturb"] = in_df["question_processed"].astype(str)
            out_df["question_char_perturb"] = ""
            out_df["question_synonym_perturb"] = ""
            out_df["question_language_perturb"] = ""
            out_df["question_paraphrase_perturb"] = ""
            out_df["question_sentence_inj_perturb"] = ""
            out_df["question_bias_perturb"] = ""
            out_df["answer"] = in_df["answer_processed"].astype(str)
            out_df["dataset"] = in_df["dataset"].astype(str)

        levels = [
            "char", "synonym", "language", "paraphrase", "sentence_inj", "bias"
        ]
        col_map = {
            "char": "question_char_perturb",
            "synonym": "question_synonym_perturb",
            "language": "question_language_perturb",
            "paraphrase": "question_paraphrase_perturb",
            "sentence_inj": "question_sentence_inj_perturb",
            "bias": "question_bias_perturb"
        }

        print(f"Starting creating perturbations of {input_path}")
        total = len(out_df)
        for idx, row in out_df.iterrows():
            print_progress_bar(idx, total)
            for level in levels:
                col = col_map[level]
                if pd.isna(row[col]) or row[col] == "":
                    q = str(row["question_none_perturb"])
                    try:
                        perturbed = perturbation_func(q, level, intensity, self.llm_service)
                    except Exception as e:
                        logging.error(f"Preprocessing: Error while perturbation of question {row['id']}: {e}\n\n")
                        perturbed = ""
                    out_df.at[idx, col] = str(perturbed)
            # Regularly save progress
            if idx % 5 == 0:
                out_df.to_csv(output_path, index=False)
        print_progress_bar(idx+1, total)
        out_df.to_csv(output_path, index=False)
        print("\nPerturbations created and saved to:", output_path)

    def filter_questions(self, input_path: str, output_path: str, filter_type: str) -> None:
        """
        Filters questions from the input file using categorize_question and saves the filtered questions to output_path.
        Each question is checked exactly once. At the end, prints statistics about how many questions were removed per dataset.
        The function is robust for long runtimes and can resume if interrupted.
        Args:
            input_path (str): Path to the input CSV file containing questions.
            output_path (str): Path to save the filtered questions.
            filter_type (str): The filter to apply to the questions, which is used in the LLM prompt.
        Raises:
            FileNotFoundError: If the input file does not exist.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File not found: {input_path}")
        df = pd.read_csv(input_path)
        if df.empty:
            print("The input file contains no questions.")
            return

        # Load progress if output file exists
        if os.path.exists(output_path):
            out_df = pd.read_csv(output_path)
            # Determine the ID of the last entry in the output file
            if not out_df.empty:
                last_id = str(out_df.iloc[-1]["id"])
                # Find the index of the last processed question in the input DataFrame
                if last_id in df["id"].astype(str).values:
                    start_idx = df.index[df["id"].astype(str) == last_id][0] + 1
                else:
                    start_idx = 0
            else:
                start_idx = 0
        else:
            out_df = pd.DataFrame(columns=df.columns)
            start_idx = 0

        print(f"Starting filtering questions from {input_path} with filter '{filter_type}'")
        stats = {}
        total = len(df)
        for idx, row in enumerate(df.itertuples(index=False), start=0):
            # Skip already processed questions
            if idx < start_idx:
                continue
            print_progress_bar(idx, total)
            q_id = str(getattr(row, "id"))
            dataset = getattr(row, "dataset", None)
            try:
                result = self.categorize_question(str(getattr(row, "question_processed")),
                                                  str(getattr(row, "answer_processed")), filter_type)
            except Exception as e:
                logging.error(f"Preprocessing: Error with question {q_id}: {e}\n\n")
                continue
            if result == "1":
                out_df = pd.concat([out_df, pd.DataFrame([row])], ignore_index=True)
            else:
                stats[dataset] = stats.get(dataset, 0) + 1
            # Save progress for robustness
            if idx % 20 == 0:
                out_df.to_csv(output_path, index=False)
        out_df.to_csv(output_path, index=False)
        print_progress_bar(idx+1, total)
        print("\nFiltering finished. Output saved to:", output_path)
        print("Removed questions per dataset:")
        for ds, count in stats.items():
            print(f"{ds}: {count} removed")

    def categorize_question(self, question: str, answer: str, filter_type: str) -> str:
        """
        Evaluates a question using the LLM service to determine whether it matches the criteria defined by the given filter.
        Args:
            question (str): The question to be categorized.
            answer (str, optional): The answer to the question. Defaults to None.
            filter_type (str): The filter to apply, which is used in the LLM prompt.
        Returns:
            str: ‘1’ if the question is fine, ‘0’ if the issue applies.
        Raises:
            ValueError: If the output from the LLM does not match the expected format.
        """

        response = self.llm_service.get_llm_response(
            prompt=build_prompt(question, answer, filter_type),
        )

        # Output only the result itself
        match = re.search(r"<result>(\d+)</result>", response)
        if match:
            return match.group(1)
        else:
           raise ValueError(f"Unexpected response format for filter '{filter_type}': {response}")