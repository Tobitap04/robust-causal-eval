import re
from evaluation.metrics import compute_metric
from evaluation.perturbation_funcs import perturbation_func
from evaluation.prompting_funcs import postprocessing_func, preprocessing_func, inprocessing_func
from services.command_line_service import print_evaluation_results, print_progress_bar
from services.llm_service import LLMService
import pandas as pd
import logging

class Evaluation:

    def __init__(self, llm_service: LLMService, llm: str, nq: int,
                 perturbs: list[str], metrics: list[str],
                 preproc: str, inproc: str, postproc: str, temp: float, sample_path: str, datasets: list[str]) -> None:
        """
        Initializes the Evaluation class with the LLM service and evaluation parameters.
        Args:
            llm_service (LLMService): The language model service used for evaluation.
            llm (str): Name of the LLM to evaluate.
            nq (int): Number of questions to evaluate.
            perturbs (list[str]): Perturbation levels to test.
            metrics (list[str]): Evaluation metrics to compute.
            preproc (str): Preprocessing method for the questions.
            inproc (str): Inprocessing method for the questions.
            postproc (str): Postprocessing method for the questions.
            temp (float): Temperature to use.
            sample_path (str): Path to sample of the  Webis-CausalQA dataset.
            datasets (list[str]): Datasets to include in the evaluation.
        """
        self.llm_service = llm_service
        self.llm_name = llm # Format checked in llm_service.py
        if nq < 1:
            raise ValueError("Number of questions to evaluate must be at least 1.")
        self.num_questions = nq
        self.perturbation_levels = perturbs # Format checked in get_cl_args()
        self.metrics = metrics # Format checked in get_cl_args()
        self.preprocessing = preproc # Format checked in get_cl_args()
        self.inprocessing = inproc  # Format checked in get_cl_args()
        self.postprocessing = postproc  # Format checked in get_cl_args()
        if not 0 <= temp <= 2:
            raise ValueError("Temperature must be between 0 and 2.")
        self.temperature = temp
        self.sample_path = sample_path # Checked in run()
        self.datasets = datasets  # Format checked in get_cl_args()

    def run(self) -> None:
        """
        Handles the evaluation process by sampling questions, applying perturbations, and computing metrics.
        Raises:
            ValueError: If an invalid perturbation technique is specified.
        """
        # Create question sample from the dataset
        df = pd.read_csv(self.sample_path)
        df_filtered = df[df['dataset'].isin(self.datasets)]
        if len(df_filtered) < self.num_questions:
            raise ValueError("Number of questions to evaluate is bigger than the dataset size.")
        sampled_df = df_filtered.sample(n=self.num_questions)
        sampled_questions = sampled_df.to_dict(orient='records')

        results = {metric: {perturb: [] for perturb in self.perturbation_levels} for metric in self.metrics}
        num_questions = len(sampled_questions)


        for idx, question in enumerate(sampled_questions, 0):
           print_progress_bar(idx, num_questions)
           try:
               question_text = question.get('question_processed')
               answer_text = question.get('answer_processed')
               if question_text is None or answer_text is None:
                   logging.warning(f"Evaluation: Question or answer missing at index {idx}. Skipping.\n\n")
                   continue
               for perturbation in self.perturbation_levels:
                   #print(f"\r\033[KEvaluation in progress: Question {idx}/{num_questions}: \"{question_text[:60]}...\" | Perturbation: {perturbation} | Status: Generating results...", end="", flush=True)
                   try:
                       answer_words_count = len(re.findall(r'\b\w+\b', answer_text))
                       prompt1 = perturbation_func(question_text, perturbation)
                       prompt2 = perturbation_func(question_text, perturbation)
                       prompt1_preprocessed = preprocessing_func(prompt1, self.preprocessing, answer_words_count)
                       prompt2_preprocessed = preprocessing_func(prompt2, self.preprocessing, answer_words_count)
                       res1 = inprocessing_func(prompt1_preprocessed, self.inprocessing, self.llm_service, self.temperature)
                       res2 = inprocessing_func(prompt2_preprocessed, self.inprocessing, self.llm_service, self.temperature)
                       res1_postprocessed = postprocessing_func(res1, self.postprocessing)
                       res2_postprocessed = postprocessing_func(res2, self.postprocessing)
                       #print(f"Question 1:\n {prompt1}")
                       #print(f"Response 1:\n {res1_postprocessed}")
                       #print(f"Question 2:\n {prompt2}")
                       #print(f"Response 2:\n {res2_postprocessed}")
                       #print(f"Ground Truth Answer:\n {answer_text}")
                       for metric in self.metrics:
                           try:
                               score = compute_metric(res1_postprocessed, res2_postprocessed, answer_text, metric)
                               results[metric][perturbation].append(score)
                               #print(f"Metric {metric}: {score}")
                               #print(f"\rEvaluation in progress: Question {idx}/{num_questions}: \"{question_text[:60]}...\" | Perturbation: {perturbation} | Metric: {metric} | Status: Score calculated: {score:.4f}", end="", flush=True)
                           except Exception as e:
                               logging.error(f"Evaluation: Error while computing metric '{metric}': {e}\n\n")
                   except Exception as e:
                       logging.error(f"Evaluation: Error during perturbation '{perturbation}': {e}\n\n")
           except Exception as e:
               logging.error(f"Evaluation: Error with question {idx}: {e}\n\n")
        print_progress_bar(num_questions, num_questions)
        avg_results = {metric: {perturb: (sum(scores) / len(scores) if scores else None)
                                for perturb, scores in perturbs.items()}
                       for metric, perturbs in results.items()}

        print_evaluation_results(llm_name=self.llm_name, num_questions=self.num_questions, temperature=self.temperature, preprocessing=self.preprocessing, inprocessing=self.inprocessing, postprocessing=self.postprocessing, metrics=self.metrics, avg_results=avg_results, perturbation_levels=self.perturbation_levels, datasets=self.datasets)

