from evaluation.metrics import compute_metric
from evaluation.prompting_funcs import processing_func
from services.command_line_service import print_evaluation_results, print_progress_bar, save_evaluation_results_latex
from services.llm_service import LLMService
import pandas as pd
import logging

class Evaluation:

    def __init__(self, llm_service: LLMService, llm: str, nq: int,
                 perturbs: list[str], metrics: list[str],
                 preproc: str, inproc: str, postproc: str, temp: float, sample_path: str, datasets: list[str], latex: bool) -> None:
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
            latex (bool): Whether to format the output for LaTeX.
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
        self.latex = latex

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

        print("Starting evaluation.")
        for idx, question in enumerate(sampled_questions, 0):
           print_progress_bar(idx, num_questions)
           try:
               question_text = question.get('question_none_perturb')
               answer_text = question.get('answer')
               dataset = question.get('dataset')
               if question_text is None or answer_text is None:
                   print()
                   logging.warning(f"Evaluation: Question or answer missing at index {idx}. Skipping.")
                   continue

               reference_response = processing_func(question_text, self.preprocessing, self.inprocessing, self.postprocessing, dataset, self.llm_service, self.temperature)
               for perturbation in self.perturbation_levels:
                   #print(f"\r\033[KEvaluation in progress: Question {idx}/{num_questions}: \"{question_text[:60]}...\" | Perturbation: {perturbation} | Status: Generating results...", end="", flush=True)
                   try:
                       perturbed_question = question.get(f'question_{perturbation}_perturb')
                       hypothesis_response = processing_func(perturbed_question, self.preprocessing, self.inprocessing, self.postprocessing, dataset, self.llm_service, self.temperature)
                       #print(f"\nQuestion:\n {perturbed_question}")
                       #print(f"Response:\n {hypothesis_response}")
                       #print(f"Reference:\n {reference_response}")
                       #print(f"Ground Truth Answer:\n {answer_text}")
                       for metric in self.metrics:
                           try:
                               score = compute_metric(hypothesis_response, reference_response, answer_text, perturbed_question, metric)
                               results[metric][perturbation].append(score)
                               #print(f"Metric {metric}: {score}")
                               #print(f"\rEvaluation in progress: Question {idx}/{num_questions}: \"{question_text[:60]}...\" | Perturbation: {perturbation} | Metric: {metric} | Status: Score calculated: {score:.4f}", end="", flush=True)
                           except Exception as e:
                               print()
                               logging.error(f"Evaluation: Error while computing metric {metric} of question {idx}: {e}")
                   except Exception as e:
                       print()
                       logging.error(f"Evaluation: Error during perturbation {perturbation} of question {idx}: {e}")
           except Exception as e:
               print()
               logging.error(f"Evaluation: Error with question {idx}: {e}")
        print_progress_bar(num_questions, num_questions)
        avg_results = {metric: {perturb: (sum(scores) / len(scores) if scores else None)
                                for perturb, scores in perturbs.items()}
                       for metric, perturbs in results.items()}
        self.sample_path = self.sample_path.split("/")[-1].removesuffix(".csv")
        print_evaluation_results(llm_name=self.llm_name, num_questions=self.num_questions,
                                 temperature=self.temperature, preprocessing=self.preprocessing,
                                 inprocessing=self.inprocessing, postprocessing=self.postprocessing,
                                 metrics=self.metrics, avg_results=avg_results,
                                 perturbation_levels=self.perturbation_levels, datasets=self.datasets, sample_path=self.sample_path)

        if self.latex:
            save_evaluation_results_latex(llm_name=self.llm_name, num_questions=self.num_questions,
                                     temperature=self.temperature, preprocessing=self.preprocessing,
                                     inprocessing=self.inprocessing, postprocessing=self.postprocessing,
                                     metrics=self.metrics, avg_results=avg_results,
                                     perturbation_levels=self.perturbation_levels, datasets=self.datasets, sample_path=self.sample_path)