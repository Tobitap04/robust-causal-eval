from evaluation.metrics import compute_metric
from evaluation.perturbation_funcs import perturbation_func
from evaluation.prompting_funcs import postprocessing_func, preprocessing_func, inprocessing_func
from services.llm_service import LLMService
import pandas as pd

class Evaluation:
    DATASET_PATH = "data/final.csv"  # Path to the causal question dataset

    def __init__(self, llm_service: LLMService, llm: str, nq: int,
                 perturbs: list[str], metrics: list[str],
                 preproc: str, inproc: str, postproc: str, temp: float) -> None:
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

    def run(self) -> None:

        # Create question sample from the dataset
        df = pd.read_csv(self.DATASET_PATH)
        if len(df) < self.num_questions:
            raise ValueError("Number of questions to evaluate is bigger than the dataset size.")
        sampled_df = df.sample(n=self.num_questions)
        sampled_questions = sampled_df.to_dict(orient='records')


        results = {metric: {perturb: [] for perturb in self.perturbation_levels} for metric in self.metrics}
        # TODO: Crash handling
        for question in sampled_questions:
            for perturbation in self.perturbation_levels:
                prompt1 = perturbation_func(question['question_processed'], perturbation)
                prompt2 = perturbation_func(question['question_processed'], perturbation)
                prompt1_preprocessed = preprocessing_func(prompt1, self.preprocessing)
                prompt2_preprocessed = preprocessing_func(prompt2, self.preprocessing)
                res1 = inprocessing_func(prompt1_preprocessed, self.inprocessing, self.llm_service, self.temperature)
                res2 = inprocessing_func(prompt2_preprocessed, self.inprocessing, self.llm_service, self.temperature)
                if res1 is None or res2 is None:
                    print(f"Error generating responses for question: {question['question_processed']}, Perturbation: {perturbation}")
                    continue
                res1_postprocessed = postprocessing_func(res1, self.postprocessing)
                res2_postprocessed = postprocessing_func(res2, self.postprocessing)
                for metric in self.metrics:
                    score = compute_metric(res1_postprocessed, res2_postprocessed, question['answer_processed'], metric)
                    results[metric][perturbation].append(score)

        # Am Ende: Durchschnitt berechnen und DataFrame erstellen
        avg_results = {metric: {perturb: (sum(scores)/len(scores) if scores else None)
                                for perturb, scores in perturbs.items()}
                       for metric, perturbs in results.items()}
        df_results = pd.DataFrame(avg_results).T
        print(df_results)