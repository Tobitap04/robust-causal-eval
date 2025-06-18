import sys
import argparse

def print_progress_bar(current: int, total: int, bar_length: int = 40) -> None:
    """
    Displays a progress bar in the terminal to indicate the current progress of a process.
    Args:
        current (int): Current progress (e.g., number of items processed).
        total (int): Total number of items to process.
        bar_length (int, optional): Length of the progress bar. Default is 40.
    """
    percent = float(current) / total
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write(f'\rProgress: [{arrow}{spaces}] {current}/{total}')
    sys.stdout.flush()

def print_evaluation_results(llm_name: str, num_questions: int, preprocessing: str, inprocessing: str, postprocessing: str,
                             temperature: float, perturbation_levels: list[str],
                             metrics: list[str], avg_results: dict) -> None:
    """
    Prints the results of the causal robustness evaluation in a formatted table.
    Args:
        llm_name (str): Name of the LLM used for evaluation.
        num_questions (int): Number of questions evaluated.
        preprocessing (str): Preprocessing method applied to the questions.
        inprocessing (str): Inprocessing method applied during question handling.
        postprocessing (str): Postprocessing method applied to the responses.
        temperature (float): Temperature setting used for the LLM.
        perturbation_levels (list[str]): List of perturbation levels tested.
        metrics (list[str]): List of metrics computed during evaluation.
        avg_results (dict): Dictionary containing average results for each metric and perturbation level.
    """

    print("\n\nCausal Robustness Evaluation")
    print("=" * 30)
    print("Input Parameters:")
    print(f"  - LLM:                  {llm_name}")
    print(f"  - Number of Questions:  {num_questions}")
    print(f"  - Preprocessing:        {preprocessing}")
    print(f"  - Inprocessing:         {inprocessing}")
    print(f"  - Postprocessing:       {postprocessing}")
    print(f"  - Temperature:          {temperature}")
    print()

    print("Evaluation Results")
    print("=" * 30)
    headers = ["Category"] + [str(p) for p in perturbation_levels]
    header_row = " | ".join([f"{h:<13}" for h in headers])
    print(header_row)
    print("-" * (18 * len(headers)))

    for metric in metrics:
        row = [metric]
        for perturb in perturbation_levels:
            val = avg_results[metric][perturb]
            row.append(f"{val:.4f}" if val is not None else "N/A")
        print(" | ".join([f"{c:<13}" for c in
                          row]) + f" (avg: {sum([avg_results[metric][p] for p in perturbation_levels if avg_results[metric][p] is not None]) / len([avg_results[metric][p] for p in perturbation_levels if avg_results[metric][p] is not None]):.4f})")

    print("-" * (18 * len(headers)))

def get_cl_args_eval() -> argparse.Namespace: # TODO: Add remaining perturbations, metrics and processing options
    """
    Parses command line arguments for evaluating LLM robustness on causal questions.
    Returns: argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate LLM robustness on causal questions.")

    parser.add_argument("--llm", type=str, default="gwdg.llama-3.3-70b-instruct",
                        help="Name of the LLM to evaluate (default: 'gwdg.llama-3.3-70b-instruct')",)

    parser.add_argument("--nq", type=int, default=1000,
                        help="Number of questions to evaluate (default: 1000)")

    parser.add_argument("--perturbs", type=str, nargs='+',
                        choices=["none", "char", "word", "sentence", "language", "all"],
                        default=["none", "char", "word", "all"],
                        help="Perturbation levels to test (default: ['none', 'char', 'word', 'sentence', 'language', 'all'])")

    parser.add_argument("--metrics", type=str, nargs='+',
                        choices=["rouge_var", "rouge_corr", "bleu_var", "bleu_corr", "bert_var", "bert_corr", "kg_var", "kg_corr"],
                        default=["rouge_var", "rouge_corr", "bleu_var", "bleu_corr", "bert_var", "bert_corr"],
                        help="Evaluation metrics to compute (default: ['rouge_var', 'rouge_corr', 'bleu_var', 'bleu_corr', 'bert_var', 'bert_corr', 'kg_var', 'kg_corr'])")

    parser.add_argument("--preproc", type=str, default="none",
                        choices=["none"],
                        help="Preprocessing of the question (default: 'none')")

    parser.add_argument("--inproc", type= str, default="none",
                        choices=["none"],
                        help="Inprocessing of the question (default: 'none')")

    parser.add_argument("--postproc", type=str, default="none",
                        choices=["none"],
                        help="Postprocessing of the question (default: 'none')")

    parser.add_argument("--temp", type=float, default=0,
                        help="Temperature setting for the LLM (default: 0)")

    parser.add_argument("--sample_path", type=str, default="data/sample.csv", help="Path to sample of the  Webis-CausalQA dataset.") # TODO: Change to data/final.csv

    return parser.parse_args()


def get_cl_args_preproc() -> argparse.Namespace:
    """
    Parses command line arguments for preprocessing question dataset.
    Returns: argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Preprocess question datasets to collect causal questions.")

    parser.add_argument("function", type=str, help="Function to execute.", choices=["data_setup", "create_sample", "filter_questions", "sample_lookup"])
    parser.add_argument("--nq", type=int, help="The number of questions to sample.", default=10000)
    parser.add_argument("--input_path", type=str, help="The path to the input dataset file (should end with .csv).", default=None)
    parser.add_argument("--output_path", type=str, help="The path to the output dataset file (should end with .csv).", default=None)
    parser.add_argument("--filter", type=str, help="The filter to apply to the questions.", choices=[], default=None) #TODO: Add filter options
    parser.add_argument("--exclude", type=str, nargs='+', default=[],
                        choices=["eli5", "gooaq", "hotpotqa", "msmarco", "naturalquestions", "newsqa", "paq", "searchqa", "squad2", "triviaqa"],
                        help="Names of the datasets to exclude from the sample (default: [])")
    parser.add_argument("--llm", type=str, default="gwdg.llama-3.3-70b-instruct",
                        help="Name of the LLM to evaluate (default: 'gwdg.llama-3.3-70b-instruct')")

    return parser.parse_args()