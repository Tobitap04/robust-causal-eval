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
                             metrics: list[str], avg_results: dict, datasets: list[str]) -> None:
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
        datasets (list[str]): List of datasets included in the evaluation.
    """

    print("\n\nCausal Robustness Evaluation")
    print("=" * 30)
    print("Input Parameters:")
    print(f"  - LLM:                  {llm_name}")
    print(f"  - Datasets:             {', '.join(datasets)}")
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


def print_evaluation_results_latex(llm_name: str, num_questions: int, preprocessing: str, inprocessing: str, postprocessing: str,
                             temperature: float, perturbation_levels: list[str],
                             metrics: list[str], avg_results: dict, datasets: list[str]) -> None:
    """
    Prints the results of the causal robustness evaluation in LaTeX format for inclusion in a report.
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
        datasets (list[str]): List of datasets included in the evaluation.
    """
    print()
    print("\\section*{Causal Robustness Evaluation}")
    print("\\textbf{Input Parameters}")
    print("\\begin{itemize}")
    print(f"  \\item LLM: \\texttt{{{llm_name}}}")
    print(f"  \\item Datasets: \\texttt{{{', '.join(datasets)}}}")
    print(f"  \\item Number of Questions: \\texttt{{{num_questions}}}")
    print(f"  \\item Preprocessing: \\texttt{{{preprocessing}}}")
    print(f"  \\item Inprocessing: \\texttt{{{inprocessing}}}")
    print(f"  \\item Postprocessing: \\texttt{{{postprocessing}}}")
    print(f"  \\item Temperature: \\texttt{{{temperature}}}")
    print("\\end{itemize}")

    print("\\vspace{1em}")
    print("\\textbf{Evaluation Results}")
    cols = "l" + "c" * len(perturbation_levels) + "c"
    print(f"\\begin{{tabular}}{{|{cols}|}}")
    print("\\hline")

    # Header
    headers = ["Category"] + [str(p) for p in perturbation_levels] + ["Avg"]
    print(" & ".join(headers) + " \\\\ \\hline")

    # Rows
    for metric in metrics:
        row = [metric]
        vals = []
        for perturb in perturbation_levels:
            val = avg_results[metric][perturb]
            if val is not None:
                vals.append(val)
                row.append(f"{val:.4f}")
            else:
                row.append("N/A")
        avg = sum(vals) / len(vals) if vals else 0
        row.append(f"{avg:.4f}")
        print(" & ".join(row) + " \\\\ \\hline")

    print("\\end{tabular}")

def get_cl_args_eval() -> argparse.Namespace: # TODO: Add processing options
    """
    Parses command line arguments for evaluating LLM robustness on causal questions.
    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate LLM robustness on causal questions.")

    parser.add_argument("--llm", type=str, default="gwdg.llama-3.3-70b-instruct",
                        help="Name of the LLM to evaluate (default: 'gwdg.llama-3.3-70b-instruct')",)

    parser.add_argument("--nq", type=int, default=1000,
                        help="Number of questions to evaluate (default: 1000)")

    parser.add_argument("--perturbs", type=str, nargs='+',
                        choices=["none", "char", "synonym", "language", "paraphrase", "sentence_inj", "bias"],
                        default=["none", "char", "synonym", "language", "paraphrase", "sentence_inj", "bias"],
                        help="Perturbation levels to test (default: ['none', 'char', 'synonym', 'language', 'paraphrase', 'sentence_inj', 'bias'])")

    parser.add_argument("--metrics", type=str, nargs='+',
                        choices=["rouge_sim", "rouge_cor", "bleu_sim", "bleu_cor", "chrf_sim", "chrf_cor", "bert_sim", "bert_cor", "s_bert_sim", "s_bert_cor", "nli_sim", "nli_cor", "q_len", "ans_len"],
                        default=["rouge_sim", "rouge_cor", "bleu_sim", "bleu_cor", "chrf_sim", "chrf_cor", "bert_sim", "bert_cor", "s_bert_sim", "s_bert_cor", "nli_sim", "nli_cor", "q_len", "ans_len"],
                        help="Evaluation metrics to compute (default: ['rouge_sim', 'rouge_cor', 'bleu_sim', 'bleu_cor', 'chrf_sim', 'chrf_cor','bert_sim', 'bert_cor', 's_bert_sim', 's_bert_corr', 'nli_sim', 'nli_cor', 'q_len', 'ans_len'])")

    parser.add_argument("--preproc", type=str, default="none",
                        choices=["none", "word_constraint"],
                        help="Preprocessing of the question (default: 'none')")

    parser.add_argument("--inproc", type= str, default="none",
                        choices=["none", "few_shot"],
                        help="Inprocessing of the question (default: 'none')")

    parser.add_argument("--postproc", type=str, default="none",
                        choices=["none"],
                        help="Postprocessing of the question (default: 'none')")

    parser.add_argument("--temp", type=float, default=0,
                        help="Temperature setting for the LLM (default: 0)")

    parser.add_argument("--sample_path", type=str, default="data/final_sample.csv", help="Path to the perturbed sample of the Webis-CausalQA dataset.")

    parser.add_argument("--latex", type=bool, default=False,
                        help="If true, prints results in LaTeX format for inclusion in a report (default: False)")

    parser.add_argument("--datasets", type=str, nargs='+',
                        default=["eli5", "gooaq", "msmarco", "naturalquestions", "squad2"],
                        choices=["eli5", "gooaq", "msmarco", "naturalquestions", "squad2"],
                        help="Names of the datasets to include in the evaluation (default: ['eli5', 'gooaq', 'msmarco', 'naturalquestions', 'squad2'])")

    return parser.parse_args()


def get_cl_args_preproc() -> argparse.Namespace:
    """
    Parses command line arguments for preprocessing question dataset.
    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Preprocess question datasets to collect causal questions.")

    parser.add_argument("function", type=str, help="Function to execute.", choices=["data_setup", "create_sample", "filter_questions", "sample_lookup", "sample_stats", "create_perturbs"])
    parser.add_argument("--nq", type=int, help="The number of questions to sample.", default=6000)
    parser.add_argument("--input_path", type=str, help="The path to the input dataset file (should end with .csv).", default=None)
    parser.add_argument("--output_path", type=str, help="The path to the output dataset file (should end with .csv).", default=None)
    parser.add_argument("--filter", type=str, help="The filter to apply to the questions.", choices=["causal_chain", "answer", "question"], default=None)
    parser.add_argument("--exclude", type=str, nargs='+', default=[],
                        choices=["eli5", "gooaq", "hotpotqa", "msmarco", "naturalquestions", "newsqa", "paq", "searchqa", "squad2", "triviaqa"],
                        help="Names of the datasets to exclude from the sample (default: [])")
    parser.add_argument("--intensity", type=int, help="The intensity of the perturbation. If not set, a custom intensity value is used for each perturbation.", default=None)
    parser.add_argument("--llm", type=str, default="gwdg.qwen2.5-72b-instruct",
                        help="Name of the LLM to evaluate (default: 'gwdg.qwen2.5-72b-instruct')")

    return parser.parse_args()