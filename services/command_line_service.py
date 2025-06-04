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

def get_cl_args() -> argparse.Namespace:
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
                        default=["none", "char", "word", "all"],  # TODO: Add remaining perturbations
                        help="Perturbation levels to test (default: ['none', 'char', 'word', 'sentence', 'language', 'all'])")

    parser.add_argument("--metrics", type=str, nargs='+',
                        choices=["rouge_var", "rouge_corr", "bleu_var", "bleu_corr", "bert_var", "bert_corr", "kg_var", "kg_corr"],
                        default=["rouge_var", "rouge_corr", "bleu_var", "bleu_corr", "bert_var", "bert_corr"], # TODO: Add remaining metrics
                        help="Evaluation metrics to compute (default: ['All'])")

    parser.add_argument("--preproc", type=str, default="none",
                        choices=["none"],
                        help="Preprocessing of the question (default: 'none')")

    parser.add_argument("--inproc", type=str, default="none",
                        choices=["none"],
                        help="Inprocessing of the question (default: 'none')")

    parser.add_argument("--postproc", type=str, default="none",
                        choices=["none"],
                        help="Postprocessing of the question (default: 'none')")

    parser.add_argument("--temp", type=float, default=0,
                        help="Temperature setting for the LLM (default: 0)")

    return parser.parse_args()