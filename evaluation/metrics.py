from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
from nltk import PorterStemmer
from rouge import Rouge


def compute_metric(prediction1: str, prediction2: str, answer: str, metric: str) -> float:
    """
    Computes the specified metric between two predictions and the ground truth answer depending on the metric type.

    Args:
        prediction1 (str): First prediction string.
        prediction2 (str): Second prediction string.
        answer (str): Ground truth answer string.
        metric (str): Metric to compute ('rouge_sim', 'rouge_cor', 'bleu_sim', 'bleu_cor', 'bert_sim', 'bert_cor').

    Returns:
        float: Computed metric score.
    Raises:
        ValueError: If an invalid metric type is specified.
    """
    if metric == "rouge_sim":
        return rouge(prediction1, prediction2)
    elif metric == "rouge_cor":
        return rouge(prediction1, answer)
    elif metric == "bleu_sim":
        return bleu(prediction1, prediction2)
    elif metric == "bleu_cor":
        return bleu(prediction1, answer)
    elif metric == "bert_sim":
        return bert(prediction1, prediction2)
    elif metric == "bert_cor":
        return bert(prediction1, answer)
    else:
        raise ValueError(f"Invalid metric specified: {metric}.")

def rouge(hypothesis: str, reference: str) -> float:
    """
    Computes the ROUGE-L F1 score between a hypothesis and a reference answer.

    Args:
        hypothesis (str): The generated answer.
        reference (str): The ground truth answer or another generated answer.

    Returns:
        float: ROUGE-L F1 score.
    """
    # Apply stemming for robuster evaluation
    stemmer = PorterStemmer()
    stem = lambda text: " ".join(stemmer.stem(word) for word in text.split())

    hypo_stemmed = stem(hypothesis)
    ref_stemmed = stem(reference)

    rouge_scorer = Rouge()
    score = rouge_scorer.get_scores(hypo_stemmed, ref_stemmed, avg=True)

    return score['rouge-l']['f']

def bleu(hypothesis: str, reference: str) -> float:
    """
    Computes the BLEU score between a hypothesis and a reference answer.

    Args:
        hypothesis (str): The generated answer.
        reference (str): The ground truth answer or another generated answer.

    Returns:
        float: BLEU score.
    """
    reference_tokens = [reference.split()]
    hypothesis_tokens = hypothesis.split()
    smoothing_fn = SmoothingFunction().method2 # Smoothing function to handle short sentences
    return sentence_bleu(reference_tokens, hypothesis_tokens, smoothing_function=smoothing_fn)

def bert(hypothesis: str, reference: str) -> float:
    """
    Computes the BERT score F1 between a hypothesis and a reference answer.

    Args:
        hypothesis (str): The generated answer.
        reference (str): The ground truth answer or another generated answer.

    Returns:
        float: BERT score F1.
    """
    P, R, F1 = bert_score([hypothesis], [reference], lang="en", rescale_with_baseline=True)
    return F1[0].item()