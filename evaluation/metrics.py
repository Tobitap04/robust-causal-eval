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
        metric (str): Metric to compute ('rouge_var', 'rouge_corr', 'bleu_var', 'bleu_corr', 'bert_var', 'bert_corr', 'kg_var', 'kg_corr').

    Returns:
        float: Computed metric score.
    Raises:
        ValueError: If an invalid metric type is specified.
    """
    if metric == "rouge_var":
        return rouge(prediction1, prediction2)
    elif metric == "rouge_corr":
        return rouge(prediction1, answer)
    elif metric == "bleu_var":
        return bleu(prediction1, prediction2)
    elif metric == "bleu_corr":
        return bleu(prediction1, answer)
    elif metric == "bert_var":
        return bert(prediction1, prediction2)
    elif metric == "bert_corr":
        return bert(prediction1, answer)
    elif metric == "kg_var":
        return kg_based_similarity(prediction1, prediction2)
    elif metric == "kg_corr":
        return kg_based_similarity(prediction1, answer)
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

def kg_based_similarity(hypothesis: str, reference: str) -> float:
    """
    Computes the knowledge graph-based similarity between a hypothesis and a reference answer.

    Args:
        hypothesis (str): The generated answer.
        reference (str): The ground truth answer or another generated answer.

    Returns:
        float: Knowledge graph-based similarity score.
    """
    pass # TODO: Implement knowledge graph-based similarity computation

