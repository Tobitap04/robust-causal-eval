from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
from nltk import PorterStemmer
from rouge import Rouge
from sacrebleu.metrics import CHRF
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def compute_metric(prediction1: str, prediction2: str, answer: str, metric: str) -> float:
    """
    Computes the specified metric between two predictions and the ground truth answer depending on the metric type.

    Args:
        prediction1 (str): First prediction string.
        prediction2 (str): Second prediction string.
        answer (str): Ground truth answer string.
        metric (str): Metric to compute ('rouge_sim', 'rouge_cor', 'bleu_sim', 'bleu_cor', 'bert_sim', 'bert_cor', chrf_sim', 'chrf_cor', 'sBert_sim', 'sBert_cor', 'nli_sim', 'nli_cor').

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
    elif metric == "chrf_sim":
        return chrf(prediction1, prediction2)
    elif metric == "chrf_cor":
        return chrf(prediction1, answer)
    elif metric == "sBert_sim":
        return sBert(prediction1, prediction2)
    elif metric == "sBert_cor":
        return sBert(prediction1, answer)
    elif metric == "nli_sim":
        return nli_entailment_score(prediction1, prediction2)
    elif metric == "nli_cor":
        return nli_entailment_score(prediction1, answer)
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


def chrf(hypothesis: str, reference: str) -> float:
    """
    Computes the CHRF score between a hypothesis and a reference.
    Args:
        hypothesis (str): The generated answer.
        reference (str): The ground truth answer or another generated answer.
    Returns:
        float: CHRF score normalized to [0, 1].
    """
    chrf_metric = CHRF()
    return chrf_metric.sentence_score(hypothesis, [reference]).score / 100


sBert_model = SentenceTransformer("all-mpnet-base-v2")

def sBert(hypothesis: str, reference: str) -> float:
    """
    Computes the cosine similarity between the embeddings of two texts.
    Args:
        hypothesis (str): The generated answer.
        reference (str): The ground truth answer or another generated answer.
    Returns:
        float: Cosine similarity score between the two text embeddings.
    """
    embeddings = sBert_model.encode([hypothesis, reference], convert_to_tensor=True)
    cosine_similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    return cosine_similarity


nli_model_name = "roberta-large-mnli"
nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)

def nli_entailment_score(hypothesis: str, reference: str) -> float:
    """
    Returns the average probability (0–1) that hypothesis and reference mutually entail each other.
    Args:
        hypothesis (str): The generated answer.
        reference (str): The ground truth answer or another generated answer.
    Returns:
        float: Average probability for entailment.
    """
    # Direction 1: hypothesis -> reference
    inputs1 = nli_tokenizer(hypothesis, reference, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits1 = nli_model(**inputs1).logits
    probs1 = torch.softmax(logits1, dim=-1).squeeze()
    score1 = probs1[2].item()

    # Direction 2: reference -> hypothesis
    inputs2 = nli_tokenizer(reference, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits2 = nli_model(**inputs2).logits
    probs2 = torch.softmax(logits2, dim=-1).squeeze()
    score2 = probs2[2].item()

    return (score1 + score2) / 2