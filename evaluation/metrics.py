from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.chrf_score import sentence_chrf
from bert_score import score as bert_score
from rouge_score import rouge_scorer
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging
logging.set_verbosity_error()


def compute_metric(hypothesis: str, reference: str, answer: str, question: str, metric: str) -> float:
    """
    Computes the specified metric between two predictions and the ground truth answer depending on the metric type.

    Args:
        hypothesis: The output produced by the LLM in response to the perturbed question.
        reference: The result without any perturbations.
        answer (str): Ground truth answer string.
        question (str): The (perturbed) question before any processing.
        metric (str): Metric to compute ('rouge_sim', 'rouge_cor', 'bleu_sim', 'bleu_cor', 'bert_sim', 'bert_cor', chrf_sim', 'chrf_cor', 's_bert_sim', 's_bert_cor', 'nli_sim', 'nli_cor', 'q_len'. 'ans_len').
    Returns:
        float: Computed metric score.
    Raises:
        ValueError: If an invalid metric type is specified.
    """
    if metric == "rouge_sim":
        return rouge(hypothesis, reference)
    elif metric == "rouge_cor":
        return rouge(hypothesis, answer)
    elif metric == "bleu_sim":
        return bleu(hypothesis, reference)
    elif metric == "bleu_cor":
        return bleu(hypothesis, answer)
    elif metric == "bert_sim":
        return bert(hypothesis, reference)
    elif metric == "bert_cor":
        return bert(hypothesis, answer)
    elif metric == "chrf_sim":
        return chrf(hypothesis, reference)
    elif metric == "chrf_cor":
        return chrf(hypothesis, answer)
    elif metric == "s_bert_sim":
        return s_bert(hypothesis, reference)
    elif metric == "s_bert_cor":
        return s_bert(hypothesis, answer)
    elif metric == "nli_sim":
        return nli_entailment_score(hypothesis, reference)
    elif metric == "nli_cor":
        return nli_entailment_score(hypothesis, answer)
    elif metric == "q_len":
        return len(question.split())
    elif metric == "ans_len":
        return len(hypothesis.split())
    else:
        raise ValueError(f"Invalid metric specified: {metric}.")


rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
def rouge(hypothesis: str, reference: str) -> float:
    """
    Computes the ROUGE-L F1 score between a hypothesis and a reference answer.

    Args:
        hypothesis (str): The generated answer.
        reference (str): The ground truth answer or another generated answer.

    Returns:
        float: ROUGE-L F1 score.
    """
    score = rouge_scorer.score(reference, hypothesis)
    return score['rougeL'].fmeasure


smoothing_fn = SmoothingFunction()
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
    return sentence_bleu(reference_tokens, hypothesis_tokens, smoothing_function=smoothing_fn.method3)


def bert(hypothesis: str, reference: str) -> float:
    """
    Computes the BERT score F1 between a hypothesis and a reference answer.

    Args:
        hypothesis (str): The generated answer.
        reference (str): The ground truth answer or another generated answer.

    Returns:
        float: BERT score F1.
    """
    P, R, F1 = bert_score([hypothesis], [reference], model_type='bert-base-uncased', lang='en')
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
    return sentence_chrf(reference, hypothesis)


sBert_model = SentenceTransformer("all-mpnet-base-v2")
def s_bert(hypothesis: str, reference: str) -> float:
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