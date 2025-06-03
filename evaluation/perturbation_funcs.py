import random
def perturbation_func(question: str, level: str) -> str:
    """
    Perturbation function that applies a specified level of perturbation to the question.
    Args:
        question (str): The question to perturb.
        level (str): The level of perturbation to apply ('none', 'char', 'word', 'sentence', 'language', 'all').
    Returns:
        str: The perturbed question.
    """
    if level == "none":
        return question  # No perturbation applied
    elif level == "char":
        return char_level(question)
    elif level == "word":
        return word_level(question)
    elif level == "sentence":
        return sentence_level(question)
    elif level == "language":
        return language_level(question)
    elif level == "all":
        return all_levels(question)
    else:
        raise ValueError("Invalid perturbation level specified.")

def char_level(question: str) -> str:
    """
    Character-level perturbation function.
    Args:
        question (dict): The question to perturb.
    Returns:
        str: The perturbed question with character-level changes.
    """
    # Example implementation: replace characters randomly
    return question + " (p1: " + str(random.random()) + ")"

def word_level(question: str) -> str:
    """
    Word-level perturbation function.
    Args:
        question (dict): The question to perturb.
    Returns:
        str: The perturbed question with word-level changes.
    """
    return question + " (p2: " + str(random.random()) + ")"

def sentence_level(question: str) -> str:
    """
    Sentence-level perturbation function.
    Args:
        question (dict): The question to perturb.
    Returns:
        str: The perturbed question with sentence-level changes.
    """
    return question + " (p3: " + str(random.random()) + ")"

def language_level(question: str) -> str:
    """
    Language-level perturbation function.
    Args:
        question (dict): The question to perturb.
    Returns:
        str: The perturbed question with language-level changes.
    """
    # Example implementation: translate to another language and back
    return question + " (p4: " + str(random.random()) + ")"

def all_levels(question: str) -> str:
    """
    Applies all perturbation levels to the question.
    Args:
        question (str): The question to perturb.
    Returns:
        str: The perturbed question with all levels of changes applied.
    """
    perturbed = char_level(question)
    perturbed = word_level(perturbed)
    perturbed = sentence_level(perturbed)
    perturbed = language_level(perturbed)
    return perturbed