import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw

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
        raise ValueError(f"Invalid perturbation level specified: {level}.")


def char_level(question: str) -> str:
    """
    Character-level perturbation function.
    Args:
        question (str): The question to perturb.
    Returns:
        str: The perturbed question with character-level changes.
    """
    # TODO: Refine the character-level perturbation (Punctuation, capitalization, etc.)
    # nlplaug implementation
    aug0 = naw.SpellingAug(aug_p=0.15)
    perturbed_question = aug0.augment(question)[0]
    aug1 = nac.KeyboardAug(aug_char_p=0.15, aug_word_p=0.15)
    perturbed_question = aug1.augment(perturbed_question)[0]
    aug2 = naw.SplitAug(aug_p=0.15)
    perturbed_question = aug2.augment(perturbed_question)[0]
    return perturbed_question


def word_level(question: str) -> str:
    """
    Word-level perturbation function.
    Args:
        question (str): The question to perturb.
    Returns:
        str: The perturbed question with word-level changes.
    """
    # TODO: Refine word-level perturbation (Better synonyms, antonyms, etc.)
    # nlplaug implementation
    aug = naw.SynonymAug(aug_src='wordnet', lang='eng')
    perturbed_question = aug.augment(question)[0]
    return perturbed_question


def sentence_level(question: str) -> str:
    """
    Sentence-level perturbation function.
    Args:
        question (dict): The question to perturb.
    Returns:
        str: The perturbed question with sentence-level changes.
    """
    pass  # TODO: Implement sentence-level perturbation

def language_level(question: str) -> str:
    """
    Language-level perturbation function.
    Args:
        question (str): The question to perturb.
    Returns:
        str: The perturbed question with language-level changes.
    """
    # Example implementation: translate to another language and back
    pass  # TODO: Implement language-level perturbation


def all_levels(question: str) -> str:
    """
    Applies all perturbation levels to the question.
    Args:
        question (str): The question to perturb.
    Returns:
        str: The perturbed question with all levels of changes applied.
    """
    # perturbed = sentence_level(question)
    perturbed = word_level(question)
    # perturbed = language_level(perturbed)
    perturbed = char_level(perturbed)

    return perturbed