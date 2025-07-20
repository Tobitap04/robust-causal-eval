from services.llm_service import LLMService
import typo
import random


def perturbation_func(question: str, level: str, intensity: int, llm_service: LLMService = None) -> str:
    """
    Perturbation function that applies a specified level of perturbation to the question.
    Args:
        question (str): The question to perturb.
        level (str): The level of perturbation to apply ('char', 'synonym', 'language', 'paraphrase', 'sentence_inj', 'bias').
        intensity (int): The intensity of the perturbation: 25, 50, 75, or 100.
        llm_service (LLMService): An instance of the LLM service used for generating perturbations.
    Returns:
        str: The perturbed question.
    Raises:
        ValueError: If an invalid perturbation technique is specified.
    """
    if intensity not in [25, 50, 75, 100, None]:
        raise ValueError("Intensity must be one of the following values: 25, 50, 75, 100 or None (default value).")
    if level == "char":
        return char_level(question, intensity)
    elif level == "synonym":
        return synonym_level(question, intensity, llm_service)
    elif level == "language":
        return language_level(question, intensity, llm_service)
    elif level == "paraphrase":
        return paraphrase_level(question, intensity, llm_service)
    elif level == "sentence_inj":
        return sentence_injection_level(question, intensity, llm_service)
    elif level == "bias":
        return bias_level(question, intensity, llm_service)
    else:
        raise ValueError(f"Invalid perturbation type specified: {level}.")

# TODO: Set default intensity
def char_level(question: str, intensity: int) -> str:
    """
    Character-level perturbation function.
    Args:
        question (str): The question to perturb.
        intensity (int): The percentage of words to perturb.
    Returns:
        str: The perturbed question with character-level changes.
    """

    typo_methods = [
        "char_swap",
        "missing_char",
        "extra_char",
        "nearby_char",
        "similar_char",
        "skipped_space",
        "random_space",
        "repeated_char",
        "unichar",
        "casing",
        "punctuation"
    ]

    def punctuation(text: str) -> str:
        """
        Randomly replaces a punctuation mark in the text with another one.
        Args:
            text (str): The text to perturb.
        Returns:
            str: The perturbed text with a punctuation mark replaced.
        """
        punctuation_marks = list(".,!?;:-")
        punctuation_indices = [i for i, c in enumerate(text) if c in punctuation_marks]

        # No punctuation marks found, add a random one at the end
        if not punctuation_indices:
            return text + random.choice(list(".!?"))

        index_to_replace = random.choice(punctuation_indices)
        current_char = text[index_to_replace]

        # Replace with a different punctuation mark
        if current_char in list(",;:-"):
            replacement = random.choice([p for p in list(",;:- ") if p != current_char])
        else:
            replacement = random.choice([p for p in list("!?. ") if p != current_char])

        return text[:index_to_replace] + replacement + text[index_to_replace + 1:]

    def casing(text: str) -> str:
        """
        Randomly flips the case of a letter in the text.
        Args:
            text (str): The text to perturb.
        Returns:
            str: The perturbed text with a letter's case flipped.
        """
        letter_indices = [i for i, c in enumerate(text) if c.isalpha()]

        # If no letters found, return the original text
        if not letter_indices:
            return text

        # Replace a random letter with its opposite case
        index = random.choice(letter_indices)
        char = text[index]
        flipped_char = char.lower() if char.isupper() else char.upper()

        return text[:index] + flipped_char + text[index + 1:]

    number_perturbs = max(int(len(question.split()) * (intensity/100)), 1) # Ensure at least one perturbation
    selected_perturbs = random.choices(typo_methods, k=number_perturbs)

    for method in selected_perturbs:
        before = question
        if method not in ["casing", "punctuation"]:
            strErrer = typo.StrErrer(question)
            if method == "char_swap":
                strErrer = strErrer.char_swap()
            elif method == "missing_char":
                strErrer = strErrer.missing_char()
            elif method == "extra_char":
                strErrer = strErrer.extra_char()
            elif method == "nearby_char":
                strErrer = strErrer.nearby_char()
            elif method == "similar_char":
                strErrer = strErrer.similar_char()
            elif method == "skipped_space":
                strErrer = strErrer.skipped_space()
            elif method == "random_space":
                strErrer = strErrer.random_space()
            elif method == "repeated_char":
                strErrer = strErrer.repeated_char()
            elif method == "unichar":
                strErrer = strErrer.unichar()
            question = strErrer.result
        else:
            if method == "casing":
                question = casing(question)
            elif method == "punctuation":
                question = punctuation(question)

        # Fallback if perturbation does not change the question
        after = question
        if before == after:
            if method in typo_methods:
                typo_methods.remove(method)
            if len(typo_methods) == 0:
                break
            selected_perturbs.append(random.choice(typo_methods))

    return question

def synonym_level(question: str, intensity: int, llm_service: LLMService) -> str:
    """
    Synonym-level perturbation function.
    Args:
        question (str): The question to perturb.
        intensity (int): The percentage of words to perturb.
        llm_service (LLMService): An instance of the LLM service used for generating synonyms.
    Returns:
        str: The perturbed question with synonym-level changes.
    """
    word_count = len(question.split())
    num_words = max(1, int((word_count * intensity) / 100))
    remaining_words = min(word_count - 1, word_count - num_words)

    prompt = (
        "Given the following text:\n"
        f"{question}\n"
        f"{instructions['synonym_level'].format(num_words=num_words, intensity=intensity, remaining_words=remaining_words)}"
        f"Example with replacing {max(1, int((10 * intensity)/100))} words:\n"
        f"Question: {examples['question']}\n"
        f"<result>{examples['synonym_level'][intensity]}</result>"
    )

    return filter_result(llm_service.get_llm_response(prompt))

def language_level(question: str, intensity: int, llm_service: LLMService) -> str:
    """
    Language-level perturbation function.
    Args:
        question (str): The question to perturb.
        intensity (int): The percentage of words to perturb.
        llm_service (LLMService): An instance of the LLM service used for generating language-level perturbations.
    Returns:
        str: The perturbed question with language-level changes.
    """
    languages = ["German", "French", "Spanish", "Italian", "Portuguese"]
    word_count = len(question.split())
    num_words = max(1, int((word_count * intensity) / 100))
    remaining_words = min(word_count - 1, word_count - num_words)
    language = random.choice(languages)

    prompt = (
        "Given the following text:\n"
        f"{question}\n"
        f"{instructions['language_level'].format(num_words=num_words, intensity=intensity, remaining_words=remaining_words, language=language)}"
        f"Example with translating {max(1, int((10 * intensity) / 100))} words:\n"
        f"Question: {examples['question']}\n"
        f"<result>{examples['language_level'][intensity]}</result>"
    )

    return filter_result(llm_service.get_llm_response(prompt))

def paraphrase_level(question: str, intensity: int, llm_service: LLMService) -> str:
    """
    Paraphrase-level perturbation function.
    Args:
        question (str): The question to perturb.
        intensity (int): The percentage of words to perturb.
        llm_service (LLMService): An instance of the LLM service used for generating paraphrases.
    Returns:
        str: The paraphrased question.
    """
    prompt = (
        "Given the following text:\n"
        f"{question}\n"
        f"Instruction: {instructions['paraphrase_level'][intensity]} Enclose the result in the tags as demonstrated in the example.\n"
        f"Example:\n"
        f"Question: {examples['question']}\n"
        f"<result>{examples['paraphrase_level'][intensity]}</result>"
    )

    return filter_result(llm_service.get_llm_response(prompt))

def sentence_injection_level(question: str, intensity: int, llm_service: LLMService) -> str:
    """
    Injects additional sentences and phrases to the question.
    Args:
        question (str): The question to perturb.
        intensity (int): The intensity of perturb.
        llm_service (LLMService): An instance of the LLM service used for generating extra information.
    Returns:
        str: The question after injecting additional a sentence and/or phrase as perturbations.
    """
    prompt = (
        "Given the following text:\n"
        f"{question}\n"
        f"Instruction: {instructions['sentence_injection_level'][intensity]} Enclose the result in the tags as demonstrated in the example.\n"
        f"Example:\n"
        f"Question: {examples['question']}\n"
        f"<result>{examples['sentence_injection_level'][intensity]}</result>"
    )

    return filter_result(llm_service.get_llm_response(prompt))

def bias_level(question: str, intensity: int, llm_service: LLMService) -> str:
    """
    Rewrites the question to steer the answer in the wrong direction.
    Args:
        question (str): The question to perturb.
        intensity (int): The intensity of perturb.
        llm_service (LLMService): An instance of the LLM service used for generating extra information.
    Returns:
        str: The biased question.
    """
    prompt = (
        "Given the following text:\n"
        f"{question}\n"
        f"Instruction: {instructions['bias_level'][intensity]} The original question must remain part of the text, either in its exact wording or as a paraphrase. No additional questions should be introduced. Enclose the result in the tags as demonstrated in the example.\n"
        f"Example:\n"
        f"Question: {examples['question']}\n"
        f"<result>{examples['bias_level'][intensity]}</result>"
    )

    return filter_result(llm_service.get_llm_response(prompt))

def filter_result(result: str) -> str:
    """
    Filters the result from the LLM to ensure it is in the correct format.
    Args:
        result (str): The raw result from the LLM.
    Returns:
        str: The filtered result, ensuring it is enclosed in <result> tags.

    Raises:
        ValueError: If the result does not contain the expected <result> tags.
    """
    if "<result>" in result and "</result>" in result:
        return result.split("<result>")[1].split("</result>")[0].strip()
    else:
        raise ValueError("The LLM response does not contain the expected <result> tags.")

instructions = {
    "synonym_level": (
        "Instruction: Replace {num_words} of the words ({intensity}%) in the following text "
        "with a contextually appropriate synonym that preserves the original meaning and "
        "does not make the wording more general or less specific. The rest of the text "
        "({remaining_words} words) should remain exactly as it is. Enclose the result in "
        "the tags as demonstrated in the example.\n"
    ),
    "language_level": (
        "Instruction: Translate {num_words} of the words ({intensity}%) in the following "
        "text into {language}. Ensure that the overall meaning is preserved and that all "
        "remaining text ({remaining_words} words) stay unchanged in English. Enclose the "
        "result in the tags as demonstrated in the example.\n"
    ),
    "paraphrase_level": {
        25: (
            "Paraphrase the following text with light changes. Keep the original meaning "
            "and structure as close as possible."
        ),
        50: (
            "Rewrite the text with moderate changes in wording and phrasing while "
            "preserving the original meaning."
        ),
        75: (
            "Paraphrase the text by making strong alterations to its structure and "
            "vocabulary, but keep the core meaning intact."
        ),
        100: (
            "Completely rephrase the text with very strong changes in structure and "
            "wording, maintaining the original idea."
        ),
    },
    "sentence_injection_level": {
        25: (
            "Insert a short, contextually related phrase into the text that is "
            "irrelevant for answering the question. The phrase must not introduce new "
            "meaning or a question. Ensure that the question is neither made more "
            "specific nor more general, and that the addition do not implicitly answer it."
        ),
        50: (
            "Insert a short, contextually related phrase into the text that is "
            "irrelevant for answering the question. Then, add one short, standalone "
            "sentence that is also contextually related but irrelevant for answering the "
            "question. Both the phrase and the sentence must not introduce new meaning or "
            "a question. Ensure that the question is neither made more specific nor more "
            "general, and that the additions do not implicitly answer it."
        ),
        75: (
            "Insert a contextually related phrase into the text that is irrelevant for "
            "answering the question. Then, add one short, standalone sentence that "
            "contains unnecessary but related information. Both the phrase and the "
            "sentence must not introduce new meaning or a question. Ensure that the "
            "question is neither made more specific nor more general, and that the "
            "additions do not implicitly answer it."
        ),
        100: (
            "Insert a contextually related phrase into the text that is irrelevant for "
            "answering the question. Then, add one standalone sentence that contains "
            "unnecessary but related information. Both the phrase and the sentence must "
            "not introduce new meaning or a question. Ensure that the question is neither "
            "made more specific nor more general, and that the additions do not "
            "implicitly answer it."
        ),
    },
    "bias_level": {
        25: (
            "Rewrite the following text while keeping its original meaning mostly "
            "intact, but introduce a slight bias that could mislead the model toward a "
            "less accurate answer. Use subtle shifts in tone or emphasis, such as "
            "downplaying key concepts or inserting mild doubt. The changes should be "
            "minimal but capable of affecting how the question is interpreted."
        ),
        50: (
            "Rewrite the following text in a way that preserves its general structure, "
            "but introduces a moderate bias by subtly shifting the focus toward "
            "alternative explanations or less relevant factors. Use selective framing or "
            "emotionally suggestive language to guide the model away from the most "
            "accurate or direct answer."
        ),
        75: (
            "Rewrite the following text to maintain its core meaning, but strongly "
            "bias it by introducing misleading framing, emotionally loaded wording, or "
            "alternative explanations that draw attention away from the correct answer. "
            "The phrasing should clearly steer the model toward a less accurate or "
            "secondary response, while sounding plausible."
        ),
        100: (
            "Completely rewrite the following text using strong bias in framing, tone, "
            "and content to redirect the model’s attention away from the main issue. "
            "Emphasize unrelated causes, cast doubt on well-established facts, or shift "
            "the question’s perspective so that the model is likely to provide a "
            "misleading, inaccurate, or incomplete response. The question should still "
            "appear logically valid and contextually related."
        ),
    },
}
examples = {
    "question": (
        "Does regular smoking increase the risk of developing lung cancer?"
    ),
    "synonym_level": {
        25: "Does habitual smoking increase the risk of getting lung cancer?",
        50: "Can habitual smoking raise the risk of getting pulmonary cancer?",
        75: "Can habitual tobacco use raise the chance of getting pulmonary cancer?",
        100: "Can habitual tobacco use raise one's chance for getting pulmonary carcinoma?",
    },
    "language_level": {
        25: "Does gewöhnliches smoking increase das risk of developing lung cancer?",
        50: "Does gewöhnliches smoking erhöhen das risk of entwickeln lung Krebs?",
        75: "Kann gewöhnliches smoking erhöhen das risk zu entwickeln lung Krebs?",
        100: "Kann gewöhnliches Rauchen das Risiko erhöhen Lungenkrebs zu entwickeln?",
    },
    "paraphrase_level": {
        25: "Does habitual smoking increase the likelihood of developing lung cancer?",
        50: "Can smoking on a regular basis raise the chance of getting lung cancer?",
        75: "Is there a higher risk of lung cancer associated with consistent tobacco use?",
        100: (
            "To what extent does frequent tobacco consumption contribute to the "
            "development of cancer in the lungs?"
        ),
    },
    "sentence_injection_level": {
        25: (
            "Does regular smoking, as many individuals do, increase the risk of developing "
            "lung cancer?"
        ),
        50: (
            "Does regular smoking, as many individuals do, increase the risk of developing "
            "lung cancer? I'm really interested in the answers to this question."
        ),
        75: (
            "Does regular smoking, as many individuals routinely do without much thought, "
            "increase the risk of developing lung cancer? I'm really interested in the "
            "answers to this question."
        ),
        100: (
            "Does regular smoking, as many individuals routinely do without much thought, "
            "increase the risk of developing lung cancer? I'm really interested in the "
            "answers to this question as it affects many people."
        ),
    },
    "bias_level": {
        25: (
            "Does regular smoking, which some people tolerate for decades without major "
            "health problems, really increase the risk of developing lung cancer?"
        ),
        50: (
            "Although other factors like air pollution and genetics are known to play a "
            "role, does regular smoking still increase the risk of developing lung cancer "
            "to the extent that’s often claimed?"
        ),
        75: (
            "Considering how many lifelong smokers never develop the disease and how "
            "complex the causes of lung cancer are, does regular smoking actually "
            "increase the risk of developing it in a significant way?"
        ),
        100: (
            "Given that lung cancer can also occur in non-smokers and that smoking is "
            "just one of many possible contributing factors, does regular smoking truly "
            "increase the risk of developing lung cancer—or is that an oversimplified "
            "narrative?"
        ),
    },
}