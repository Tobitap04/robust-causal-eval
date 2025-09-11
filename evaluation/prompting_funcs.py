import random
import re

from services.llm_service import LLMService

dataset_answer_lengths = {'eli5': int(99), 'gooaq': int(44.3), 'msmarco': int(17.5), 'naturalquestions': int(10.8),
                          'squad2': int(6.2)}


def processing_func(question: str, preproc: str, inproc: str, postproc: str, dataset: str, llm_service: LLMService,
                    temp: float) -> str:
    """
    This function handels the pre-, in- and postprocessing of the question.
    Args:
        question (str): The (perturbed) question
        preproc (str): The preprocessing function
        inproc (str): The inprocessing function
        postproc (str): The postprocessing function
        dataset (str): The dataset from which the question originates.
        llm_service (LLMService): An instance of the LLM service used to generate the response.
        temp (float): Temperature setting controlling the randomness of the LLM output.
    Returns:
        str: The preprocessed question ready for prompting.
    Raises:
        ValueError: If an invalid preprocessing type is specified.
    """
    # Preprocessing
    if preproc == "none":
        pass
    elif preproc == "translate":
        prompt = (
            "If the question is not fully in English, translate it into English while preserving the original wording "
            "as closely as possible. "
            "If it is already entirely in English, leave it unchanged. Output only the final text inside "
            "<result> and </result> tags."
            f"\n\nQuestion: {question}"
        )
        question = filter_result(llm_service.get_llm_response(prompt))
    elif preproc == "filter":
        prompt = (
            "Remove all biased or irrelevant information from the question, including any details that are not "
            "essential to understanding or answering it. "
            "Preserve the core meaning and the essential question exactly. Output only the final text inside "
            "<result> and </result> tags."
            f"\n\nQuestion: {question}"
        )
        question = filter_result(llm_service.get_llm_response(prompt))
    elif preproc == "correct":
        prompt = (
            "Correct all spelling mistakes in the text, including letter swaps, missing letters, extra letters, "
            "or incorrect capitalization. "
            "If a word is unclear, infer the most likely intended word based on context. "
            "Also standardize capitalization and ensure punctuation is meaningful and contextually appropriate. "
            "Do not change word order, wording, or phrasing except as required to fix spelling, capitalization, "
            "and punctuation. "
            "Do not add any new words that were not present in the original text. "
            "Output only the corrected text inside <result> and </result> tags."
            f"\n\nQuestion: {question}"
        )
        question = filter_result(llm_service.get_llm_response(prompt))
    else:
        raise ValueError(f"Invalid preprocessing type specified: {preproc}.")

    # Inprocessing
    if inproc == "none":
        pass
    elif inproc == "translate":
        question = question + ("\nIf the question is not fully in English, first translate it into English while preserving the original wording as "
                               "closely as possible, then answer the translated question. Any constraints given apply only "
                               "to the final answer. Place the final answer within <result> and </result> tags.")
    elif inproc == "cot":
        question = question + ("\nLet's think step by step. Any constraints given apply only to the final answer, not to the reasoning steps. "
                               "Place the final answer within <result> and </result> tags.")
    elif inproc == "subproblems":
        question = question + (
            "\nYou are an expert in causal reasoning. "
            "Your task is to solve the causal question by breaking it down into logical subproblems and reasoning through each one step by step. "
            "Follow these steps:\n"
            "1. Decompose the main question into distinct subproblems, identifying possible causes, mechanisms, and potential effects.\n"
            "2. For each subproblem, provide a concise, evidence-based reasoning path, clearly explaining the logic behind your conclusions.\n"
            "3. Once all subproblems are analyzed, extract the statements or mechanisms that are most consistently supported across your reasoning.\n"
            "4. Based on these consistent insights, generate a final, consolidated, factual answer and enclose it "
            "within <result> and </result> tags. Any constraints provided apply only to the final answer, not to the "
            "intermediate reasoning."
        )
    elif bool(re.fullmatch(r'few_shot[0-9]', inproc)):
        question = question + few_shot(int(inproc[-1]))
    elif inproc == "few_shot_gooaq":
        question = question + few_shot_qooaq()
    elif inproc == "robust":
        question = question + (
            "\nYou are an expert in robust causal question answering. Provide a clear, consistent, and coherent answer "
            "that remains robust to variations in the wording or phrasing of the question."
        )
    else:
        raise ValueError(f"Invalid inprocessing type specified: {inproc}.")

    # Postprocessing
    if postproc == "none":
        pass
    elif postproc == "length":
        question = f"Constraint: Answer the question using {dataset_answer_lengths[dataset]} words.\nQuestion: {question}"
    elif postproc == "list1":
        question = (f"Constraint: Output only a comma-separated list of causes or effects in the format A, B, C, … "
                    f"For binary questions, output only ‘yes’ or ‘no’ (followed by a list of causes or "
                    f"effects for explanation)."
                    f"No additional text.\nQuestion: {question}")
    elif postproc == "list2":
        question = (f"Constraint: Output only a comma-separated list of causes or effects in the format A, B, C, … "
                    f"For binary questions, output only ‘yes’ or ‘no’ (followed by a list of causes or "
                    f"effects for explanation)."
                    f"Do not list any cause or effect more than once and add no additional text.\nQuestion: {question}")
    elif postproc == "self_consistency":
        all_answers = [
            filter_result(llm_service.get_llm_response(question, 1)) # Use temperature 1 for diversity
            for _ in range(3)
        ]
        question = (
            "Below are three answers generated independently. "
            "Identify the statements or ideas that recur most frequently across them. "
            "Using only those recurring statements, write a final, consolidated answer "
            "that is clear, coherent, and consistent.\n\n"
        )
        answers_text = "\n\n".join([f"Answer {i + 1}:\n{ans}" for i, ans in enumerate(all_answers)])
        question += answers_text + "\n\nPlease provide the final consolidated answer in <result> and </result> tags."
    else:
        raise ValueError(f"Invalid postprocessing type specified: {postproc}.")

    return filter_result(llm_service.get_llm_response(question, temp))


def few_shot(n: int) -> str:
    """
    Applies few-shot prompting by appending examples to the input prompt. Contains examples of all datasets.
    Args:
        n (int): The number of examples to apply.
    Returns:
        str: The generated response from the LLM after applying few-shot prompting.
    """
    instruction = "\n\nHere are some examples of questions and their answers:"
    end = "Now answer the given question based on the examples and constraints provided."
    examples = [
        "Question: Why do human arms sway naturally back and forth as we walk? \n"
        "Answer: It makes us more energy efficient as two legged animals. Makes it a lot easier to walk. In fact, I read somewhere that purposely not swinging your arms is similar to wearing a 70lbs pack. "
        "Obviously that number would be different for everyone since we are not all the same size, shape, and weight.",

        "Question: why is magnesium good for muscles?\n"
        "Answer: magnesium is absolutely necessary for proper muscle function. it works with other essential minerals in your body to keep the muscles loose and flexible. "
        "when you exercise or do some kind of physical activity, magnesium relaxes your muscles and controls their contractions.",

        "Question: chemical and biological causes for borderline personality disorder\n"
        "Answer: mood, thinking, behavior, personal relations, and self-image.",

        "Question: why did the us navy stop using battleships?\n"
        "Answer: the increasing importance of the aircraft carrier",

        "Question: why were foxes originally hunted?\n"
        "Answer: form of vermin control to protect livestock",

        "Question: How come that many people get creative thoughts when they try to sleep?\n"
        "Answer: When you're falling asleep, your brain receives fewer external sensory inputs (like sounds or sights), so it's free to wander and make new connections—similar to when you're in the shower.",

        "Question: Can waist trainers cause back pain?\n"
        "Answer: Yes, wearing waist trainers can weaken your back and core muscles over time, since the corset supports your posture instead of your muscles doing the job.",

        "Question: What is caused by eating red meat?\n"
        "Answer: Regularly eating large amounts of red or processed meat has been linked to an increased risk of certain cancers, like colorectal cancer.",

        "Question: Why are some cats born with extra toes?\n"
        "Answer: It's caused by a harmless genetic condition called polydactyly, which results in more than the usual number of toes on a cat’s paws."
    ]
    selected_examples = random.sample(examples, n)
    return f"{instruction}\n" + "\n".join(selected_examples) + f"\n\n{end}"


def few_shot_qooaq() -> str:
    """
    Applies few-shot prompting specifically for the Gooaq dataset. Contains only examples from the Gooaq dataset.
    Returns:
        str: The generated response from the LLM after applying few-shot prompting.
    """
    instruction = "\n\nHere are some examples of questions and their answers:"
    end = "Now answer the given question based on the examples and constraints provided."
    examples = [
        "Question: do birth control pills cause weight gain?\n"
        "Answer: it's often a temporary side effect that's due to fluid retention, not extra fat. a review of 44 studies showed no evidence that birth control pills caused weight gain in most women. "
        "and, as with other possible side effects of the pill, any weight gain is generally minimal and goes away within 2 to 3 months.",

        "Question: why amsterdam is so expensive?\n"
        "Answer: the main thing that's expensive in amsterdam is the accommodation - the rest is average/on par with the rest of western europe and much more affordable than london. "
        "the reason accommodation is so expensive is simply economics: high demand (year-round) and very limited quantity.",

        "Question: why is my phone youtube so slow?\n"
        "Answer: the reason for your slow youtube experience is most likely your internet connection. ... this means if your connection is spotty or intermittent, "
        "you will have a poor youtube experience. your device isn't able to get the data packets from the server faster enough to give you a smooth video streaming experience.",

        "Question: what happens to your body if you go vegan?\n"
        "Answer: within a few months, a well-balanced vegan diet which is low in salt and processed food may have impressive benefits for cardiovascular health, "
        "helping to prevent heart disease, stroke and reducing the risk of diabetes.",

        "Question: what is the major cause of greenhouse effect?\n"
        "Answer: greenhouse effect, a warming of earth's surface and troposphere (the lowest layer of the atmosphere) caused by the presence of water vapour, carbon dioxide, methane, and certain other gases in the air. "
        "of those gases, known as greenhouse gases, water vapour has the largest effect."
    ]
    return f"{instruction}\n" + "\n".join(examples) + f"\n\n{end}"


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
        return result.strip()
prompt = (
            "Below are several answers generated independently. "
            "Identify the statements or ideas that recur most frequently across them. "
            "Using only those recurring statements, write a final, consolidated answer "
            "that is clear, coherent, and consistent.\n\n"
        )


print(prompt)