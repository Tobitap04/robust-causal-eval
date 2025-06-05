import random

def build_prompt(question: str, option: str) -> str:
    """
    Builds a prompt for the LLM to classify a question.
    Args:
        question (str): The question to be analyzed.
        option (str): The prompting function to apply ('overall', ).
    Returns:
        str: The constructed prompt for the LLM.
    """
    if option == "overall":
        return overall_filter_v1(question)
    else:
        raise ValueError("Invalid prompting type specified.")


def overall_filter_v1(question: str) -> str:
    # TODO: Rework and more examples
    instruction = (
        "You are an expert at identifying whether a question is truly causal, has a clear reference, and is not nonsensical.\n"
        "A causal question seeks to identify the cause or reason behind an event or phenomenon.\n"
        "It should involve a clear causal relationship or chain of events, not merely an explanation of purpose, intention, effect, definition, technical mechanisms, or historical facts without a causal link.\n"
        "For each input, answer only with \"1\" (if the question is causal, clear, and sensible) or \"0\" (otherwise) without any additional symbols before or after.\n"
        "Do not provide any explanation or further examples.\n\n"
    )

    few_shot_examples = [
        {"input": "what is the cause of dry mouth while sleeping", "output": "1"}, # Clear causal question
        {"input": "Why was he banned?", "output": "0"},  # Unclear reference
        {"input": "direct cause definition", "output": "0"},  # Definition
        {"input": "can beer cause liver damage", "output": "1"}, # Clear causal question
        {"input": "Why is einsteinium?", "output": "0"},  # Nonsensical question
        {"input": "why was plant taxonomy developed?", "output": "0"},  # Purpose
        {"input": "what is the most common cause of diarrhea?", "output": "1"},  # Clear causal question
        {"input": "why was the united states concerned about nuclear missiles in cuba?", "output": "0"}, # Intention
        {"input": "how did she die after she returned from a trip to england to sell her jewels in 1793?", "output": "0"}, # Historical fact without causal mechanism and unclear reference
        #{"input": "why is a voltmeter needed in a circuit?", "output": "0"},
    ]

    # Construct the prompt with instruction and shuffled examples
    random.shuffle(few_shot_examples)
    examples = ""
    for idx, ex in enumerate(few_shot_examples, 1):
        examples += f"Example {idx}:\nInput:{ex['input']}\nOutput:{ex['output']}\n\n"
    prompt = (
            instruction
            + examples
            + f"Now analyze only the following question in the same way:\nInput:{question}\nOutput:"
    )
    return prompt