import random

def build_prompt(question: str, answer: str, option: str) -> str:
    """
    Builds a prompt for the LLM to classify a question.
    Args:
        question (str): The question to be analyzed.
        answer (str, optional): The answer to the question.
        option (str): The prompting function to apply ('overall', ).
    Returns:
        str: The constructed prompt for the LLM.
    Raises:
        ValueError: If prompting option is not valid.
    """
    if option == "overall":
        return overall_filter_v1(question)
    elif option == "causal_chain":
        return causal_chain_filter(question, answer)
    else:
        raise ValueError(f"Invalid filter specified: {option}.")


def overall_filter_v1(question: str) -> str:
    """Old version of the overall filter prompt. Now deprecated."""
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

def causal_chain_filter(question: str, answer: str) -> str:
    """
    Constructs a prompt that evaluates whether a given question-answer pair contains a causal chain of events.
    """

    instruction = (
        "Task:\n"
        "Evaluate the following question-answer pair to determine whether it contains a causal chain of events.\n\n"
        "Instructions:\n"
        "Step 0: If the question follows the structure “(Can / Could / Does / Might) A (cause / lead to / result in / affect) B?” "
        "or a similar phrasing that clearly asks whether A causes B, immediately output: <result>1</result>.\n"
        "Step 1: Otherwise, identify the main cause and main effect that reflect the core focus of the question. "
        "If no clear cause-effect relationship can be identified, no valid causal chain can be constructed.\n"
        "Step 2: Attempt to construct a causal chain between the identified cause and effect in the form: Cause → … → Effect, "
        "where each link A → B represents a direct causal influence (not merely correlation or a temporal sequence). "
        "Intermediate steps do not need to be explicitly stated in the question or answer.\n"
        "Step 3: For each link A → B in the chain, verify that A directly causes B. "
        "If all links fulfill this condition, output: <result>1</result>. "
        "If no valid causal chain can be constructed, briefly explain why and output: <result>0</result>.\n\n"
    )

    few_shot_examples = [
        {
            "question": "Why does my ankle pop?",
            "answer": "When you move your ankle, you stretch the joint capsule, which contains lubricating fluid. "
                      "This change in pressure can cause gas bubbles to form or be released, and when these bubbles collapse, a popping sound may occur.",
            "evaluation": (
                "Step 0: The question does not follow the structure from step 0.\n"
                "Step 1: Cause: Movement of the ankle, Effect: Popping sound of ankle\n"
                "Step 2: A causal chain can be constructed: Move ankle → Stretch joint capsule → Drop in joint pressure → "
                "Gas bubbles form or are released → Bubbles collapse → Popping sound of ankle.\n"
                "Step 3: Each step in the chain represents a direct causal influence. The ankle movement causes the capsule to stretch, "
                "which leads to a drop in pressure. This pressure change causes gas bubbles to form or be released. "
                "The collapse of these bubbles directly results in the popping sound. All steps satisfy the criteria.\n"
                "<result>1</result>"
            )
        },
        {
            "question": "When did the second season of 13 Reasons Why come out?",
            "answer": "May 18, 2018",
            "evaluation": (
                "Step 0: The question does not follow the structure from step 0.\n"
                "Step 1: Cause: None identifiable, Effect: None identifiable\n"
                "Step 2: No causal chain can be constructed.\n"
                "Step 3: This is a factual question about a release date. It contains no explanation of why the event occurred or what it caused. "
                "Therefore, no causal relation can be derived.\n"
                "<result>0</result>"
            )
        },
        {
            "question": "Does birth control pills affect your liver?",
            "answer": "Estrogens and oral contraceptives are both associated with several liver-related complications including intrahepatic cholestasis, "
                      "sinusoidal dilatation, peliosis hepatis, hepatic adenomas, hepatocellular carcinoma, hepatic venous thrombosis, "
                      "and an increased risk of gallstones.",
            "evaluation": (
                "Step 0: The question follows the structure “Does A (birth control pills) affect B (your liver)?”, "
                "which clearly asks whether A causes B. Since this matches the predefined causal question format, no further evaluation is necessary.\n"
                "<result>1</result>"
            )
        },
    ]

    # Build the few-shot section
    example_block = ""
    for idx, ex in enumerate(few_shot_examples, 1):
        example_block += (
            f"Example {idx}:\n"
            f"Question: {ex['question']}\n"
            f"Answer: {ex['answer']}\n"
            f"{ex['evaluation']}\n\n"
        )

    # Append user input
    full_prompt = (
        instruction
        + example_block
        + f"Now analyze the following pair in the same way:\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
    )

    return full_prompt