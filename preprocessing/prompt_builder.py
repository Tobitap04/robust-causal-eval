import random

def build_prompt(question: str, answer: str, option: str) -> str:
    """
    Builds a prompt for the LLM to classify a question.
    Args:
        question (str): The question to be analyzed.
        answer (str, optional): The answer to the question.
        option (str): The prompting function to apply ('causal_chain', 'answer', 'question').
    Returns:
        str: The constructed prompt for the LLM.
    Raises:
        ValueError: If prompting option is not valid.
    """

    if option == "causal_chain":
        return causal_chain_filter(question, answer)
    elif option == "answer":
        return answer_filter(question, answer)
    elif option == "question":
        return question_filter(question)
    else:
        raise ValueError(f"Invalid filter specified: {option}.")

def causal_chain_filter(question: str, answer: str) -> str:
    """
    Constructs a prompt that evaluates whether a given question-answer pair contains a causal chain of events.
    Args:
        question (str): The question to be analyzed.
        answer (str): The answer to the question.
    Returns:
        str: The constructed prompt for the LLM.
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

    # Append example to analyse
    full_prompt = (
        instruction
        + example_block
        + f"Now analyze the following pair in the same way:\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
    )

    return full_prompt

def answer_filter(question: str, answer: str) -> str:
    """
    Constructs a prompt that evaluates whether the answer of a given question-answer pair is appropriate in content and form.
    Args:
        question (str): The question to be analyzed.
        answer (str): The answer to the question.
    Returns:
        str: The constructed prompt for the LLM.
    """

    instruction = (
        "Task:\n"
        "Evaluate the following question–answer pair to determine whether the answer is appropriate in content and form.\n\n"
        "Instructions:\n"
        "Assess the answer based on the following criteria:\n"
        "\t1.\tRelevance: Does the answer directly and fully address the question?\n"
        "\t2.\tObjectivity: Is the answer factual and neutral, free from personal opinions or emotional language?\n"
        "\t3.\tClarity: Is the answer clear, precise, and free from rambling or vagueness?\n"
        "\t4.\tFormatting: Is the answer cleanly presented, free from formatting like brackets, hyperlinks, or edit notes?\n"
        "If any of these criteria are not met, briefly explain why and output: <result>0</result>\n"
        "If the answer meets all criteria and fits the question well, output: <result>1</result>\n\n"
    )

    few_shot_examples = [
        {
            "question": "What are the side effects of losartan potassium 50mg?",
            "answer": "[‘diarrhea.’, ‘stomach pain.’, ‘muscle cramps.’, ‘leg or back pain.’, ‘dizziness.’, ‘headache.’, ‘sleep problems (insomnia)’, ‘tiredness, and.’]",
            "reason": "Poor formatting (brackets, Python-style list, incomplete final item).",
            "result": "<result>0</result>"
        },
        {
            "question": "what causes eyelid to droop",
            "answer": "a stroke, brain tumor, or cancer of the nerves or muscles.",
            "reason": "Factual, concise, and clearly answers the question.",
            "result": "<result>1</result>"
        },
        {
            "question": "How is it that grown men have started enjoying My Little Pony so much, and why is this not creeping more people out?",
            "answer": (
                "It started on 4chan as a troll, but then a lot of people noticed that it does not actually suck. "
                "This is due to Lauren Faust designed the series from the ground up to be entertaining to not just the 3-7 demographic, but that the parents would not hate.\n\n"
                "THe story lines while cliche and predictable, are entertaining. The VA talent, would be the best and most talented if only they had one or two more names. "
                "The characters are far more tha the stereotypes they initially present as, characters actually develop and change, mostly.\n\n"
                "The episodes range from both Slice of Life, to disney esque stories, to traditional adventure. Not to mention the lessons presented, help remind folks of how we should act. "
                "References to other amazing works see the episodes Read it and weep, Canterlot wedding, and Bridle gossip.\n\n"
                "EDIT: I'm more than open to talking more on this if ya want to. I would especially look into seeing one of the better episodes, the season 2 opener would be my choice if you really want to see why the show is great."
            ),
            "reason": "Excessively long, subjective, includes personal opinions, off-topic content, and an edit note.",
            "result": "<result>0</result>"
        },
        {
            "question": "does neck pain cause shoulder pain?",
            "answer": (
                "a pinched nerve in your neck can cause pain that radiates toward your shoulder. this is also known as cervical radiculopathy. "
                "cervical radiculopathy most often comes from changes in your spine due to aging or injury. bone spurs can cause a pinching of the nerves that run through the hollow space in the vertebrae."
            ),
            "reason": "Clear, relevant, and well-explained without unnecessary details.",
            "result": "<result>1</result>"
        },
    ]

    # Build the few-shot section
    example_block = ""
    for idx, ex in enumerate(few_shot_examples, 1):
        example_block += (
            f"Example {idx}:\n"
            f"Question: {ex['question']}\n"
            f"Answer: {ex['answer']}\n"
            f"Reason: {ex['reason']}\n"
            f"{ex['result']}\n\n"
        )

    # Append example to analyse
    full_prompt = (
        instruction
        + example_block
        + f"Now analyze the following pair in the same way:\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
    )

    return full_prompt

def question_filter(question: str) -> str:
    """
    Constructs a prompt that evaluates whether a question is clearly phrased and free of contextual ambiguity.

    Args:
        question (str): The question to be analyzed.

    Returns:
        str: The constructed prompt for the LLM.
    """

    instruction = (
        "Task:\n"
        "Evaluate the following question to determine whether it is clearly phrased and free of contextual ambiguity.\n\n"
        "Instructions:\n"
        "Assess each question based on the following two criteria:\n"
        "\t1.\tClarity: Can a single, clear question be identified, or does the phrasing contain multiple questions?\n"
        "\t2.\tContext Independence: Can the question be understood and answered without relying on external context (e.g., who “he” refers to or which event is meant)?\n"
        "If either criterion is not met, briefly explain why and output:\n"
        "<result>0</result>\n"
        "If both criteria are fully met, output:\n"
        "<result>1</result>\n\n"
    )

    few_shot_examples = [
        {
            "question": "Does neck pain cause shoulder pain?",
            "reason": "Clearly phrased and context-independent.",
            "result": "<result>1</result>"
        },
        {
            "question": "Why did he die?",
            "reason": "Unclear who “he” refers to → context-sensitive.",
            "result": "<result>0</result>"
        },
        {
            "question": "Why did the Berlin Blockade happen?",
            "reason": "Specific and self-contained question.",
            "result": "<result>1</result>"
        },
        {
            "question": "What are impurities in alcohol and why do they matter? Is drinking alcohol with more impurities less safe? Or does it just taste worse?",
            "reason": "Multiple questions asked in one → lacks clarity.",
            "result": "<result>0</result>"
        },
    ]

    # Build the few-shot section
    example_block = ""
    for idx, ex in enumerate(few_shot_examples, 1):
        example_block += (
            f"Example {idx}:\n"
            f"Question: {ex['question']}\n"
            f"Reason: {ex['reason']}\n"
            f"{ex['result']}\n\n"
        )

    # Append example to analyse
    full_prompt = (
        instruction
        + example_block
        + f"Now analyze the following question in the same way:\n"
        f"Question: {question}\n"
    )

    return full_prompt