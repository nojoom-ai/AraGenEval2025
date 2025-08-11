
# AraGenEval2025/prompts/prompt_templates.py

def build_fewshot_prompt(author_name: str, examples: list, new_text: str) -> str:
    """
    Create a prompt that includes multiple examples and a new input for generation.
    :param author_name: Name of the target author
    :param examples: List of (neutral, styled) tuples
    :param new_text: The new text to rewrite
    :return: Full prompt string
    """
    prompt = f"Rewrite the following text in the style of {author_name}.\n"
    for i, (neutral, styled) in enumerate(examples):
        prompt += f"\nExample {i+1}:\nInput: {neutral}\nOutput: {styled}\n"
    prompt += f"\nNow rewrite this:\nInput: {new_text}\nOutput:"
    return prompt