INSTRUCTIONS = {
    "simple_router_instructions":  """You are an expert at routing a user question to a web search, or to generate a anwser.

                                    If you have any uncertainty on the anwser, and especially for current events, use web search. You have the web search available. 

                                    Return ONLY a JSON with single key, web_search_needed, that is 'true' or 'false', depending on the question."""
}

def get_instruction(key):
    """
    Retrieve a instruction by key.
    """
    instruction = INSTRUCTIONS.get(key)
    if instruction is None:
        raise KeyError(f"Prompt key '{key}' not found")
    return instruction