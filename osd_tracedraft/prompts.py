"""Profiles render exactly the prompt used for inference; training stores token IDs."""
VICUNA_SYSTEM = ("A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions.")


def render_vicuna(messages):
    result=VICUNA_SYSTEM+" "
    for i,m in enumerate(messages):
        expected="user" if i%2==0 else "assistant"
        if m["role"] != expected: raise ValueError("Vicuna expects alternating user/assistant messages")
        role="USER" if expected=="user" else "ASSISTANT"
        result+=role+": "+m["content"]+(" " if expected=="user" else "</s>")
    if not messages or messages[-1]["role"]!="user": raise ValueError("Expected a pending user turn")
    return result+"ASSISTANT:"


def prompt_ids(tokenizer, messages, profile):
    if profile=="vicuna13b":
        return tokenizer(render_vicuna(messages),add_special_tokens=True)["input_ids"]
    if not getattr(tokenizer,"chat_template",None): raise ValueError("DeepSeek tokenizer has no chat_template")
    text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    return tokenizer(text,add_special_tokens=False)["input_ids"]
