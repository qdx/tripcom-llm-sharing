import tiktoken

_encoding = tiktoken.encoding_for_model("gpt-4")


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken gpt-4 encoding."""
    if not text:
        return 0
    return len(_encoding.encode(text))


def count_message_tokens(message: dict) -> int:
    """Count tokens for a single chat message (role + content + overhead)."""
    tokens = 4  # every message has <|start|>role\n ... <|end|>\n overhead
    tokens += count_tokens(message.get("role", ""))
    content = message.get("content", "")
    if content:
        tokens += count_tokens(content)
    if message.get("tool_calls"):
        for tc in message["tool_calls"]:
            fn = tc.get("function", {})
            tokens += count_tokens(fn.get("name", ""))
            tokens += count_tokens(fn.get("arguments", ""))
    if message.get("name"):
        tokens += count_tokens(message["name"])
    return tokens
