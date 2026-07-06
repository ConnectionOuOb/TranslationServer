def estimate_tokens(text: str) -> int:
    """Rough token estimate for mixed-language text (conservative for limit checks)."""
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    other_chars = len(text) - ascii_chars
    return int(ascii_chars / 4 + other_chars / 1.5)
