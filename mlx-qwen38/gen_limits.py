"""Output-length limits for server.py (pure Python, no mlx import).

Default is unlimited, like llama.cpp (n_predict = -1) and Ollama (num_predict = -1):
generation stops at an EOS / stop token, a stop string, or when the context window is
full. "Unlimited" is therefore capped at the remaining context, never an endless loop.
"""

from __future__ import annotations

import argparse
from typing import Any, Iterable, Mapping

UNLIMITED = -1
FALLBACK_CONTEXT_LENGTH = 32768  # used only if the model config has no context length


def parse_limit(value: Any, name: str = "max_tokens") -> int | None:
    """Return a token limit (>= 0) or None for unlimited (-1). Other negatives raise ValueError."""
    if isinstance(value, bool):
        raise ValueError(f"'{name}' must be an integer")
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{name}' must be an integer") from None
    if n == UNLIMITED:
        return None
    if n < 0:
        raise ValueError(f"'{name}' must be non-negative, or -1 for unlimited")
    return n


def max_tokens_arg(value: str) -> int:
    """argparse type for --max-tokens: N >= 0, or -1 for unlimited."""
    try:
        parse_limit(value, "--max-tokens")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    return int(value)


def requested_limit(
    req: Mapping[str, Any], keys: Iterable[str], default: int | None
) -> int | None:
    """First non-null request field in `keys` wins; otherwise `default` (None = unlimited)."""
    for key in keys:
        if req.get(key) is not None:
            return parse_limit(req[key], key)
    return default


def effective_max_tokens(limit: int | None, prompt_len: int, context_length: int) -> int:
    """Cap `limit` (None = unlimited) at the context left after the prompt."""
    remaining = int(context_length) - int(prompt_len)
    if remaining <= 0:
        raise ValueError(
            f"Prompt is {prompt_len} tokens, which fills the model context of {context_length} tokens"
        )
    return remaining if limit is None else min(int(limit), remaining)


def context_length_from_config(
    config: Mapping[str, Any] | None, tokenizer_max: Any = None
) -> int:
    """Model context length: text_config / top-level max_position_embeddings, then the
    tokenizer's model_max_length, else FALLBACK_CONTEXT_LENGTH."""
    for section in ((config or {}).get("text_config"), config):
        if isinstance(section, Mapping):
            v = section.get("max_position_embeddings")
            if isinstance(v, int) and not isinstance(v, bool) and v > 0:
                return v
    if isinstance(tokenizer_max, int) and not isinstance(tokenizer_max, bool) and 0 < tokenizer_max < 10**8:
        return tokenizer_max
    return FALLBACK_CONTEXT_LENGTH
