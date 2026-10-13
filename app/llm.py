"""LLM providers behind one streaming interface.

gemini    -> Google Gemini via its OpenAI-compatible endpoint (GEMINI_API_KEY)
anthropic -> Claude (ANTHROPIC_API_KEY)
openai    -> OpenAI or any OpenAI-compatible server such as Groq (OPENAI_API_KEY, OPENAI_BASE_URL)
ollama    -> local model through Ollama (offline, no key)
"extractive" is not an LLM; it is implemented in rag.py and can never fail.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Iterator

from . import config

log = logging.getLogger("veridesk.llm")


class LLMError(Exception):
    pass


def provider_available(name: str) -> bool:
    if name == "gemini":
        return bool(config.GEMINI_API_KEY)
    if name == "anthropic":
        return bool(config.ANTHROPIC_API_KEY)
    if name == "openai":
        return bool(config.OPENAI_API_KEY)
    if name == "ollama":
        return True  # checked lazily (connection error -> fallback)
    return name == "extractive"


def model_name(name: str) -> str:
    return {
        "gemini": config.GEMINI_MODEL,
        "anthropic": config.ANTHROPIC_MODEL,
        "openai": config.OPENAI_MODEL,
        "ollama": config.OLLAMA_MODEL,
        "extractive": "extractive (no LLM)",
    }.get(name, name)


def _openai_stream(client, model: str, system: str, messages: list[dict], extra: dict | None = None) -> Iterator[str]:
    kwargs = dict(
        model=model,
        messages=[{"role": "system", "content": system}, *messages],
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
        stream=True,
    )
    if extra:
        kwargs["extra_body"] = extra
    stream = client.chat.completions.create(**kwargs)
    for ev in stream:
        if ev.choices and ev.choices[0].delta and ev.choices[0].delta.content:
            yield ev.choices[0].delta.content


def stream_chat(provider: str, system: str, messages: list[dict]) -> Iterator[str]:
    """Yield text deltas. Raises LLMError for configuration/transport failures."""
    try:
        if provider in ("gemini", "openai", "ollama"):
            from openai import OpenAI

            if provider == "gemini":
                if not config.GEMINI_API_KEY:
                    raise LLMError("GEMINI_API_KEY is not set")
                client = OpenAI(api_key=config.GEMINI_API_KEY, base_url=config.GEMINI_BASE_URL, timeout=config.LLM_TIMEOUT, max_retries=1)
                effort = os.environ.get("GEMINI_REASONING_EFFORT", "none" if "flash" in config.GEMINI_MODEL else "")
                extra = {"reasoning_effort": effort} if effort else None
                try:
                    yield from _openai_stream(client, config.GEMINI_MODEL, system, messages, extra)
                except Exception as e:  # noqa: BLE001 - some models reject reasoning_effort; retry once without it
                    if extra and "reasoning" in str(e).lower():
                        yield from _openai_stream(client, config.GEMINI_MODEL, system, messages)
                    else:
                        raise
                return
            if provider == "openai":
                if not config.OPENAI_API_KEY:
                    raise LLMError("OPENAI_API_KEY is not set")
                client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL or None, timeout=config.LLM_TIMEOUT, max_retries=1)
                yield from _openai_stream(client, config.OPENAI_MODEL, system, messages)
                return
            client = OpenAI(api_key="ollama", base_url=config.OLLAMA_BASE_URL, timeout=max(config.LLM_TIMEOUT, 120), max_retries=0)
            yield from _openai_stream(client, config.OLLAMA_MODEL, system, messages)
            return
        if provider == "anthropic":
            if not config.ANTHROPIC_API_KEY:
                raise LLMError("ANTHROPIC_API_KEY is not set")
            import anthropic

            client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY, timeout=config.LLM_TIMEOUT, max_retries=1)
            with client.messages.stream(
                model=config.ANTHROPIC_MODEL,
                system=system,
                messages=messages,
                max_tokens=config.LLM_MAX_TOKENS,
                temperature=config.LLM_TEMPERATURE,
            ) as s:
                yield from s.text_stream
            return
        raise LLMError(f"Unknown provider {provider}")
    except LLMError:
        raise
    except Exception as e:  # noqa: BLE001 - normalise SDK errors; never include keys in messages
        msg = f"{type(e).__name__}: {str(e)[:200]}"
        for key in (config.GEMINI_API_KEY, config.ANTHROPIC_API_KEY, config.OPENAI_API_KEY):
            if key:
                msg = msg.replace(key, "***")
        raise LLMError(msg) from e


def complete(provider: str, system: str, messages: list[dict]) -> str:
    return "".join(stream_chat(provider, system, messages))
