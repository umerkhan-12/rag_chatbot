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

import re
import time
from collections.abc import Iterator

from . import config

log = logging.getLogger("veridesk.llm")


class LLMError(Exception):
    pass


class _LastModel:  # which model actually answered (fallback models may take over on rate limits); demo is single-user
    value: str | None = None


_last_model = _LastModel()


def last_model(provider: str) -> str:
    m = getattr(_last_model, "value", None)
    return m if (m and provider == "openai") else model_name(provider)


def provider_available(name: str) -> bool:
    if name == "gemini":
        return bool(config.GEMINI_API_KEY)
    if name == "anthropic":
        return bool(config.ANTHROPIC_API_KEY)
    if name == "openai":
        return bool(config.OPENAI_API_KEY)
    if name == "compat2":
        return bool(config.COMPAT2_API_KEY)
    if name == "ollama":
        return True  # checked lazily (connection error -> fallback)
    return name == "extractive"


def model_name(name: str) -> str:
    return {
        "gemini": config.GEMINI_MODEL,
        "anthropic": config.ANTHROPIC_MODEL,
        "openai": config.OPENAI_MODEL,
        "ollama": config.OLLAMA_MODEL,
        "compat2": config.COMPAT2_MODEL,
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
        if provider == "compat2":
            from openai import OpenAI

            if not config.COMPAT2_API_KEY:
                raise LLMError("COMPAT2_API_KEY is not set")
            client = OpenAI(api_key=config.COMPAT2_API_KEY, base_url=config.COMPAT2_BASE_URL, timeout=config.LLM_TIMEOUT, max_retries=1)
            yield from _openai_stream(client, config.COMPAT2_MODEL, system, messages)
            return
        if provider in ("gemini", "openai", "ollama"):
            from openai import OpenAI

            if provider == "gemini":
                if not config.GEMINI_API_KEY:
                    raise LLMError("GEMINI_API_KEY is not set")
                client = OpenAI(api_key=config.GEMINI_API_KEY, base_url=config.GEMINI_BASE_URL, timeout=config.LLM_TIMEOUT, max_retries=1)
                effort = os.environ.get("GEMINI_REASONING_EFFORT", "")  # not sent by default: some models reject it
                models = [config.GEMINI_MODEL] + [m for m in [config.GEMINI_FALLBACK_MODEL] if m and m != config.GEMINI_MODEL]
                last: Exception | None = None
                for model in models:  # quotas are per model, so a second model often works when the first is rate-limited
                    for extra in ([{"reasoning_effort": effort}, None] if effort else [None]):
                        started = False
                        try:
                            for delta in _openai_stream(client, model, system, messages, extra):
                                started = True
                                yield delta
                            return
                        except Exception as e:  # noqa: BLE001
                            if started:
                                raise  # never retry after text was streamed (would duplicate output)
                            last = e
                            log.warning("Gemini model %s failed (%s)%s", model, type(e).__name__, "; retrying" if model != models[-1] or extra else "")
                raise last  # type: ignore[misc]
            if provider == "openai":
                if not config.OPENAI_API_KEY:
                    raise LLMError("OPENAI_API_KEY is not set")
                # max_retries=0: on a rate limit we switch to the next model at once instead of waiting.
                # Free-tier limits (e.g. Groq: 8k tokens/min) are per model, so rotating models multiplies capacity.
                client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL or None, timeout=config.LLM_TIMEOUT, max_retries=0)
                models = list(dict.fromkeys([config.OPENAI_MODEL, *config.OPENAI_FALLBACK_MODELS]))
                last: Exception | None = None
                min_wait = 999.0
                for rnd in range(2):  # second round after a short pause, in case every model was throttled
                    for model in models:
                        started = False
                        try:
                            for delta in _openai_stream(client, model, system, messages):
                                if not started:
                                    _last_model.value = model
                                started = True
                                yield delta
                            return
                        except Exception as e:  # noqa: BLE001
                            if started:
                                raise
                            last = e
                            log.warning("Model %s failed (%s); trying next", model, type(e).__name__)
                            m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", str(e))
                            if m:  # rate limit: remember the shortest advertised wait
                                wait = int(m.group(1) or 0) * 60 + float(m.group(2))
                                min_wait = min(min_wait, wait)
                    if rnd == 0:
                        if min_wait > 15:
                            break  # waiting longer would stall the demo; let the extractive fallback answer
                        time.sleep(min(max(min_wait, 1.0), 15) if min_wait < 999 else 3)
                raise last  # type: ignore[misc]
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
