"""
DCX-AgenticTrader — LLM Client (OpenRouter)

Thin wrapper around the OpenRouter (OpenAI-compatible) API.
All LLM calls in the trading pipeline go through this module.

Features:
- Candidate model fallback chain (tries models in order)
- Structured JSON output with Pydantic parsing
- Retry with exponential backoff on 429 / 5xx
- Full audit logging via Loguru
- Global kill switch (use_llm flag)
"""

import json
import time
import hashlib
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from config.settings import get_settings
from utils.logger import get_agent_logger
from utils.error_handler import (
    LLMError,
    LLMParseError,
    LLMRateLimitError,
)

log = get_agent_logger("llm_client")

T = TypeVar("T", bound=BaseModel)

# Default candidate models — tried in order.
# Free-tier model IDs rotate on OpenRouter; keep this list in settings.py.
DEFAULT_CANDIDATE_MODELS = [
    "meta-llama/llama-4-maverick:free",
    "google/gemini-2.0-flash-exp:free",
    "openrouter/auto",
]

# Maximum temperature allowed for this decision pipeline
MAX_TEMPERATURE = 0.3


def _prompt_hash(text: str) -> str:
    """Short hash of the prompt for audit logging (not the full text)."""
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _build_json_schema_instruction(model_class: Type[BaseModel]) -> str:
    """Build a JSON schema instruction string from a Pydantic model."""
    schema = model_class.model_json_schema()
    # Remove definitions and allOf nesting for clarity in prompt
    schema.pop("$defs", None)
    return (
        "You MUST respond with ONLY a valid JSON object matching this exact schema. "
        "Do NOT include any text outside the JSON object.\n"
        f"Schema:\n```json\n{json.dumps(schema, indent=2)}\n```"
    )


def call_llm(
    system_prompt: str,
    user_prompt: str,
    response_model: Type[T],
    *,
    candidate_models: Optional[List[str]] = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    max_retries: int = 2,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    agent_name: str = "unknown",
) -> T:
    """
    Call an LLM via OpenRouter and parse the response into a Pydantic model.

    Tries candidate models in order. Retries on 429/5xx per model.
    Raises LLMError if all models and retries are exhausted.

    Args:
        system_prompt: System-level instruction for the LLM.
        user_prompt: The main user/task prompt.
        response_model: Pydantic model class for structured output parsing.
        candidate_models: Ordered list of model IDs to try. Falls back to config default.
        temperature: Sampling temperature (capped at MAX_TEMPERATURE).
        max_tokens: Maximum response tokens.
        max_retries: Retries per model on transient errors.
        initial_delay: Initial retry delay in seconds.
        backoff_factor: Retry delay multiplier.
        agent_name: Name of the calling agent (for logging).

    Returns:
        Parsed Pydantic model instance.

    Raises:
        LLMError: If all models and retries are exhausted.
        LLMParseError: If response cannot be parsed (after retries).
    """
    settings = get_settings()

    # Global kill switch
    if settings.llm_provider == "none":
        raise LLMError("LLM disabled (LLM_PROVIDER=none)", model="none")

    # Resolve candidate models
    models = candidate_models or settings.llm_candidate_models or DEFAULT_CANDIDATE_MODELS

    # Enforce temperature cap
    temperature = min(temperature, MAX_TEMPERATURE)

    # Build JSON schema instruction
    schema_instruction = _build_json_schema_instruction(response_model)
    full_system = f"{system_prompt}\n\n{schema_instruction}"

    api_key = settings.openrouter_api_key
    if not api_key:
        raise LLMError("OPENROUTER_API_KEY not set", model="none")

    prompt_id = _prompt_hash(user_prompt)
    last_error: Optional[Exception] = None

    for model_id in models:
        for attempt in range(max_retries + 1):
            start_time = time.time()
            try:
                result = _call_openrouter(
                    api_key=api_key,
                    model=model_id,
                    system_prompt=full_system,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                latency = time.time() - start_time

                # Parse response
                parsed = _parse_response(result, response_model, model_id)

                # Audit log
                log.info(
                    f"[LLM] agent={agent_name} model={model_id} "
                    f"latency={latency:.2f}s prompt={prompt_id} "
                    f"result={_safe_summary(parsed)}"
                )
                return parsed

            except LLMRateLimitError as e:
                last_error = e
                delay = e.retry_after or (initial_delay * (backoff_factor ** attempt))
                log.warning(
                    f"[LLM] Rate limited on {model_id} (attempt {attempt+1}/{max_retries+1}). "
                    f"Retrying in {delay:.1f}s..."
                )
                if attempt < max_retries:
                    time.sleep(delay)

            except LLMParseError as e:
                last_error = e
                latency = time.time() - start_time
                log.warning(
                    f"[LLM] Parse error on {model_id} (attempt {attempt+1}/{max_retries+1}): {e}. "
                    f"Raw output: {e.raw_output[:200]}"
                )
                if attempt < max_retries:
                    time.sleep(initial_delay * (backoff_factor ** attempt))

            except LLMError as e:
                last_error = e
                latency = time.time() - start_time
                log.warning(
                    f"[LLM] Error on {model_id} (attempt {attempt+1}/{max_retries+1}): {e}"
                )
                if attempt < max_retries:
                    time.sleep(initial_delay * (backoff_factor ** attempt))

        # All retries exhausted for this model — try next
        log.warning(f"[LLM] All retries exhausted for {model_id}, trying next model...")

    # All models exhausted
    raise LLMError(
        f"All LLM models exhausted after trying {len(models)} models. Last error: {last_error}",
        model="all_exhausted",
    )


def _call_openrouter(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """
    Make a raw HTTP call to the OpenRouter chat completions endpoint.

    Returns the assistant message content string.
    Raises LLMRateLimitError on 429, LLMError on other failures.
    """
    import requests

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/DCX-AgenticTrader",
        "X-Title": "DCX-AgenticTrader",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
    except requests.exceptions.Timeout:
        raise LLMError(f"Request to {model} timed out after 60s", model=model)
    except requests.exceptions.ConnectionError as e:
        raise LLMError(f"Connection error to OpenRouter: {e}", model=model)

    if response.status_code == 429:
        retry_after = None
        try:
            retry_after = float(response.headers.get("Retry-After", "5"))
        except (ValueError, TypeError):
            retry_after = 5.0
        raise LLMRateLimitError(
            f"Rate limited by OpenRouter for model {model}",
            retry_after=retry_after,
            model=model,
        )

    if response.status_code >= 500:
        raise LLMError(
            f"OpenRouter server error ({response.status_code}): {response.text[:200]}",
            model=model,
        )

    if response.status_code != 200:
        raise LLMError(
            f"OpenRouter API error ({response.status_code}): {response.text[:300]}",
            model=model,
        )

    try:
        data = response.json()
    except Exception:
        raise LLMError(f"Invalid JSON response from {model}", model=model)

    # Check for error in response body (OpenRouter sometimes returns 200 with error)
    if "error" in data:
        error_msg = data["error"].get("message", str(data["error"]))
        raise LLMError(f"OpenRouter error for {model}: {error_msg}", model=model)

    # Extract content
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise LLMError(f"Unexpected response structure from {model}: {str(data)[:200]}", model=model)

    return content


def _parse_response(raw: str, model_class: Type[T], model_id: str) -> T:
    """
    Parse raw LLM text output into a Pydantic model.

    Handles markdown code fences, leading/trailing whitespace, etc.
    Raises LLMParseError on failure.
    """
    cleaned = raw.strip()

    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    # Try to find JSON object within the text
    if not cleaned.startswith("{"):
        # Look for first { to last }
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start:end + 1]

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise LLMParseError(
            f"JSON decode error: {e}",
            raw_output=raw[:500],
            model=model_id,
        )

    try:
        return model_class.model_validate(data)
    except ValidationError as e:
        raise LLMParseError(
            f"Pydantic validation error: {e}",
            raw_output=raw[:500],
            model=model_id,
        )


def _safe_summary(obj: BaseModel) -> str:
    """Create a short summary of a parsed result for logging."""
    d = obj.model_dump()
    # Truncate long string values
    summary = {}
    for k, v in d.items():
        if isinstance(v, str) and len(v) > 80:
            summary[k] = v[:80] + "..."
        elif isinstance(v, list) and len(v) > 3:
            summary[k] = f"[{len(v)} items]"
        else:
            summary[k] = v
    return json.dumps(summary, default=str)
