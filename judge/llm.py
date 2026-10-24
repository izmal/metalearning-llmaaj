"""Thin Gemini API client for Gemma 4: disk cache, retries on rate limits, parallel map."""
import hashlib
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from google import genai
from google.genai import errors, types

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemma-4-26b-a4b-it")
INDUCE_MODEL = os.environ.get("INDUCE_MODEL", "gemma-4-31b-it")
CACHE_DIR = Path(os.environ.get("LLM_CACHE_DIR", ".cache/llm"))
MAX_WORKERS = int(os.environ.get("LLM_MAX_WORKERS", "4"))
REQUEST_TIMEOUT_S = int(os.environ.get("LLM_REQUEST_TIMEOUT_S", "120"))

_client = None
_client_lock = threading.Lock()


def client():
    # One shared client: a second instance created by a racing thread gets garbage-collected and closes its HTTP pool.
    global _client
    with _client_lock:
        if _client is None:
            key = os.environ.get("GEMINI_API_KEY")
            if not key:
                raise SystemExit("GEMINI_API_KEY is not set")
            # A per-request timeout: with thinking on, some prompts otherwise hang for many minutes.
            _client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_S * 1000))
    return _client


def _cache_path(payload):
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return CACHE_DIR / digest[:2] / f"{digest}.json"


def generate(prompt, system=None, model=None, thinking="minimal", json_mode=True,
             temperature=0.0, max_tokens=2048, seed=0, retries=8):
    """Return the model's text. `thinking` is "minimal" or "high" (Gemma 4 rejects other values)."""
    model = model or JUDGE_MODEL
    payload = {"m": model, "s": system, "p": prompt, "t": thinking, "j": json_mode,
               "temp": temperature, "max": max_tokens, "seed": seed}
    path = _cache_path(payload)
    if path.exists():
        return json.loads(path.read_text())["text"]
    config = types.GenerateContentConfig(
        temperature=temperature, max_output_tokens=max_tokens, system_instruction=system,
        thinking_config=types.ThinkingConfig(thinking_level=thinking),
        response_mime_type="application/json" if json_mode else None)
    attempt = rate_waits = 0
    while True:
        try:
            response = client().models.generate_content(model=model, contents=prompt, config=config)
            text = response.text or ""
            if json_mode and not text.strip():
                raise ValueError(f"empty response, finish={response.candidates[0].finish_reason}")
            break
        except Exception as error:  # API errors, timeouts, empty answers
            code = getattr(error, "code", None)
            if code == 429 and rate_waits < 12:
                # Per-minute quota: wait it out without spending a retry (up to ~6 minutes in total).
                rate_waits += 1
                time.sleep(20 + 20 * random.random())
                continue
            attempt += 1
            if attempt >= retries or (isinstance(code, int) and 400 <= code < 500):
                raise
            time.sleep(min(60, 2 ** attempt + random.random()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"payload": payload, "text": text}, ensure_ascii=False))
    return text


def generate_json(prompt, models, **kwargs):
    """Parsed JSON from the first model that answers; the large model is often overloaded (500/503)."""
    last = None
    for model in models:
        try:
            return parse_json(generate(prompt, model=model, retries=4, **kwargs))
        except Exception as error:
            last = error
            print(f"{model} failed: {str(error)[:120]}", file=sys.stderr)
    raise last


def parse_json(text):
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def pmap(fn, items, workers=None):
    with ThreadPoolExecutor(workers or MAX_WORKERS) as pool:
        return list(pool.map(fn, items))
