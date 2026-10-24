"""Thin Gemini API client for Gemma 4: disk cache, retries on rate limits, parallel map."""
import hashlib
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from google import genai
from google.genai import errors, types

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemma-4-26b-a4b-it")
INDUCE_MODEL = os.environ.get("INDUCE_MODEL", "gemma-4-31b-it")
CACHE_DIR = Path(os.environ.get("LLM_CACHE_DIR", ".cache/llm"))
MAX_WORKERS = int(os.environ.get("LLM_MAX_WORKERS", "8"))

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
            _client = genai.Client(api_key=key)
    return _client


def _cache_path(payload):
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return CACHE_DIR / digest[:2] / f"{digest}.json"


def generate(prompt, system=None, model=None, thinking="minimal", json_mode=True,
             temperature=0.0, max_tokens=2048, seed=0):
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
    for attempt in range(8):
        try:
            response = client().models.generate_content(model=model, contents=prompt, config=config)
            text = response.text or ""
            if json_mode and not text.strip():
                raise ValueError(f"empty response, finish={response.candidates[0].finish_reason}")
            break
        except (errors.APIError, ValueError) as error:
            code = getattr(error, "code", None)
            if attempt == 7 or (isinstance(code, int) and 400 <= code < 500 and code != 429):
                raise
            time.sleep(min(60, 2 ** attempt + random.random()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"payload": payload, "text": text}, ensure_ascii=False))
    return text


def parse_json(text):
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def pmap(fn, items, workers=None):
    with ThreadPoolExecutor(workers or MAX_WORKERS) as pool:
        return list(pool.map(fn, items))
