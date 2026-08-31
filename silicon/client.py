"""OpenAI chat wrapper: single-call `ask` with exponential-backoff retry and
threaded `batch_ask` over (system, user) prompt pairs. Model comes from
SIM_MODEL (default gpt-4o), thread count from SIM_THREADS (default 24).
Requires OPENAI_API_KEY at call time."""
import os

from silicon import require_env
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

MODEL = os.environ.get("SIM_MODEL", "gpt-4o")
N_THREADS = int(os.environ.get("SIM_THREADS", "24"))

_client = None


def get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=require_env("OPENAI_API_KEY"))
    return _client


def ask(system, user, temperature=1.0, max_tokens=80, json_mode=False):
    kwargs = dict(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    for attempt in range(4):
        try:
            r = get_client().chat.completions.create(**kwargs)
            return r.choices[0].message.content
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def batch_ask(prompts, temperature=1.0, max_tokens=80, json_mode=False, progress_every=200):
    """prompts: list of (system, user) tuples. Returns responses in order."""
    n = len(prompts)
    results = [None] * n
    done = [0]

    def worker(idx):
        system, user = prompts[idx]
        results[idx] = ask(system, user, temperature, max_tokens, json_mode)
        done[0] += 1
        if done[0] % progress_every == 0:
            print(f"    batch progress: {done[0]}/{n}")

    with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
        futures = [pool.submit(worker, i) for i in range(n)]
        for f in as_completed(futures):
            f.result()
    return results
