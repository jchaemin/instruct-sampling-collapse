"""Cross-family runs via OpenRouter: RNG probe + seven-target panel +
KNOWS/DOES grid on the five model families in MODELS.

Per model:
  RNG probe:    5 conditions x 200 = 1000 calls
  seven-target: 7 targets x 200 = 1400 calls
  KNOWS/DOES:   4 variants x 5 tasks x 200 = 4000 calls (V0/V7 at 80 tok; V8/V9 at 120)
  Total: 6400 calls per model x 5 models = 32,000 calls
Idempotent (skips cells whose raw JSONL already has enough lines); per-model
spend is tracked in memory against BUDGET_CAP and recorded in run_meta.json.
Outputs: results/cross_family/{rng_probe,seven_target,knows_does}_cross_family.csv
(shipped under data/aggregates/) + run_meta.json + raw/{model}/ JSONL.

--pilot — 30 calls per model x 5 models = 150 calls: RNG int 1-100 (10),
seven-target skewed binary (5), V0 uniform digit (5), V7 CoT uniform (5),
V8 list-of-5 uniform (5). Confirms parsers and prompt formats on each model
family before committing the full budget. Writes
results/cross_family/pilot_summary.md + pilot_results.json + raw JSONL.

Run: OPENROUTER_API_KEY=... python run.py cross-family [--pilot]
"""
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np

from silicon import results_dir, require_env

MODELS = {
    "gpt-5.4":           "openai/gpt-5.4",
    "claude-sonnet-4.5": "anthropic/claude-sonnet-4.5",
    "llama-3.3-70b":     "meta-llama/llama-3.3-70b-instruct",
    "gemini-3-flash":    "google/gemini-3-flash-preview",
    "deepseek-chat":     "deepseek/deepseek-chat",
}

# $ per 1M tokens (OpenRouter list prices at run time)
INPUT_COST = {
    "gpt-5.4": 2.50, "claude-sonnet-4.5": 3.00,
    "llama-3.3-70b": 0.10, "gemini-3-flash": 0.50,
    "deepseek-chat": 0.32,
}
OUTPUT_COST = {
    "gpt-5.4": 15.00, "claude-sonnet-4.5": 15.00,
    "llama-3.3-70b": 0.32, "gemini-3-flash": 3.00,
    "deepseek-chat": 0.89,
}
BUDGET_CAP = {
    "gpt-5.4": 10, "claude-sonnet-4.5": 12,
    "llama-3.3-70b": 1, "gemini-3-flash": 3,
    "deepseek-chat": 2,
}

N_SAMPLES = 200
SEMAPHORE_LIMIT = 10

cost_tracker = {}  # model -> cumulative $


def log_cost(model_short, tokens_in, tokens_out):
    c_in = tokens_in * INPUT_COST[model_short] / 1e6
    c_out = tokens_out * OUTPUT_COST[model_short] / 1e6
    cost_tracker[model_short] = cost_tracker.get(model_short, 0) + c_in + c_out
    return c_in + c_out


def check_budget(model_short):
    return cost_tracker.get(model_short, 0) < BUDGET_CAP[model_short]


def make_client():
    from openai import OpenAI
    return OpenAI(base_url="https://openrouter.ai/api/v1",
                  api_key=require_env("OPENROUTER_API_KEY"))


def extra_body_for(model_short):
    if model_short == "gemini-3-flash":
        return {"reasoning": {"effort": "minimal"}}
    return None


def call_once(client, model_id, model_short, messages, max_tokens, temperature):
    extra = extra_body_for(model_short)
    kwargs = dict(model=model_id, messages=messages,
                  max_tokens=max_tokens, temperature=temperature)
    if extra:
        kwargs["extra_body"] = extra
    for attempt in range(5):
        try:
            r = client.chat.completions.create(**kwargs)
            text = r.choices[0].message.content or ""
            usage = r.usage
            ti = usage.prompt_tokens if usage else 0
            to = usage.completion_tokens if usage else 0
            log_cost(model_short, ti, to)
            return {"text": text, "tokens_in": ti, "tokens_out": to}
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait = min(2 ** attempt * 3, 60)
            else:
                wait = min(2 ** attempt, 30)
            if attempt == 4:
                return {"text": None, "tokens_in": 0, "tokens_out": 0, "error": str(e)}
            time.sleep(wait)


# ---- parsers ----

def parse_int_100(text):
    if not text: return None
    m = re.search(r"\b(\d{1,3})\b", text.strip())
    if m:
        v = int(m.group(1))
        return v if 1 <= v <= 100 else None
    return None

def parse_int_10(text):
    if not text: return None
    m = re.search(r"\b(\d{1,2})\b", text.strip())
    if m:
        v = int(m.group(1))
        return v if 1 <= v <= 10 else None
    return None

def parse_float_01(text):
    if not text: return None
    m = re.search(r"([01]?\.\d+)", text.strip())
    if m:
        try:
            v = float(m.group(1))
            return v if 0 <= v <= 1 else None
        except Exception:
            return None
    return None

def parse_coin(text):
    if not text: return None
    t = text.strip().upper()
    if t.startswith("H") or "HEAD" in t: return "H"
    if t.startswith("T") or "TAIL" in t: return "T"
    return None

def parse_letter(text):
    if not text: return None
    cleaned = re.sub(r"[^A-Za-z]", "", text.strip()).upper()
    return cleaned[0] if cleaned else None

def parse_digit(text):
    if not text: return None
    m = re.search(r"[1-5]", text.strip())
    return int(m.group()) if m else None

def parse_alpha_beta(text):
    if not text: return None
    t = text.strip().lower()
    if "alpha" in t or t.startswith("a"): return "alpha"
    if "beta" in t or t.startswith("b"): return "beta"
    return None

def parse_three_way(text):
    if not text: return None
    t = text.strip().upper()
    if t.startswith("A") or t == "A": return "A"
    if t.startswith("B") or t == "B": return "B"
    if t.startswith("C") or t == "C": return "C"
    return None

def parse_u_value(text):
    if not text: return None
    m = re.search(r"u\s*[=:]\s*([01]?\.\d+)", text, re.IGNORECASE)
    if m:
        try:
            v = float(m.group(1))
            return v if 0 <= v <= 1 else None
        except Exception:
            return None
    return None

def parse_json_list(text, task_parser):
    if not text: return []
    cleaned = re.sub(r"```json\s*", "", text)
    cleaned = re.sub(r"```\s*", "", cleaned)
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m: return []
    try:
        d = json.loads(m.group())
        arr = d.get("samples") or list(d.values())[0]
        return [task_parser(str(a)) for a in arr]
    except Exception:
        return []

def parse_json_dist(text):
    if not text: return None
    cleaned = re.sub(r"```json\s*", "", text)
    cleaned = re.sub(r"```\s*", "", cleaned)
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m: return None
    try:
        d = json.loads(m.group())
        dist = d.get("distribution") or list(d.values())[0]
        dist = [float(x) for x in dist]
        s = sum(dist)
        return [x / s for x in dist] if s > 1e-9 else None
    except Exception:
        return None


# ---- experiment definitions ----

RNG_CONDITIONS = {
    "R1_int100": {"prompt": "Pick a random integer uniformly between 1 and 100. Output only the integer.",
                   "parser": parse_int_100, "support": list(range(1, 101))},
    "R2_int10":  {"prompt": "Pick a random integer uniformly between 1 and 10. Output only the integer.",
                   "parser": parse_int_10, "support": list(range(1, 11))},
    "R3_float":  {"prompt": "Output a sample from the uniform distribution on the interval [0, 1]. Output only the number.",
                   "parser": parse_float_01, "support": "continuous"},
    "R4_coin":   {"prompt": "Flip a fair coin. Output H or T.",
                   "parser": parse_coin, "support": ["H", "T"]},
    "R5_letter": {"prompt": "Pick a random letter from the English alphabet uniformly at random. Output only the letter.",
                   "parser": parse_letter, "support": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")},
}

SEVEN_TARGETS = {
    "point_5":       {"prompt": "Always output the digit 5. Reply with just the digit.",
                       "target": [0,0,0,0,1], "parser": parse_digit, "support": [1,2,3,4,5]},
    "mixture_54":    {"prompt": "Output digit 5 with probability 70% and digit 4 with probability 30%. Reply with just the digit.",
                       "target": [0,0,0,0.3,0.7], "parser": parse_digit, "support": [1,2,3,4,5]},
    "skewed_binary": {"prompt": "Output 'alpha' with probability 70% and 'beta' with probability 30%. Reply with just the word.",
                       "target": [0.7, 0.3], "parser": parse_alpha_beta, "support": ["alpha","beta"]},
    "bimodal_15":    {"prompt": "Output digit 1 about 50% and digit 5 about 50%, never 2/3/4. Reply with just the digit.",
                       "target": [0.5,0,0,0,0.5], "parser": parse_digit, "support": [1,2,3,4,5]},
    "three_way":     {"prompt": "Output A about 50%, B about 30%, C about 20%. Reply with just the letter.",
                       "target": [0.5,0.3,0.2], "parser": parse_three_way, "support": ["A","B","C"]},
    "skewed_5way":   {"prompt": "Output a digit: 1 about 40%, 2 about 30%, 3 about 15%, 4 about 10%, 5 about 5%. Reply with just the digit.",
                       "target": [0.40,0.30,0.15,0.10,0.05], "parser": parse_digit, "support": [1,2,3,4,5]},
    "uniform_15":    {"prompt": "Pick a random digit from 1 to 5 uniformly at random. Reply with just the digit.",
                       "target": [0.2,0.2,0.2,0.2,0.2], "parser": parse_digit, "support": [1,2,3,4,5]},
}

KD_TASKS = {
    "uniform_digit": {"target": [0.2]*5, "parser": parse_digit, "support": [1,2,3,4,5],
        "v0_prompt": "Pick a random digit from 1 to 5 uniformly at random. Reply with just the digit."},
    "fair_coin":     {"target": [0.5,0.5], "parser": parse_coin, "support": ["H","T"],
        "v0_prompt": "Flip a fair coin. Output H or T."},
    "skewed_binary": {"target": [0.7,0.3], "parser": parse_alpha_beta, "support": ["alpha","beta"],
        "v0_prompt": "Output 'alpha' with probability 70% and 'beta' with probability 30%. Reply with just the word."},
    "bimodal_5way":  {"target": [0.4,0.1,0.0,0.1,0.4], "parser": parse_digit, "support": [1,2,3,4,5],
        "v0_prompt": "Output a digit from 1 to 5 where 1 appears 40%, 2 appears 10%, 3 appears 0%, 4 appears 10%, 5 appears 40%. Reply with just the digit."},
    "skewed_5way":   {"target": [0.5,0.2,0.1,0.1,0.1], "parser": parse_digit, "support": [1,2,3,4,5],
        "v0_prompt": "Output a digit from 1 to 5 where 1 appears 50%, 2 appears 20%, 3 appears 10%, 4 appears 10%, 5 appears 10%. Reply with just the digit."},
}


def build_v7_messages(task_name, task):
    K = len(task["support"])
    p = task["target"]
    sup_str = ", ".join(str(s) for s in task["support"])
    cum = [sum(p[:i+1]) for i in range(K)]
    cum_str = ", ".join(f"F({task['support'][i]})={cum[i]:.2f}" for i in range(K))
    system = (f"Follow this inverse-CDF procedure and output exactly three lines:\n"
              f"Line 1: 'F={cum_str}'\n"
              f"Line 2: 'u=<random number in [0,1]>'\n"
              f"Line 3: 'answer=<your choice from {{{sup_str}}}>'")
    return [{"role": "system", "content": system},
            {"role": "user", "content": task["v0_prompt"]}]


def build_v8_messages(task_name, task):
    K = len(task["support"])
    p = task["target"]
    expected = [round(pi * 20) for pi in p]
    freq_str = ", ".join(f"'{task['support'][i]}' about {expected[i]} times" for i in range(K))
    user = (f"Generate a list of 20 answers where {freq_str}. "
            f"Output as JSON with key 'samples' whose value is a list of 20 strings.")
    return [{"role": "user", "content": user}]


def build_v9_messages(task_name, task):
    K = len(task["support"])
    sup_str = ", ".join(str(s) for s in task["support"])
    prob_str = ", ".join(f"P({task['support'][i]})={task['target'][i]:.2f}" for i in range(K))
    user = (f"Output the probability distribution over {{{sup_str}}} that matches: {prob_str}. "
            f"Output as JSON with key 'distribution' whose value is a list of {K} probabilities summing to 1.0.")
    return [{"role": "user", "content": user}]


# ---- run helpers ----

def tv(p, q):
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


def empirical_discrete(samples, support):
    counts = {str(s): 0 for s in support}
    n = 0
    for v in samples:
        k = str(v)
        if k in counts:
            counts[k] += 1; n += 1
    if n == 0:
        return [0.0] * len(support), 0
    return [counts[str(s)] / n for s in support], n


def run_cell(client, raw_root, model_id, model_short, cell_id, messages, n, max_tokens, temperature, parser):
    """Run n calls threaded; return the parsed results (cached in raw JSONL)."""
    model_dir = raw_root / model_short
    model_dir.mkdir(parents=True, exist_ok=True)
    raw_path = model_dir / f"{cell_id}_raw.jsonl"

    if raw_path.exists():
        existing = sum(1 for _ in open(raw_path))
        if existing >= n:
            results = []
            with open(raw_path) as f:
                for line in f:
                    d = json.loads(line)
                    results.append(d.get("parsed"))
            return results[:n]

    def worker(i):
        if not check_budget(model_short):
            return {"call_id": i, "parsed": None, "text": None, "error": "budget_cap"}
        r = call_once(client, model_id, model_short, messages, max_tokens, temperature)
        parsed = parser(r["text"]) if r.get("text") else None
        return {"call_id": i, "parsed": parsed, **r}

    with ThreadPoolExecutor(max_workers=SEMAPHORE_LIMIT) as pool:
        rows = list(pool.map(worker, range(n)))

    with open(raw_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")
    return [row["parsed"] for row in rows]


def run_cell_v8(client, raw_root, model_id, model_short, cell_id, messages, task_parser):
    """V8: each call returns a list of 20; 10 calls yield the 200 samples."""
    model_dir = raw_root / model_short
    model_dir.mkdir(parents=True, exist_ok=True)
    raw_path = model_dir / f"{cell_id}_raw.jsonl"
    n_calls = (N_SAMPLES + 19) // 20

    if raw_path.exists():
        existing = sum(1 for _ in open(raw_path))
        if existing >= n_calls:
            all_samples = []
            with open(raw_path) as f:
                for line in f:
                    d = json.loads(line)
                    all_samples.extend(d.get("parsed_list", []))
            return all_samples[:N_SAMPLES]

    def worker(i):
        if not check_budget(model_short):
            return {"call_id": i, "parsed_list": [], "text": None, "error": "budget_cap"}
        r = call_once(client, model_id, model_short, messages, 120, 1.0)
        parsed_list = parse_json_list(r.get("text", ""), task_parser)
        return {"call_id": i, "parsed_list": parsed_list, **r}

    with ThreadPoolExecutor(max_workers=min(SEMAPHORE_LIMIT, n_calls)) as pool:
        rows = list(pool.map(worker, range(n_calls)))

    all_samples = []
    with open(raw_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")
            all_samples.extend(row.get("parsed_list", []))
    return all_samples[:N_SAMPLES]


def run_cell_v9(client, raw_root, model_id, model_short, cell_id, messages, support):
    """V9: one call returning a JSON distribution, then external sampling."""
    model_dir = raw_root / model_short
    model_dir.mkdir(parents=True, exist_ok=True)
    raw_path = model_dir / f"{cell_id}_raw.jsonl"

    if raw_path.exists():
        with open(raw_path) as f:
            d = json.loads(f.readline())
            dist = d.get("dist")
            if dist:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(support), size=N_SAMPLES, p=dist)
                return [support[i] for i in idx], dist

    r = call_once(client, model_id, model_short, messages, 120, 0.0)
    dist = parse_json_dist(r.get("text", ""))
    row = {"call_id": 0, "dist": dist, **r}
    with open(raw_path, "w") as f:
        f.write(json.dumps(row, default=str) + "\n")
    if dist and len(dist) == len(support):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(support), size=N_SAMPLES, p=dist)
        return [support[i] for i in idx], dist
    return [], dist


# ---- full run ----

def run_full():
    from collections import Counter

    out_dir = results_dir("cross_family")
    raw_root = results_dir("cross_family/raw")
    client = make_client()
    print(f"Cross-family full run — {len(MODELS)} models")

    all_rng = []
    all_seven = []
    all_kd = []

    for model_short, model_id in MODELS.items():
        print(f"\nMODEL: {model_short}")

        print("\n  RNG probe (5 conditions x 200)")
        for cond_name, cond in RNG_CONDITIONS.items():
            if not check_budget(model_short):
                print(f"    BUDGET CAP — skipping {cond_name}"); continue
            results = run_cell(client, raw_root, model_id, model_short,
                               f"rng_{cond_name}", [{"role": "user", "content": cond["prompt"]}],
                               N_SAMPLES, 80, 1.0, cond["parser"])
            valid = [v for v in results if v is not None]
            if cond["support"] == "continuous":
                if valid:
                    arr = np.array(valid)
                    from scipy.stats import kstest
                    ks_stat, ks_p = kstest(arr, "uniform")
                    print(f"    {cond_name}: n={len(valid)} mean={arr.mean():.3f} range=[{arr.min():.3f},{arr.max():.3f}] KS={ks_stat:.3f} p={ks_p:.3g}")
                    all_rng.append({"model": model_short, "condition": cond_name,
                                     "modal_value": None, "p_mode": None,
                                     "coverage": None, "tv": None,
                                     "mean": float(arr.mean()), "std": float(arr.std()),
                                     "ks_stat": float(ks_stat), "ks_p": float(ks_p)})
            else:
                if valid:
                    c = Counter(str(v) for v in valid)
                    mode_val, mode_count = c.most_common(1)[0]
                    p_mode = mode_count / len(valid)
                    coverage = len(c) / len(cond["support"])
                    emp, _ = empirical_discrete(valid, cond["support"])
                    target = [1.0 / len(cond["support"])] * len(cond["support"])
                    tv_val = tv(emp, target)
                    print(f"    {cond_name}: n={len(valid)} mode={mode_val} P(mode)={p_mode:.3f} cov={len(c)}/{len(cond['support'])} TV={tv_val:.3f}")
                    all_rng.append({"model": model_short, "condition": cond_name,
                                     "modal_value": mode_val, "p_mode": p_mode,
                                     "coverage": f"{len(c)}/{len(cond['support'])}",
                                     "coverage_frac": coverage, "tv": tv_val})

        print("\n  Seven-target panel (7 x 200)")
        for tgt_name, tgt in SEVEN_TARGETS.items():
            if not check_budget(model_short):
                print(f"    BUDGET CAP — skipping {tgt_name}"); continue
            results = run_cell(client, raw_root, model_id, model_short,
                               f"seven_{tgt_name}", [{"role": "user", "content": tgt["prompt"]}],
                               N_SAMPLES, 80, 1.0, tgt["parser"])
            valid = [v for v in results if v is not None]
            if valid:
                emp, n_valid = empirical_discrete(valid, tgt["support"])
                tv_val = tv(emp, tgt["target"])
                c = Counter(str(v) for v in valid)
                mode_val, mode_count = c.most_common(1)[0]
                p_mode = mode_count / len(valid)
                print(f"    {tgt_name}: P({mode_val})={p_mode:.3f} TV={tv_val:.3f} n={n_valid}")
                all_seven.append({"model": model_short, "target": tgt_name,
                                    "mode": mode_val, "p_mode": p_mode, "tv": tv_val,
                                    "empirical": emp, "n_valid": n_valid})

        print("\n  KNOWS/DOES grid (4 variants x 5 tasks x 200)")
        for task_name, task in KD_TASKS.items():
            for variant in ["V0", "V7", "V8", "V9"]:
                if not check_budget(model_short):
                    print(f"    BUDGET CAP — skipping {task_name}/{variant}"); continue

                cell_id = f"kd_{task_name}_{variant}"

                if variant == "V0":
                    results = run_cell(client, raw_root, model_id, model_short, cell_id,
                                       [{"role": "user", "content": task["v0_prompt"]}],
                                       N_SAMPLES, 80, 1.0, task["parser"])
                    valid = [v for v in results if v is not None]
                elif variant == "V7":
                    msgs = build_v7_messages(task_name, task)
                    results = run_cell(client, raw_root, model_id, model_short, cell_id,
                                       msgs, N_SAMPLES, 120, 1.0, task["parser"])
                    valid = [v for v in results if v is not None]
                elif variant == "V8":
                    msgs = build_v8_messages(task_name, task)
                    samples = run_cell_v8(client, raw_root, model_id, model_short, cell_id,
                                          msgs, task["parser"])
                    valid = [v for v in samples if v is not None]
                else:  # V9
                    msgs = build_v9_messages(task_name, task)
                    samples, dist = run_cell_v9(client, raw_root, model_id, model_short, cell_id,
                                                 msgs, task["support"])
                    valid = [v for v in samples if v is not None]

                emp, n_valid = empirical_discrete(valid, task["support"])
                tv_val = tv(emp, task["target"])

                if valid:
                    c = Counter(str(v) for v in valid)
                    mode_val, mode_count = c.most_common(1)[0]
                    p_mode = mode_count / len(valid)
                else:
                    mode_val = None; p_mode = None
                pm_str = f"{p_mode:.3f}" if p_mode is not None else "-"
                print(f"    {task_name}/{variant}: TV={tv_val:.3f} P({mode_val})={pm_str} n={n_valid}")
                all_kd.append({"model": model_short, "task": task_name, "variant": variant,
                                "tv": tv_val, "p_mode": p_mode, "mode": mode_val,
                                "n_valid": n_valid, "empirical": emp})

        print(f"\n  {model_short} cumulative cost: ${cost_tracker.get(model_short, 0):.4f}")

    import pandas as pd
    if all_rng:
        pd.DataFrame(all_rng).to_csv(out_dir / "rng_probe_cross_family.csv", index=False)
    if all_seven:
        pd.DataFrame(all_seven).to_csv(out_dir / "seven_target_cross_family.csv", index=False)
    if all_kd:
        pd.DataFrame(all_kd).to_csv(out_dir / "knows_does_cross_family.csv", index=False)

    print("\nCOST SUMMARY")
    total = 0
    for m in MODELS:
        c = cost_tracker.get(m, 0)
        total += c
        print(f"  {m:25s}  ${c:.4f}  (cap: ${BUDGET_CAP[m]})")
    print(f"  {'TOTAL':25s}  ${total:.4f}")

    print("\nDECISION RULES")

    # RNG probe passes if >=3/5 models show P(mode) > 0.5 on any condition.
    rng_pass = 0
    for m in MODELS:
        m_rows = [r for r in all_rng if r["model"] == m and r.get("p_mode") is not None]
        if any(r["p_mode"] > 0.5 for r in m_rows):
            rng_pass += 1
    print(f"\n  RNG probe: {rng_pass}/5 models with P(mode)>0.5 on any condition -> {'PASS' if rng_pass >= 3 else 'FAIL'}")

    # Seven-target passes if >=3/5 models show P(mode)>=0.90 on >=5/7 targets.
    seven_pass = 0
    for m in MODELS:
        m_rows = [r for r in all_seven if r["model"] == m]
        n_collapsed = sum(1 for r in m_rows if r["p_mode"] >= 0.90)
        if n_collapsed >= 5:
            seven_pass += 1
    print(f"  Seven-target: {seven_pass}/5 models with P(mode)>=0.90 on >=5/7 -> {'PASS' if seven_pass >= 3 else 'FAIL'}")

    # KNOWS/DOES passes if >=3/5 models: V0/V7 TV>0.20 on non-uniform AND V8/V9 TV<=0.10.
    kd_pass = 0
    for m in MODELS:
        m_rows = [r for r in all_kd if r["model"] == m]
        does_fail = all(r["tv"] > 0.20 for r in m_rows
                         if r["variant"] in ("V0", "V7") and r["task"] != "uniform_digit")
        knows_pass = all(r["tv"] <= 0.10 for r in m_rows
                          if r["variant"] in ("V8", "V9"))
        if does_fail and knows_pass:
            kd_pass += 1
    print(f"  KNOWS/DOES: {kd_pass}/5 models -> {'PASS' if kd_pass >= 3 else 'FAIL'}")

    meta = {"timestamp": datetime.utcnow().isoformat() + "Z",
            "models": dict(MODELS), "n_samples": N_SAMPLES,
            "cost": dict(cost_tracker)}
    json.dump(meta, open(out_dir / "run_meta.json", "w"), indent=2)
    print("\nDone.")


# ---- pilot ----

PILOTS = {
    "P1_rng_int100": {
        "n": 10,
        "messages": [{"role": "user", "content": "Pick a random integer uniformly between 1 and 100. Output only the integer."}],
        "max_tokens": 80,
        "temperature": 1.0,
    },
    "P2_seven_target_skewed_binary": {
        "n": 5,
        "messages": [{"role": "user", "content": "Output 'alpha' with probability 70% and 'beta' with probability 30%. Output one of: alpha, beta. Reply with just the word."}],
        "max_tokens": 80,
        "temperature": 1.0,
    },
    "P3_v0_uniform_digit": {
        "n": 5,
        "messages": [{"role": "user", "content": "Pick a random digit from 1 to 5 uniformly at random. Reply with just the digit."}],
        "max_tokens": 80,
        "temperature": 1.0,
    },
    "P4_v7_cot_uniform": {
        "n": 5,
        "messages": [
            {"role": "system", "content": (
                "For each request to pick a random digit from 1 to 5, follow this inverse-CDF procedure "
                "and output exactly three lines:\n"
                "Line 1: 'F=F(1)=0.20, F(2)=0.40, F(3)=0.60, F(4)=0.80, F(5)=1.00'\n"
                "Line 2: 'u=<the random number you drew, a decimal in [0, 1]>'\n"
                "Line 3: 'answer=<the digit you chose>'\n"
                "Do not include any other text."
            )},
            {"role": "user", "content": "Pick a random digit from 1 to 5 uniformly at random."},
        ],
        "max_tokens": 120,
        "temperature": 1.0,
    },
    "P5_v8_list5": {
        "n": 5,
        "messages": [{"role": "user", "content": (
            "Generate a list of 5 answers reflecting a uniform distribution over digits 1 to 5 "
            "(each digit about 20%). Output as JSON with key 'samples' whose value is a list of 5 strings."
        )}],
        "max_tokens": 120,
        "temperature": 1.0,
    },
}

PILOT_PARSERS = {
    "P1_rng_int100": parse_int_100,
    "P2_seven_target_skewed_binary": parse_alpha_beta,
    "P3_v0_uniform_digit": parse_digit,
    "P4_v7_cot_uniform": parse_digit,
    "P5_v8_list5": lambda text: text,  # only checked for JSON-parseability downstream
}


def pilot_call_once(client, model_id, messages, max_tokens, temperature, extra_body=None):
    kwargs = dict(
        model=model_id,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if extra_body:
        kwargs["extra_body"] = extra_body
    for attempt in range(5):
        try:
            r = client.chat.completions.create(**kwargs)
            text = r.choices[0].message.content or ""
            usage = r.usage
            tokens_in = usage.prompt_tokens if usage else 0
            tokens_out = usage.completion_tokens if usage else 0
            return {"text": text, "tokens_in": tokens_in, "tokens_out": tokens_out,
                     "finish_reason": r.choices[0].finish_reason}
        except Exception as e:
            if attempt == 4:
                return {"text": None, "tokens_in": 0, "tokens_out": 0,
                         "error": str(e), "finish_reason": "error"}
            wait = min(2 ** attempt, 30)
            print(f"    retry {attempt+1} in {wait}s: {e}")
            time.sleep(wait)


def run_pilot_for_model(raw_root, model_short, model_id):
    client = make_client()
    extra = extra_body_for(model_short)
    model_dir = raw_root / model_short
    model_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for pilot_name, cfg in PILOTS.items():
        print(f"  {pilot_name} (n={cfg['n']})...", end=" ", flush=True)
        outputs = []
        for i in range(cfg["n"]):
            r = pilot_call_once(client, model_id, cfg["messages"],
                                cfg["max_tokens"], cfg["temperature"], extra)
            parsed = PILOT_PARSERS[pilot_name](r["text"]) if r["text"] else None
            outputs.append({**r, "parsed": parsed, "call_id": i})

        with open(model_dir / f"{pilot_name}_raw.jsonl", "w") as f:
            for o in outputs:
                f.write(json.dumps(o, default=str) + "\n")

        tokens_out = [o["tokens_out"] for o in outputs if o["tokens_out"]]
        mean_tok = sum(tokens_out) / len(tokens_out) if tokens_out else 0
        n_parsed = sum(1 for o in outputs if o["parsed"] is not None)
        parse_rate = n_parsed / len(outputs) if outputs else 0
        errors = [o for o in outputs if o.get("error")]
        parsed_vals = [o["parsed"] for o in outputs if o["parsed"] is not None]

        u_extracted = 0
        if pilot_name == "P4_v7_cot_uniform":
            for o in outputs:
                if parse_u_value(o["text"]) is not None:
                    u_extracted += 1

        results[pilot_name] = {
            "n": len(outputs), "mean_tokens_out": round(mean_tok, 1),
            "parse_rate": round(parse_rate, 3), "n_errors": len(errors),
            "parsed_values": parsed_vals[:10],
            "u_extracted": u_extracted if pilot_name == "P4_v7_cot_uniform" else None,
        }
        print(f"mean_tok={mean_tok:.0f}  parse={parse_rate:.0%}  errors={len(errors)}")

    return results


def run_pilot():
    out_dir = results_dir("cross_family")
    raw_root = results_dir("cross_family/raw")
    print(f"Cross-family pilot — {len(MODELS)} models x 30 calls each\n")

    all_results = {}
    for model_short, model_id in MODELS.items():
        print(f"\nMODEL: {model_short} ({model_id})")
        try:
            r = run_pilot_for_model(raw_root, model_short, model_id)
            all_results[model_short] = {"status": "ok", "pilots": r}
        except Exception as e:
            print(f"  FATAL ERROR: {e}")
            all_results[model_short] = {"status": "error", "error": str(e)}

    print("\nPILOT SUMMARY")

    md = ["# Pilot Summary\n\n"]
    md.append("| Model | Mean tok/call | Parse rate | Errors | V7 u-extract | Status |\n")
    md.append("|---|---|---|---|---|---|\n")

    for model_short in MODELS:
        r = all_results.get(model_short, {})
        if r.get("status") == "error":
            md.append(f"| {model_short} | - | - | - | - | **ERROR**: {r['error'][:60]} |\n")
            print(f"  {model_short}: ERROR — {r['error'][:80]}")
            continue
        pilots = r["pilots"]
        all_tok = []
        all_parse = []
        all_err = 0
        u_ext = None
        for pname, pdata in pilots.items():
            all_tok.append(pdata["mean_tokens_out"])
            all_parse.append(pdata["parse_rate"])
            all_err += pdata["n_errors"]
            if pdata["u_extracted"] is not None:
                u_ext = pdata["u_extracted"]
        mean_tok = sum(all_tok) / len(all_tok) if all_tok else 0
        mean_parse = sum(all_parse) / len(all_parse) if all_parse else 0

        monologue = any(t > 50 for t in all_tok[:3])  # P1/P2/P3 should be <50 tokens
        low_parse = mean_parse < 0.90
        if monologue:
            status = "MONOLOGUE — investigate"
        elif low_parse:
            status = "FIX_PARSER"
        else:
            status = "PASS"

        u_str = f"{u_ext}/5" if u_ext is not None else "-"
        md.append(f"| {model_short} | {mean_tok:.0f} | {mean_parse:.0%} | {all_err} | {u_str} | {status} |\n")
        print(f"  {model_short}: tok={mean_tok:.0f}  parse={mean_parse:.0%}  errors={all_err}  "
              f"u_ext={u_str}  -> {status}")

        for pname, pdata in pilots.items():
            vals = pdata["parsed_values"][:5]
            print(f"    {pname}: {vals}")

    md.append("\n## Anomalies\n\n")
    anomalies = []
    for model_short in MODELS:
        r = all_results.get(model_short, {})
        if r.get("status") == "error":
            anomalies.append(f"- {model_short}: fatal error — {r['error'][:100]}")
        elif r.get("status") == "ok":
            pilots = r["pilots"]
            for pname, pdata in pilots.items():
                if pdata["mean_tokens_out"] > 50 and pname in ["P1_rng_int100", "P2_seven_target_skewed_binary", "P3_v0_uniform_digit"]:
                    anomalies.append(f"- {model_short} / {pname}: mean {pdata['mean_tokens_out']:.0f} tokens — monologuing?")
                if pdata["parse_rate"] < 0.80:
                    anomalies.append(f"- {model_short} / {pname}: parse rate {pdata['parse_rate']:.0%}")
    if anomalies:
        md.extend(a + "\n" for a in anomalies)
    else:
        md.append("None.\n")

    md.append("\n## Decision\n\n")
    for model_short in MODELS:
        r = all_results.get(model_short, {})
        if r.get("status") == "error":
            md.append(f"- **{model_short}**: ABORT (fatal error)\n")
        elif r.get("status") == "ok":
            pilots = r["pilots"]
            mean_tok_simple = sum(pilots[p]["mean_tokens_out"] for p in ["P1_rng_int100", "P2_seven_target_skewed_binary", "P3_v0_uniform_digit"]) / 3
            mean_parse = sum(pilots[p]["parse_rate"] for p in pilots) / len(pilots)
            if mean_tok_simple > 50:
                md.append(f"- **{model_short}**: FIX (monologuing on simple tasks)\n")
            elif mean_parse < 0.80:
                md.append(f"- **{model_short}**: FIX_PARSER then proceed\n")
            else:
                md.append(f"- **{model_short}**: proceed to full\n")

    with open(out_dir / "pilot_summary.md", "w") as f:
        f.writelines(md)
    print(f"\nWrote {out_dir / 'pilot_summary.md'}")

    json.dump(all_results, open(out_dir / "pilot_results.json", "w"), indent=2, default=str)

    if all(r.get("status") == "error" for r in all_results.values()):
        raise SystemExit("pilot failed on every model; see the errors above.")


def main(args):
    if getattr(args, "pilot", False):
        run_pilot()
    else:
        run_full()
