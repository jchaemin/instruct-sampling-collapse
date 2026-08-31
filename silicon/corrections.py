"""Correction strategies on synthetic sampling tasks: can prompting repair
per-call sampling?

Default variant — V1-V6 on uniform_digit and coin_flip:
  V1 verbatim measured-behavior feedback (the variant reported in the paper)
  V2 apologetic framing            V3 third-person framing
  V4 explicit bias framing         V5 20-shot ICL of corrected behavior
  V6 reasoning-chain prompting
  Task parameters + ICL sequences: correction_strategies_prompts.json.
  Per (task, variant): N behavioral samples at T=1.0 plus one T=0 logit read.

--extended — the same V1-V6 templates instantiated on three non-uniform
  targets (skewed_binary 0.7/0.3, bimodal_5way, skewed_5way), 200 calls per
  (task, variant).

--stronger — V7-V9 on uniform_digit and coin_flip:
  V7 chain-of-thought with explicit per-outcome probability computation
  V8 list-of-20 outputs in one call, sampled by the caller
  V9 JSON sampler spec {distribution: [...]}, sampled externally with NumPy
     (the paper's prescription reframed as a correction for fair comparison)

Outputs: results/corrections/ (correction_results.csv + correction_run_meta.json;
extended_*.json + extended_v1_v6_summary.csv; stronger_correction.csv +
stronger_run_meta.json).
Run: OPENAI_API_KEY=... python run.py corrections [--extended | --stronger]
"""
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from silicon import results_dir, require_env

HERE = Path(__file__).parent

client = None
MODEL = os.environ.get("SIM_MODEL", "gpt-4o")
N_THREADS = int(os.environ.get("SIM_THREADS", "16"))
N_SAMPLES = int(os.environ.get("CORRECTION_N", "200"))

CONFIG = json.load(open(HERE / "correction_strategies_prompts.json"))


def _ensure_client():
    global client
    if client is None:
        from openai import OpenAI
        client = OpenAI(api_key=require_env("OPENAI_API_KEY"))


def call_text(messages, temperature=1.0, max_tokens=10):
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=MODEL, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
            return r.choices[0].message.content
        except Exception:
            if attempt == 3: raise
            time.sleep(2 ** attempt)


def call_logprobs(messages, max_tokens=1):
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=MODEL, messages=messages,
                temperature=0.0, max_tokens=max_tokens,
                logprobs=True, top_logprobs=20,
            )
            top = r.choices[0].logprobs.content[0].top_logprobs
            return {t.token: t.logprob for t in top}
        except Exception:
            if attempt == 3: raise
            time.sleep(2 ** attempt)


def call_json(messages, max_tokens=200):
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=MODEL, messages=messages,
                temperature=0.0, max_tokens=max_tokens,
                response_format={"type": "json_object"})
            return r.choices[0].message.content
        except Exception:
            if attempt == 3: raise
            time.sleep(2 ** attempt)


# ---- V1-V6 system-prompt builders (parametrized over task) ----

def variant_V1_verbatim(task):
    return (f'I have measured your actual behavior on the prompt "{task["base_user_prompt"]}". '
            f'Across 200 runs, you produced: {task["baseline_behavior_str"]}. This is far from '
            f'the target. For this task, please correct: produce a {task["target_phrase"]}.')


def variant_V2_apologetic(task):
    return ("I'm sorry for the confusion in our previous conversation. I realize my earlier "
            "instructions may not have been clear. Let me try again with clearer guidance. "
            f'When you respond to the prompt "{task["base_user_prompt"]}", I would like you to '
            f'produce a truly {task["target_outcome_short"]}.')


def variant_V3_third_person(task):
    return (f'The assistant has been producing {task["mode_label"]} in approximately '
            f'{task["mode_pct"]} of trials on the prompt "{task["base_user_prompt"]}". '
            f'The target behavior is a {task["target_outcome_short"]}. The assistant should '
            f'now produce outputs consistent with this target.')


def variant_V4_explicit_bias(task):
    return (f'You have a known bias toward producing {task["mode_label"]} when asked to '
            f'respond to "{task["base_user_prompt"]}". This is a documented failure mode in '
            f'language models. Please consciously overcome this bias on the following task '
            f'and produce a {task["target_phrase"]}.')


def variant_V5_icl(task):
    """V5 system prompt; the ICL turns are added in the message builder."""
    if task is CONFIG["tasks"]["uniform_digit"]:
        return ("Below are examples of correctly producing uniformly-random digits from 1 to 5. "
                "Each subsequent response should be drawn from the same uniform distribution.")
    else:
        return ("Below are examples of correctly producing fair coin flips. Each subsequent "
                "response should be drawn from the same fair-coin distribution.")


def variant_V6_reasoning(task):
    if task is CONFIG["tasks"]["uniform_digit"]:
        return ("When you respond to the next prompt, follow these steps:\n"
                "Step 1: Acknowledge that uniform random sampling means each outcome has equal probability.\n"
                "Step 2: Identify what the support of the distribution is (digits 1, 2, 3, 4, 5).\n"
                "Step 3: Mentally simulate a process that produces each outcome with probability 1/5.\n"
                "Step 4: Report the outcome of that simulation as your response.")
    else:
        return ("When you respond to the next prompt, follow these steps:\n"
                "Step 1: Acknowledge that a fair coin has two outcomes with equal probability.\n"
                "Step 2: Identify the support (heads, tails).\n"
                "Step 3: Mentally simulate a process that produces each outcome with probability 1/2.\n"
                "Step 4: Report the outcome of that simulation as your response.")


VARIANT_BUILDERS = {
    "V1_verbatim":      variant_V1_verbatim,
    "V2_apologetic":    variant_V2_apologetic,
    "V3_third_person":  variant_V3_third_person,
    "V4_explicit_bias": variant_V4_explicit_bias,
    "V5_icl":           variant_V5_icl,
    "V6_reasoning":     variant_V6_reasoning,
}


def build_messages(variant_id, system_text, task, user_prompt_override=None):
    user_prompt = user_prompt_override or task["base_user_prompt"]
    if variant_id == "V5_icl":
        task_name = ("uniform_digit" if task is CONFIG["tasks"]["uniform_digit"] else "coin_flip")
        icl = CONFIG["icl_examples"][task_name]
        msgs = [{"role": "system", "content": system_text}]
        for ex in icl:
            msgs.append({"role": "user", "content": task["base_user_prompt"]})
            msgs.append({"role": "assistant", "content": str(ex)})
        msgs.append({"role": "user", "content": user_prompt})
        return msgs
    elif variant_id == "V6_reasoning":
        msgs = [{"role": "system", "content": system_text}]
        msgs.append({"role": "user",
                     "content": (task["base_user_prompt"] +
                                 " Apply the steps above silently and respond with only the result.")})
        return msgs
    else:
        return [{"role": "system", "content": system_text},
                {"role": "user", "content": user_prompt}]


# ---- parsing and metrics ----

def parse_uniform_digit(text):
    if text is None: return None
    m = re.search(r'[1-5]', text)
    return int(m.group()) if m else None


def parse_coin_flip(text):
    if text is None: return None
    t = text.strip().lower()
    if t.startswith("h") or "heads" in t: return "heads"
    if t.startswith("t") or "tails" in t: return "tails"
    return None


def parse_for_task(task_name, text):
    if task_name == "uniform_digit":
        return parse_uniform_digit(text)
    elif task_name == "coin_flip":
        return parse_coin_flip(text)
    return None


def empirical(samples, support):
    counts = {s: 0 for s in support}
    n = 0
    for v in samples:
        if v in counts:
            counts[v] += 1
            n += 1
    if n == 0: return {s: 0.0 for s in support}, 0
    return {s: counts[s] / n for s in support}, n


def tv(p_hat, target, support):
    p = np.array([p_hat[s] for s in support])
    q = np.array(target)
    return float(0.5 * np.abs(p - q).sum())


def kl(p_hat, target, support, eps=1e-6):
    p = np.array([p_hat[s] for s in support])
    q = np.array(target) + eps
    q = q / q.sum()
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def match_support_logprob(token, target):
    """Loose token match against a target element."""
    t = str(token).strip().lower()
    tg = str(target).strip().lower()
    return t == tg or t.startswith(tg) or tg.startswith(t)


def extract_support_logprobs(logp_dict, support):
    out = {s: float("-inf") for s in support}
    for tok, lp in logp_dict.items():
        for s in support:
            if match_support_logprob(tok, s):
                out[s] = max(out[s], lp)
    return out


# ---- default variant: V1-V6 with logit reads ----

def run_pair(task_name, variant_id):
    task = CONFIG["tasks"][task_name]
    system_text = VARIANT_BUILDERS[variant_id](task)

    def worker(_):
        msgs = build_messages(variant_id, system_text, task)
        text = call_text(msgs, temperature=1.0, max_tokens=12)
        return parse_for_task(task_name, text)

    samples = []
    with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
        futs = [pool.submit(worker, i) for i in range(N_SAMPLES)]
        for f in as_completed(futs):
            samples.append(f.result())

    p_hat, n_valid = empirical(samples, task["support"])

    msgs = build_messages(variant_id, system_text, task)
    logp = call_logprobs(msgs, max_tokens=1)
    support_lp = extract_support_logprobs(logp, task["support"])

    raw_p = {s: (np.exp(lp) if lp != float("-inf") else 0.0) for s, lp in support_lp.items()}
    total = sum(raw_p.values())
    if total > 0:
        renorm_p = {s: raw_p[s] / total for s in raw_p}
    else:
        renorm_p = raw_p

    return {
        "task": task_name, "variant": variant_id,
        "n_valid": n_valid,
        "p_hat": p_hat,
        "tv_to_target": tv(p_hat, task["target_dist"], task["support"]),
        "kl_to_target": kl(p_hat, task["target_dist"], task["support"]),
        "p_mode": max(p_hat.values()),
        "mode_elem": str(max(p_hat, key=p_hat.get)),
        "T0_logprobs_raw": {str(s): support_lp[s] for s in task["support"]},
        "T0_renorm_p":     {str(s): renorm_p[s]   for s in task["support"]},
        "T0_p_mode":       max(renorm_p.values()),
    }


def run_base(out_dir):
    print(f"Model: {MODEL}  Threads: {N_THREADS}  N per (task,variant): {N_SAMPLES}\n")

    rows = []
    variant_ids = list(VARIANT_BUILDERS.keys())
    task_names = list(CONFIG["tasks"].keys())

    for task_name in task_names:
        for variant_id in variant_ids:
            print(f"{task_name} / {variant_id}")
            r = run_pair(task_name, variant_id)
            print(f"  p_hat = {r['p_hat']}")
            print(f"  TV={r['tv_to_target']:.3f}  KL={r['kl_to_target']:.3f}  "
                  f"P(mode)={r['p_mode']:.3f}  mode={r['mode_elem']}  n_valid={r['n_valid']}")
            print(f"  T=0 renorm P|support = {r['T0_renorm_p']}  P(mode)|T=0 = {r['T0_p_mode']:.4f}")
            rows.append({
                "task": r["task"], "variant": r["variant"],
                "n_valid": r["n_valid"], "p_hat": json.dumps(r["p_hat"]),
                "tv_to_target": r["tv_to_target"], "kl_to_target": r["kl_to_target"],
                "p_mode": r["p_mode"], "mode_elem": r["mode_elem"],
                "T0_renorm_p": json.dumps(r["T0_renorm_p"]),
                "T0_p_mode": r["T0_p_mode"],
                "T0_logprobs_raw": json.dumps(r["T0_logprobs_raw"]),
            })
            pd.DataFrame(rows).to_csv(out_dir / "correction_results.csv", index=False)
        print()

    df = pd.DataFrame(rows)
    print(f"Behavioral results (N={N_SAMPLES})")
    pd.options.display.float_format = "{:.3f}".format
    print(df[["task", "variant", "n_valid", "p_mode", "tv_to_target", "kl_to_target", "mode_elem"]].to_string(index=False))

    print("\nT=0 within-support renormalized P(mode)")
    print(df[["task", "variant", "T0_p_mode"]].to_string(index=False))

    import openai
    meta = {"timestamp": datetime.utcnow().isoformat() + "Z",
            "openai_version": openai.__version__, "model_string": MODEL,
            "n_samples": N_SAMPLES, "n_threads": N_THREADS,
            "variants": variant_ids, "tasks": task_names}
    with open(out_dir / "correction_run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nWrote {out_dir / 'correction_results.csv'} and correction_run_meta.json")


# ---- extended variant: V1-V6 on three non-uniform targets ----

# Task specs: the V0 prompts plus the measured baseline behavior each variant quotes.
EXTENDED_TASKS = {
    "skewed_binary": {
        "base_user_prompt": "Output 'alpha' with probability 70% and 'beta' with probability 30%. Output one of: alpha, beta. Reply with just the word.",
        "target_dist": [0.7, 0.3],
        "support": ["alpha", "beta"],
        "baseline_behavior_str": "alpha: 100%, beta: 0%",
        "mode_label": "alpha",
        "mode_pct": "100%",
        "target_phrase": "[alpha, beta]-valued response so that across many responses you give alpha about 70% of the time and beta about 30% of the time",
        "target_outcome_short": "skewed binary (alpha about 70%, beta about 30%)",
        "icl_examples": ["alpha", "beta", "alpha", "alpha", "beta", "alpha", "alpha", "alpha",
                          "beta", "alpha", "alpha", "beta", "alpha", "alpha", "alpha", "beta",
                          "alpha", "alpha", "alpha", "beta"],  # 14 alpha, 6 beta = 70/30
        "v6_reasoning_lines": [
            "Step 1: Acknowledge that the target distribution is alpha with probability 0.70 and beta with probability 0.30.",
            "Step 2: Identify the support (alpha, beta).",
            "Step 3: Mentally simulate a process that produces alpha with probability 0.70 and beta with probability 0.30.",
            "Step 4: Report the outcome of that simulation as your response.",
        ],
    },
    "bimodal_5way": {
        "base_user_prompt": ("Output a digit from 1 to 5 where 1 appears 40% of the time, "
                              "2 appears 10%, 3 appears 0%, 4 appears 10%, and 5 appears 40%. "
                              "Reply with just the digit."),
        "target_dist": [0.40, 0.10, 0.00, 0.10, 0.40],
        "support": [1, 2, 3, 4, 5],
        "baseline_behavior_str": "digit 1: 100%, others: 0%",
        "mode_label": "digit 1",
        "mode_pct": "100%",
        "target_phrase": "digit drawn from [P(1)=0.40, P(2)=0.10, P(3)=0.00, P(4)=0.10, P(5)=0.40]",
        "target_outcome_short": "bimodal on {1,2,4,5} with peaks at 1 and 5 (40% each); 3 is impossible",
        "icl_examples": [1, 5, 4, 1, 5, 1, 5, 2, 1, 5, 1, 5, 4, 5, 1, 5, 5, 1, 2, 5],
        "v6_reasoning_lines": [
            "Step 1: Acknowledge that the target distribution is P(1)=0.40, P(2)=0.10, P(3)=0.00, P(4)=0.10, P(5)=0.40.",
            "Step 2: Identify the support (1, 2, 4, 5) — digit 3 has zero mass and must never be produced.",
            "Step 3: Mentally simulate a process that produces those frequencies.",
            "Step 4: Report the outcome of that simulation as your response.",
        ],
    },
    "skewed_5way": {
        "base_user_prompt": ("Output a digit from 1 to 5 where 1 appears 50%, 2 appears 20%, "
                              "3 appears 10%, 4 appears 10%, 5 appears 10%. Reply with just the digit."),
        "target_dist": [0.50, 0.20, 0.10, 0.10, 0.10],
        "support": [1, 2, 3, 4, 5],
        "baseline_behavior_str": "digit 1: 100%, others: 0%",
        "mode_label": "digit 1",
        "mode_pct": "100%",
        "target_phrase": "digit drawn from [P(1)=0.50, P(2)=0.20, P(3)=0.10, P(4)=0.10, P(5)=0.10]",
        "target_outcome_short": "skewed 5-way (1 dominant at 50%, others 20/10/10/10)",
        "icl_examples": [1, 1, 2, 1, 3, 1, 2, 4, 1, 1, 5, 2, 1, 1, 3, 2, 1, 4, 1, 5],
        "v6_reasoning_lines": [
            "Step 1: Acknowledge that the target distribution is P(1)=0.50, P(2)=0.20, P(3)=0.10, P(4)=0.10, P(5)=0.10.",
            "Step 2: Identify the support (1, 2, 3, 4, 5).",
            "Step 3: Mentally simulate a process that produces those frequencies.",
            "Step 4: Report the outcome of that simulation as your response.",
        ],
    },
}


# Builders below instantiate the same V1-V4 templates as VARIANT_BUILDERS,
# with a generic V5 system prompt and per-task V6 reasoning lines.

def ext_V1(t):
    return (f'I have measured your actual behavior on the prompt "{t["base_user_prompt"]}". '
            f'Across 200 runs, you produced: {t["baseline_behavior_str"]}. This is far from '
            f'the target. For this task, please correct: produce a {t["target_phrase"]}.')


def ext_V2(t):
    return ("I'm sorry for the confusion in our previous conversation. I realize my earlier "
            "instructions may not have been clear. Let me try again with clearer guidance. "
            f'When you respond to the prompt "{t["base_user_prompt"]}", I would like you to '
            f'produce a truly {t["target_outcome_short"]}.')


def ext_V3(t):
    return (f'The assistant has been producing {t["mode_label"]} in approximately '
            f'{t["mode_pct"]} of trials on the prompt "{t["base_user_prompt"]}". '
            f'The target behavior is a {t["target_outcome_short"]}. The assistant should '
            f'now produce outputs consistent with this target.')


def ext_V4(t):
    return (f'You have a known bias toward producing {t["mode_label"]} when asked to '
            f'respond to "{t["base_user_prompt"]}". This is a documented failure mode in '
            f'language models. Please consciously overcome this bias on the following task '
            f'and produce a {t["target_phrase"]}.')


def ext_V5(t):
    return ("Below are examples of correctly producing samples from the target distribution. "
            "Each subsequent response should be drawn from the same distribution.")


def ext_V6(t):
    return ("When you respond to the next prompt, follow these steps:\n" +
            "\n".join(t["v6_reasoning_lines"]))


EXTENDED_BUILDERS = {"V1_verbatim": ext_V1, "V2_apologetic": ext_V2, "V3_third_person": ext_V3,
                     "V4_explicit_bias": ext_V4, "V5_icl": ext_V5, "V6_reasoning": ext_V6}


def ext_build(variant, system, task):
    if variant == "V5_icl":
        msgs = [{"role": "system", "content": system}]
        for ex in task["icl_examples"]:
            msgs.append({"role": "user", "content": task["base_user_prompt"]})
            msgs.append({"role": "assistant", "content": str(ex)})
        msgs.append({"role": "user", "content": task["base_user_prompt"]})
        return msgs
    if variant == "V6_reasoning":
        return [{"role": "system", "content": system},
                 {"role": "user", "content": task["base_user_prompt"] +
                                              " Apply the steps above silently and respond with only the result."}]
    return [{"role": "system", "content": system},
             {"role": "user", "content": task["base_user_prompt"]}]


def ext_parse_for_task(task_name, text):
    if text is None: return None
    t = text.strip().lower()
    if task_name == "skewed_binary":
        if t.startswith("a") or "alpha" in t: return "alpha"
        if t.startswith("b") or "beta"  in t: return "beta"
        return None
    m = re.search(r"[1-5]", t)
    return int(m.group()) if m else None


def ext_empirical(samples, support):
    counts = {s: 0 for s in support}
    n = 0
    for v in samples:
        if v in counts:
            counts[v] += 1; n += 1
    if n == 0: return [0.0] * len(support), 0
    return [counts[s] / n for s in support], n


def ext_tv(p, q):
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


def ext_run_cell(out_dir, task_name, task, variant):
    out_path = out_dir / f"extended_{task_name}_{variant}.json"
    if out_path.exists():
        return json.load(open(out_path))
    system = EXTENDED_BUILDERS[variant](task)
    def worker(_):
        msgs = ext_build(variant, system, task)
        text = call_text(msgs, max_tokens=12)
        return ext_parse_for_task(task_name, text)
    with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
        samples = list(pool.map(worker, range(N_SAMPLES)))
    emp, n_valid = ext_empirical(samples, task["support"])
    tv_val = ext_tv(emp, task["target_dist"])
    mode_idx = int(np.argmax(emp))
    p_mode = float(max(emp))
    summary = {"task": task_name, "variant": variant, "n_valid": n_valid,
                "p_hat": dict(zip([str(s) for s in task["support"]], emp)),
                "tv_to_target": tv_val, "p_mode": p_mode,
                "mode_elem": str(task["support"][mode_idx]),
                "system_prompt": system}
    json.dump(summary, open(out_path, "w"), indent=2)
    return summary


def run_extended(out_dir):
    print(f"Model: {MODEL}  N: {N_SAMPLES}  threads: {N_THREADS}")
    rows = []
    for task_name, task in EXTENDED_TASKS.items():
        print(f"\nTask: {task_name} (target={task['target_dist']})")
        for variant in EXTENDED_BUILDERS:
            s = ext_run_cell(out_dir, task_name, task, variant)
            print(f"  {variant:18s} p_hat={ {k: round(v, 3) for k, v in s['p_hat'].items()} } "
                   f"TV={s['tv_to_target']:.3f} P(mode={s['mode_elem']})={s['p_mode']:.3f}")
            rows.append(s)

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "system_prompt"} for r in rows])
    df.to_csv(out_dir / "extended_v1_v6_summary.csv", index=False)
    print("\nSummary (TV)")
    pivot = df.pivot(index="variant", columns="task", values="tv_to_target")
    print(pivot.round(3).to_string())


# ---- stronger variant: V7-V9 ----

STRONGER_TASKS = {
    "uniform_digit": {
        "support": ["1", "2", "3", "4", "5"],
        "target": [0.2, 0.2, 0.2, 0.2, 0.2],
        "user_prompt": "Pick a random digit from 1 to 5 uniformly at random. Reply with just the digit.",
        "target_str": "uniform random digit (each digit 1-5 about 20% of the time)",
    },
    "coin_flip": {
        "support": ["heads", "tails"],
        "target": [0.5, 0.5],
        "user_prompt": "Flip a fair coin. Report heads or tails. Reply with just the word.",
        "target_str": "fair coin outcome (heads about 50%, tails about 50%)",
    },
}


def variant_V7_cot_explicit_prob(task_name, task):
    """CoT with explicit per-outcome probability computation; one sample per call."""
    if task_name == "uniform_digit":
        system = ("For each request to pick a random digit from 1 to 5, think through "
                  "the following steps silently, then output only the final digit:\n"
                  "  Step 1: Acknowledge the target distribution. Uniform on {1,2,3,4,5} means "
                  "P(1)=0.2, P(2)=0.2, P(3)=0.2, P(4)=0.2, P(5)=0.2.\n"
                  "  Step 2: Compute the cumulative distribution: F(1)=0.2, F(2)=0.4, "
                  "F(3)=0.6, F(4)=0.8, F(5)=1.0.\n"
                  "  Step 3: Mentally draw a random number u in [0, 1].\n"
                  "  Step 4: Find the smallest digit d with F(d) >= u and output d.\n"
                  "Output only the final digit, nothing else.")
    else:
        system = ("For each request to flip a fair coin, think through the following "
                  "steps silently, then output only the final result:\n"
                  "  Step 1: Acknowledge the target distribution. Fair coin: P(heads)=0.5, P(tails)=0.5.\n"
                  "  Step 2: Mentally draw u in [0, 1].\n"
                  "  Step 3: If u < 0.5 output heads; otherwise output tails.\n"
                  "Output only the final result (heads or tails), nothing else.")

    def worker(_):
        text = call_text([{"role": "system", "content": system},
                          {"role": "user", "content": task["user_prompt"]}],
                          temperature=1.0, max_tokens=8)
        return parse_for_task(task_name, text)
    with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
        samples = list(pool.map(worker, range(N_SAMPLES)))
    return samples


def variant_V8_list_of_N(task_name, task, list_size=20):
    """One call produces a list of `list_size` outputs matching the target;
    the caller collects N_SAMPLES samples across ceil(N_SAMPLES/list_size) calls."""
    if task_name == "uniform_digit":
        ask_msg = (f"Produce a list of {list_size} digits where the digits 1-5 each appear "
                   f"about {list_size // 5} times (so the distribution across the {list_size} "
                   f"outputs is approximately uniform on 1-5). Output the {list_size} digits "
                   f"as a JSON array under the key 'samples'.")
    else:
        ask_msg = (f"Produce a list of {list_size} coin-flip outcomes (each is either 'heads' "
                   f"or 'tails') where heads appears about {list_size // 2} times and tails "
                   f"appears about {list_size // 2} times (so the distribution is approximately "
                   f"fair). Output the {list_size} outcomes as a JSON array under the key 'samples'.")

    samples = []
    n_calls = (N_SAMPLES + list_size - 1) // list_size

    def parse_list(json_text):
        try:
            d = json.loads(json_text)
            arr = d.get("samples") or list(d.values())[0]
            return arr
        except Exception:
            return []

    def worker(_):
        text = call_json([{"role": "user", "content": ask_msg}], max_tokens=500)
        arr = parse_list(text)
        out = []
        for a in arr:
            out.append(parse_for_task(task_name, str(a)))
        return out

    # Fewer threads: each call is larger than a single-sample call.
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(worker, range(n_calls)))
    for lst in results:
        samples.extend(lst)
    samples = samples[:N_SAMPLES]
    return samples


def variant_V9_sampler_spec(task_name, task):
    """One call returns a JSON {distribution: [...]}; sampled externally with NumPy."""
    if task_name == "uniform_digit":
        ask_msg = ("Output the probability distribution over digits 1 to 5 that represents "
                   "a uniform distribution. Output as JSON with key 'distribution' whose value "
                   "is a list of 5 probabilities [p1, p2, p3, p4, p5] summing to 1.0.")
    else:
        ask_msg = ("Output the probability distribution over {heads, tails} that represents "
                   "a fair coin. Output as JSON with key 'distribution' whose value is "
                   "a list of 2 probabilities [p_heads, p_tails] summing to 1.0.")

    text = call_json([{"role": "user", "content": ask_msg}], max_tokens=200)
    try:
        d = json.loads(text)
        vals = d.get("distribution") or list(d.values())[0]
        vals = [float(v) for v in vals]
        s = sum(vals)
        if s < 1e-9: return None, None
        dist = np.array(vals) / s
    except Exception:
        return None, None

    rng = np.random.default_rng(42)
    idx = rng.choice(len(task["support"]), size=N_SAMPLES, p=dist)
    samples = [task["support"][i] for i in idx]
    return samples, dist.tolist()


def stronger_empirical(samples, support):
    counts = {str(s): 0 for s in support}
    n = 0
    for v in samples:
        if v is not None:
            counts[str(v)] = counts.get(str(v), 0) + 1
            n += 1
    if n == 0: return {str(s): 0.0 for s in support}, 0
    return {s: counts[s] / n for s in counts}, n


def stronger_tv(p_dict, target, support):
    p = np.array([p_dict[str(s)] for s in support])
    return float(0.5 * np.abs(p - np.array(target)).sum())


def run_stronger(out_dir):
    print(f"Model: {MODEL}  Threads: {N_THREADS}  N: {N_SAMPLES}\n")
    rows = []
    for task_name, task in STRONGER_TASKS.items():
        print(f"Task: {task_name} (target={task['target']})")

        for variant_name, runner in [
            ("V7_CoT_explicit_prob", lambda: variant_V7_cot_explicit_prob(task_name, task)),
            ("V8_list_of_N",         lambda: variant_V8_list_of_N(task_name, task)),
        ]:
            samples = runner()
            p_hat, n_valid = stronger_empirical(samples, task["support"])
            tv_val = stronger_tv(p_hat, task["target"], task["support"])
            mode = max(p_hat, key=p_hat.get) if n_valid > 0 else None
            p_mode = p_hat[mode] if mode else 0.0
            print(f"  {variant_name:25s}  p_hat={ {k: round(v,3) for k,v in p_hat.items()} }  "
                  f"TV={tv_val:.3f}  mode={mode}  P(mode)={p_mode:.3f}  n_valid={n_valid}")
            rows.append({"task": task_name, "variant": variant_name,
                          "n_valid": n_valid, "p_hat": json.dumps(p_hat),
                          "tv_to_target": tv_val, "p_mode": p_mode,
                          "mode_elem": str(mode)})

        # V9 returns (samples, dist) and is handled separately.
        result = variant_V9_sampler_spec(task_name, task)
        if result is not None and result[0] is not None:
            samples, dist = result
            p_hat, n_valid = stronger_empirical(samples, task["support"])
            tv_val = stronger_tv(p_hat, task["target"], task["support"])
            mode = max(p_hat, key=p_hat.get) if n_valid > 0 else None
            p_mode = p_hat[mode] if mode else 0.0
            print(f"  V9_sampler_spec           model_dist={[round(d,3) for d in dist]}  empirical_p_hat={ {k: round(v,3) for k,v in p_hat.items()} }  "
                  f"TV={tv_val:.3f}  n_valid={n_valid}")
            rows.append({"task": task_name, "variant": "V9_sampler_spec",
                          "n_valid": n_valid, "p_hat": json.dumps(p_hat),
                          "tv_to_target": tv_val, "p_mode": p_mode,
                          "mode_elem": str(mode),
                          "model_specified_dist": json.dumps(dist)})
        print()

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "stronger_correction.csv", index=False)
    print("Summary")
    print(df[["task", "variant", "tv_to_target", "p_mode", "mode_elem"]].to_string(index=False))

    meta = {"timestamp": datetime.utcnow().isoformat() + "Z",
            "model_string": MODEL, "n_samples": N_SAMPLES}
    with open(out_dir / "stronger_run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def main(args):
    global MODEL
    if getattr(args, "model", None):
        MODEL = args.model
    _ensure_client()
    out_dir = results_dir("corrections")
    if getattr(args, "extended", False):
        run_extended(out_dir)
    elif getattr(args, "stronger", False):
        run_stronger(out_dir)
    else:
        run_base(out_dir)
