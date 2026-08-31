"""OpinionQA personas and the demographically-matched Argyle baseline (Appendix C).

This module provides the shared building blocks used across the OpinionQA
experiments -- the demographic persona pool (`sample_personas`), the 100-item
loader (`load_100_items`), and the per-item matched-persona sampler
(`sample_matched_personas`) -- and runs the matched-Argyle baseline of
Appendix C: for each item, 100 personas drawn from that item's actual Pew
respondents (weighted), one T=1.0 answer each, aggregated and compared to the
standard-Argyle and describe TVs shipped in per_item_100.csv.

Inputs: OpinionQA waves under data/human_resp/ (see README) and the shipped
data/ppa_per_item/per_item_100.csv. Output: results/opinionqa/matched/.
Run: OPENAI_API_KEY=... python run.py opinionqa [--model gpt-4o]
"""
import json
import os
import re
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from silicon import DATA, PROJECT, results_dir, require_env
from silicon.pew_items import ITEMS, _BASE, pew_demographic_proportions

client = None
MODEL = os.environ.get("SIM_MODEL", "gpt-4o")
N_THREADS = 16
N_PERSONAS = 100
SEED = 42

ITEMS_50_DIR = PROJECT / "paper_extensions/robustness/exp_e6_50items"


def _ensure_client():
    global client
    if client is None:
        from openai import OpenAI
        client = OpenAI(api_key=require_env("OPENAI_API_KEY"))


# ---- persona construction ----

PEW_TO_PROMPT_AGE = {
    "18-29": "in their 20s",
    "30-49": "in their 30s or 40s",
    "50-64": "in their 50s or early 60s",
    "65+": "in their late 60s or older",
}
PEW_TO_PROMPT_EDUC = {
    "H.S. graduate or less": "with a high-school education",
    "Some College": "with some college experience",
    "College graduate+": "with a college degree or more",
}
PEW_TO_PROMPT_RACE = {
    "White non-Hispanic": "non-Hispanic white",
    "Black non-Hispanic": "non-Hispanic Black",
    "Hispanic": "Hispanic",
    "Asian non-Hispanic": "non-Hispanic Asian American",
    "Other": "of another racial background",
}
PEW_TO_PROMPT_PARTY = {
    "Dem/Lean Dem": "Democrat or Democratic-leaning",
    "Rep/Lean Rep": "Republican or Republican-leaning",
    "DK/Refused/No lean": "politically independent",
}
PEW_TO_PROMPT_SEX = {
    "Female": "woman",
    "Male": "man",
    "In some other way": "person",
}
PEW_TO_PROMPT_INC = {
    "Lower income": "in a lower-income household",
    "Middle income": "in a middle-income household",
    "Upper income": "in an upper-income household",
}

PERSONA_COLS = ["AGE", "SEX", "F_RACETHNMOD", "F_EDUCCAT",
                "F_PARTYSUM_FINAL", "F_INC_TIER2"]


def _describe_row(r):
    """Turn a Pew respondent row into a one-sentence persona description."""
    gender = PEW_TO_PROMPT_SEX.get(r.get("SEX", ""), "person")
    age = PEW_TO_PROMPT_AGE.get(r.get("AGE", ""), r.get("AGE", ""))
    race = PEW_TO_PROMPT_RACE.get(r.get("F_RACETHNMOD", ""), r.get("F_RACETHNMOD", ""))
    educ = PEW_TO_PROMPT_EDUC.get(r.get("F_EDUCCAT", ""), r.get("F_EDUCCAT", ""))
    party = PEW_TO_PROMPT_PARTY.get(r.get("F_PARTYSUM_FINAL", ""), "")
    inc = PEW_TO_PROMPT_INC.get(r.get("F_INC_TIER2", ""), "")
    return f"You are a {race} {gender} {age} {educ}, {inc}, who is {party}."


def sample_personas(n, seed=42):
    """n personas drawn from Pew's national demographic marginals."""
    props = pew_demographic_proportions(PERSONA_COLS)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(props), size=n, replace=True, p=props["p"].values)
    personas = []
    for i, row_i in enumerate(idx):
        r = props.iloc[row_i]
        personas.append({"id": i, "descrip": _describe_row(r),
                         "demo": {c: r[c] for c in PERSONA_COLS}})
    return personas


def sample_matched_personas(item, n=100, seed=42):
    """n personas drawn from the Pew respondents who actually answered `item`, weighted."""
    path = os.path.join(_BASE, item["wave"], "responses.csv")
    df = pd.read_csv(path, low_memory=False)
    if item["key"] not in df.columns:
        return []
    sub = df[df[item["key"]].notna() & (df[item["key"]].astype(str).str.strip() != "")]
    wave_num = item["wave"].split("_W")[-1] if "_W" in item["wave"] else ""
    weight_col = f"WEIGHT_W{wave_num}"
    if weight_col in sub.columns:
        weights = sub[weight_col].fillna(1.0).values
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(sub)) / len(sub)
    rng = np.random.default_rng(seed + zlib.crc32(item["key"].encode()) % 100000)
    idx = rng.choice(len(sub), size=n, replace=True, p=weights)
    return [{"id": i, "descrip": _describe_row(sub.iloc[j])} for i, j in enumerate(idx)]


# ---- LLM + parsing helpers ----

def call_text(messages, max_tokens=10):
    for attempt in range(4):
        try:
            r = client.chat.completions.create(model=MODEL, messages=messages,
                                               temperature=1.0, max_tokens=max_tokens)
            return r.choices[0].message.content
        except Exception:
            if attempt == 3:
                return None
            time.sleep(2 ** attempt)


def parse_digit(text):
    if not text:
        return None
    m = re.search(r"[1-5]", text.strip())
    return int(m.group()) if m else None


def empirical(samples, k=5):
    counts = np.bincount([s for s in samples if s is not None], minlength=k + 1)[1:k + 1].astype(float)
    if counts.sum() == 0:
        return np.full(k, 1.0 / k)
    return counts / counts.sum()


def tv(p, q):
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


# ---- 100-item loader shared with the open-model modules ----

def load_100_items():
    """The 100 OpinionQA items (13 curated + 50 expansion + duplicates), each with its Pew target."""
    items = []
    for it in ITEMS:
        path = os.path.join(_BASE, it["wave"], "responses.csv")
        df = pd.read_csv(path, low_memory=False)
        if it["key"] not in df.columns:
            continue
        counts = np.zeros(5)
        for v in df[it["key"]].astype(str):
            v = v.strip()
            for i, opt in enumerate(it["likert_to_pew"]):
                if v == opt:
                    counts[i] += 1
                    break
        if counts.sum() < 10:
            continue
        items.append({"key": it["key"], "wave": it["wave"], "question": it["question"],
                      "options": it["likert_to_pew"], "pew_dist": (counts / counts.sum()).tolist()})
    for src in [ITEMS_50_DIR / "item_list.json", DATA / "item_list_v2.json"]:
        for d in json.load(open(src)).get("new_item_details", []):
            items.append({"key": d["key"], "wave": f"American_Trends_Panel_{d['wave']}",
                          "question": d["question"], "options": d["options_pew_order"],
                          "pew_dist": d["pew_dist"]})
    return items


# ---- matched-Argyle baseline (Appendix C) ----

def run_matched_argyle(items, matched_dir):
    per_item_100 = pd.read_csv(DATA / "ppa_per_item" / "per_item_100.csv")
    shipped = {r["item_key"]: r for _, r in per_item_100.iterrows()}

    rows = []
    for item in items:
        cache_path = matched_dir / f"{item['key']}_matched.json"
        if cache_path.exists():
            tv_matched = json.load(open(cache_path))["tv_matched"]
        else:
            personas = sample_matched_personas(item, n=100, seed=SEED)
            if len(personas) < 50:
                print(f"  {item['key']:25s}  SKIP (only {len(personas)} matched personas)")
                continue

            def worker(persona):
                options = "\n".join(f"{i+1}. {item['options'][i]}" for i in range(5))
                user = f"{item['question']}\n\nOptions:\n{options}\n\nRespond with just the number (1-5)."
                msgs = [{"role": "system", "content": persona["descrip"]},
                        {"role": "user", "content": user}]
                return parse_digit(call_text(msgs))

            with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
                answers = list(pool.map(worker, personas))
            p_hat = empirical(answers)
            tv_matched = tv(p_hat, item["pew_dist"])
            json.dump({"tv_matched": tv_matched, "p_hat": p_hat.tolist()}, open(cache_path, "w"))

        rows.append({"item_key": item["key"], "TV_matched_argyle": tv_matched,
                     "TV_standard_argyle": shipped.get(item["key"], {}).get("baseline_tv"),
                     "TV_describe": shipped.get(item["key"], {}).get("describe_tv")})

    df = pd.DataFrame(rows)
    df.to_csv(matched_dir / "matched_argyle_per_item.csv", index=False)

    from scipy.stats import wilcoxon
    valid = df.dropna(subset=["TV_matched_argyle", "TV_standard_argyle", "TV_describe"])
    print(f"\nValid items: {len(valid)}")
    print(f"  mean TV matched Argyle:  {valid['TV_matched_argyle'].mean():.4f}")
    print(f"  mean TV standard Argyle: {valid['TV_standard_argyle'].mean():.4f}")
    print(f"  mean TV describe:        {valid['TV_describe'].mean():.4f}")
    p_matched_vs_std = wilcoxon(valid["TV_matched_argyle"] - valid["TV_standard_argyle"]).pvalue
    p_describe_vs_matched = wilcoxon(valid["TV_describe"] - valid["TV_matched_argyle"]).pvalue
    print(f"  matched vs standard Argyle: Wilcoxon p = {p_matched_vs_std:.4g}")
    print(f"  describe vs matched Argyle: Wilcoxon p = {p_describe_vs_matched:.4g}")
    return df


def main(args):
    global MODEL
    if getattr(args, "model", None):
        MODEL = args.model
    _ensure_client()
    items = load_100_items()
    print(f"Loaded {len(items)} items")
    run_matched_argyle(items, results_dir("opinionqa/matched"))
    print("\nDONE")
