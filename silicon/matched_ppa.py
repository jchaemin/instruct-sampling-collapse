"""Matched-Argyle vs matched-Argyle+PPA, on local open models or the API model.

Reconstructs the per-item demographically-matched personas exactly
(silicon.opinionqa.sample_matched_personas, seed 42 + per-item offset,
deterministic) and runs both the fixed-prompt matched pipeline and
matched+PPA over the 100 items: 100 calls per (item, pipeline),
~20,000 calls total. Tests whether PPA's improvement persists once the
persona pool is matched to each item's actual respondent pool.

Default: a local open model (--model <key from silicon.ladder.MODELS>;
needs torch + transformers and GPUs).
--api: the paper's main model over the OpenAI API (SIM_MODEL, default
gpt-4o; needs OPENAI_API_KEY).

Needs OpinionQA data under data/human_resp/ for persona matching and the
original 50-item run's item_list.json under PROJECT_ROOT.
Outputs: results/matched_ppa/{model}_matched_summary.csv (+ per-item JSON);
shipped combined copy: data/openmodel/openmodel_matched_summary.csv.
Run: python run.py matched-ppa --model qwen25-instruct
     OPENAI_API_KEY=... python run.py matched-ppa --api
"""
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("USE_TF", "0")

import numpy as np

from silicon import results_dir, require_env
from silicon.open_model import (ORDERINGS, PHRASINGS, POSITIONS, build_argyle,
                                parse_digit, empirical, tv)

SEED = 42
BATCH = int(os.environ.get("OM_BATCH", "24"))


def _summarize(rows, out_dir, key):
    import pandas as pd
    from scipy.stats import wilcoxon
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{key}_matched_summary.csv", index=False)
    print(f"\n== {key} (n={len(df)}) ==")
    print(f"matched {df.tv_matched.mean():.3f}  matched+PPA {df.tv_matched_ppa.mean():.3f}")
    print(f"Wilcoxon p={wilcoxon(df.tv_matched, df.tv_matched_ppa).pvalue:.3g}")


def run_local(key, out_dir):
    from silicon.ladder import MODELS
    from silicon.opinionqa import load_100_items, sample_matched_personas

    if key not in MODELS:
        raise SystemExit(f"unknown checkpoint key {key!r}; valid keys: {', '.join(MODELS)}")
    spec = MODELS[key]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(spec["hf"])
    model = AutoModelForCausalLM.from_pretrained(
        spec["hf"], torch_dtype=torch.float16, device_map="auto")
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    @torch.no_grad()
    def gen_batch(message_lists, max_new=8):
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                   for m in message_lists]
        outs = []
        for i in range(0, len(prompts), BATCH):
            enc = tok(prompts[i:i + BATCH], return_tensors="pt", padding=True).to(model.device)
            torch.manual_seed(SEED + i)
            g = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0,
                               max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
            for row in g:
                outs.append(tok.decode(row[enc.input_ids.shape[1]:], skip_special_tokens=True))
        return outs

    items = load_100_items()
    rng = random.Random(SEED)
    rows = []
    for idx, item in enumerate(items):
        out_path = out_dir / f"{key}_{item['key']}.json"
        if out_path.exists():
            rows.append(json.load(open(out_path)))
            continue
        try:
            personas = sample_matched_personas(item, n=100, seed=SEED)
        except Exception as e:
            print(f"[skip] {item['key']}: {e}", flush=True)
            continue
        record = {"item_key": item["key"], "model": key, "pew_dist": item["pew_dist"]}

        for pipe in ["matched", "matched_ppa"]:
            msgs, orderings = [], []
            for p in personas:
                if pipe == "matched":
                    m, o = build_argyle(item, p)
                else:
                    m, o = build_argyle(item, p, rng.choice(list(ORDERINGS)),
                                        rng.choice(list(PHRASINGS)), rng.choice(POSITIONS))
                msgs.append(m); orderings.append(o)
            texts = gen_batch(msgs)
            canon = []
            for t, o in zip(texts, orderings):
                d = parse_digit(t)
                canon.append(o[d - 1] if d else None)
            p_hat = empirical(canon)
            record[f"{pipe}_n_valid"] = sum(1 for c in canon if c)
            record[f"tv_{pipe}"] = tv(p_hat, item["pew_dist"])

        json.dump(record, open(out_path, "w"))
        rows.append(record)
        if (idx + 1) % 10 == 0:
            print(f"[{idx+1}] matched {np.mean([r['tv_matched'] for r in rows]):.3f} "
                  f"matched+ppa {np.mean([r['tv_matched_ppa'] for r in rows]):.3f}", flush=True)

    _summarize(rows, out_dir, key)


def run_api(out_dir):
    from openai import OpenAI

    from silicon.opinionqa import load_100_items, sample_matched_personas

    model_name = os.environ.get("SIM_MODEL", "gpt-4o")
    client = OpenAI(api_key=require_env("OPENAI_API_KEY"))

    def call(messages):
        for attempt in range(4):
            try:
                r = client.chat.completions.create(model=model_name, messages=messages,
                                                   temperature=1.0, max_tokens=8)
                return r.choices[0].message.content
            except Exception:
                if attempt == 3:
                    return None
                time.sleep(2 ** attempt)

    items = load_100_items()
    rng = random.Random(SEED)
    rows = []
    for idx, item in enumerate(items):
        out_path = out_dir / f"{model_name}_{item['key']}.json"
        if out_path.exists():
            rows.append(json.load(open(out_path)))
            continue
        personas = sample_matched_personas(item, n=100, seed=SEED)
        record = {"item_key": item["key"], "model": model_name, "pew_dist": item["pew_dist"]}
        for pipe in ["matched", "matched_ppa"]:
            jobs = []
            for p in personas:
                if pipe == "matched":
                    m, o = build_argyle(item, p)
                else:
                    m, o = build_argyle(item, p, rng.choice(list(ORDERINGS)),
                                        rng.choice(list(PHRASINGS)), rng.choice(POSITIONS))
                jobs.append((m, o))
            with ThreadPoolExecutor(max_workers=16) as pool:
                texts = list(pool.map(lambda j: call(j[0]), jobs))
            canon = []
            for t, (_, o) in zip(texts, jobs):
                d = parse_digit(t)
                canon.append(o[d - 1] if d else None)
            record[f"{pipe}_n_valid"] = sum(1 for c in canon if c)
            record[f"tv_{pipe}"] = tv(empirical(canon), item["pew_dist"])
        json.dump(record, open(out_path, "w"))
        rows.append(record)
        if (idx + 1) % 10 == 0:
            print(f"[{idx+1}] matched {np.mean([r['tv_matched'] for r in rows]):.3f} "
                  f"matched+ppa {np.mean([r['tv_matched_ppa'] for r in rows]):.3f}", flush=True)

    _summarize(rows, out_dir, model_name)


def main(args):
    out_dir = results_dir("matched_ppa")
    if getattr(args, "api", False):
        run_api(out_dir)
    else:
        run_local(args.model, out_dir)
