"""Non-Likert applied replication: Argyle vs PPA vs describe on Pew items
with 2-4 substantive options (not 5-point scales), on local open models.
Tests whether the KNOWS/DOES gap is an artifact of the 5-point Likert format.

Same pipeline logic as the open-model replication, generalized to k options:
  - orderings: ascending, descending, and two seeded shuffles of 1..k
  - phrasing / persona-position grids identical
  - describe asks for k probabilities in option order
Item set: data/openmodel/nonlikert_items.json (shipped), or a fresh copy in
results/nonlikert/ if --build-items has been run.

--build-items — rebuild the item set deterministically: scan a fixed wave
list in order, keep items whose options (excluding 'Refused') number 2-4 and
whose responses.csv values match the option strings with >= 200 substantive
responses; take the first 8 per option-count bucket (2, 3, 4), cap 24 items.
Needs the OpinionQA metadata CSVs under data/model_input/ and the waves
under data/human_resp/ (see README).

Outputs: results/nonlikert/ (per-item JSON + {model}_nonlikert_summary.csv;
shipped combined copy: data/openmodel/openmodel_nonlikert_summary.csv).
The benchmark needs torch + transformers and GPUs.
Run: python run.py nonlikert --model qwen25-instruct
     python run.py nonlikert --build-items
"""
import ast
import json
import os
import random
import re

os.environ.setdefault("USE_TF", "0")

import numpy as np

from silicon import DATA, results_dir

PHRASINGS = {
    "direct":      "{q}",
    "indirect":    "How would you rate the following question: {q}",
    "third_person": "How would you, in your role, respond to this question: {q}",
}
POSITIONS = ["system_then_user", "user_only", "user_with_recap"]
SEED = 42
BATCH = int(os.environ.get("OM_BATCH", "24"))

BUILD_WAVES = ["W26", "W27", "W32", "W34", "W41", "W42", "W43", "W45",
               "W49", "W50", "W54", "W92"]


def _items_json_path():
    fresh = results_dir("nonlikert") / "nonlikert_items.json"
    if fresh.exists():
        return fresh
    return DATA / "openmodel" / "nonlikert_items.json"


# ---- --build-items stage ----

def build_items():
    import pandas as pd

    if not (DATA / "model_input").exists():
        raise SystemExit(
            "--build-items needs the public OpinionQA release under data/ "
            "(model_input/ and human_resp/); see README, External data. The "
            "selected items already ship as data/openmodel/nonlikert_items.json.")

    out_path = results_dir("nonlikert") / "nonlikert_items.json"
    buckets = {2: [], 3: [], 4: []}
    for w in BUILD_WAVES:
        meta = pd.read_csv(DATA / "model_input" / f"Pew_American_Trends_Panel_{w}.csv", sep="\t")
        try:
            resp = pd.read_csv(DATA / "human_resp" / f"American_Trends_Panel_{w}" / "responses.csv",
                               low_memory=False)
        except FileNotFoundError:
            continue
        for _, row in meta.iterrows():
            keyname = row["key"]
            opts = [o for o in ast.literal_eval(row["options"]) if o != "Refused"]
            k = len(opts)
            if k not in buckets or len(buckets[k]) >= 8:
                continue
            if keyname not in resp.columns:
                continue
            vals = resp[keyname].astype(str).str.strip()
            counts = np.array([float((vals == o).sum()) for o in opts])
            if counts.sum() < 200 or (counts == 0).any():
                continue
            # Skip items whose responses contain many values not in the option list.
            matched = counts.sum()
            substantive = (~vals.isin(["Refused", "nan"])).sum()
            if matched / max(substantive, 1) < 0.95:
                continue
            buckets[k].append({
                "key": keyname, "wave": f"American_Trends_Panel_{w}",
                "question": row["question"], "options": opts, "k": k,
                "pew_dist": (counts / counts.sum()).tolist(),
            })
        if all(len(v) >= 8 for v in buckets.values()):
            break

    items = [it for k in (2, 3, 4) for it in buckets[k]]
    json.dump(items, open(out_path, "w"), indent=1)
    print(f"selected {len(items)} items:",
          {k: len(v) for k, v in buckets.items()})
    for it in items[:6]:
        print(f"  {it['key']} k={it['k']} pew={[round(p,2) for p in it['pew_dist']]}")
    print(f"[written] {out_path}")


# ---- benchmark stage ----

def orderings_for(k, rng):
    base = list(range(1, k + 1))
    o = {"ascending": base, "descending": base[::-1]}
    for name in ("shuffled_a", "shuffled_b"):
        p = base[:]
        rng.shuffle(p)
        o[name] = p
    return o


def build_argyle(item, persona, ordering, phr_n="direct", pos_n="system_then_user"):
    k = item["k"]
    labels = item["options"]
    phrasing = PHRASINGS[phr_n].format(q=item["question"])
    options = "\n".join(f"{i+1}. {labels[ordering[i] - 1]}" for i in range(k))
    user = f"{phrasing}\n\nOptions:\n{options}\n\nRespond with just the number (1-{k})."
    p = persona["descrip"]
    if pos_n == "system_then_user":
        return [{"role": "system", "content": p}, {"role": "user", "content": user}]
    if pos_n == "user_only":
        return [{"role": "user", "content": p + "\n\n" + user}]
    return [{"role": "system", "content": p},
            {"role": "user", "content": "Given this background, answer:\n\n" + user}]


def build_describe(item):
    k = item["k"]
    options = "\n".join(f"  {i+1}. {opt}" for i, opt in enumerate(item["options"]))
    user = (
        f"Estimate the distribution of responses to the following Pew survey "
        f"item across a representative sample of U.S. adults. Output a JSON "
        f"object with a single key 'distribution' whose value is a list of "
        f"{k} probabilities summing to 1.0, in the order of options 1 through {k}.\n\n"
        f"Question: {item['question']}\n\nOptions:\n{options}"
    )
    return [{"role": "user", "content": user}]


def parse_digit(text, k):
    if not text: return None
    m = re.search(rf"[1-{k}]", text)
    return int(m.group()) if m else None


def parse_dist(text, k):
    if not text: return None
    cleaned = re.sub(r"```(json)?", "", text)
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group())
            dist = d.get("distribution") or list(d.values())[0]
            dist = [float(x) for x in dist][:k]
            s = sum(dist)
            if s > 1e-9 and len(dist) == k:
                return [x / s for x in dist]
        except Exception:
            pass
    pcts = re.findall(r"(\d+(?:\.\d+)?)\s*%", text)
    if len(pcts) >= k:
        vals = [float(p) for p in pcts[:k]]
        s = sum(vals)
        if s > 1e-9:
            return [v / s for v in vals]
    return None


def empirical(samples, k):
    counts = np.bincount([s for s in samples if s], minlength=k + 1)[1:k + 1].astype(float)
    if counts.sum() == 0: return np.full(k, 1.0 / k)
    return counts / counts.sum()


def tv(p, q):
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


def run_benchmark(key):
    from silicon.ladder import MODELS
    from silicon.opinionqa import sample_personas

    out_dir = results_dir("nonlikert")
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
    def gen_batch(message_lists, max_new=8, sample=True):
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                   for m in message_lists]
        outs = []
        for i in range(0, len(prompts), BATCH):
            enc = tok(prompts[i:i + BATCH], return_tensors="pt", padding=True).to(model.device)
            torch.manual_seed(SEED + i)
            g = model.generate(**enc, do_sample=sample,
                               temperature=1.0 if sample else None,
                               top_p=1.0 if sample else None,
                               max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
            for row in g:
                outs.append(tok.decode(row[enc.input_ids.shape[1]:], skip_special_tokens=True))
        return outs

    personas = sample_personas(100, seed=SEED)
    items = json.load(open(_items_json_path()))
    rng = random.Random(SEED)
    rows = []
    for idx, item in enumerate(items):
        out_path = out_dir / f"{key}_{item['key']}.json"
        if out_path.exists():
            rows.append(json.load(open(out_path)))
            continue
        k = item["k"]
        orderings = orderings_for(k, random.Random(SEED + k))
        record = {"item_key": item["key"], "model": key, "k": k, "pew_dist": item["pew_dist"]}

        for pipe in ["argyle", "ppa"]:
            msgs, used = [], []
            for p in personas:
                if pipe == "argyle":
                    o = orderings["ascending"]; msgs.append(build_argyle(item, p, o))
                else:
                    o = orderings[rng.choice(list(orderings))]
                    msgs.append(build_argyle(item, p, o,
                                             rng.choice(list(PHRASINGS)), rng.choice(POSITIONS)))
                used.append(o)
            texts = gen_batch(msgs)
            canon = []
            for t, o in zip(texts, used):
                d = parse_digit(t, k)
                canon.append(o[d - 1] if d else None)
            record[f"{pipe}_n_valid"] = sum(1 for c in canon if c)
            record[f"tv_{pipe}"] = tv(empirical(canon, k), item["pew_dist"])

        text = gen_batch([build_describe(item)], max_new=200, sample=False)[0]
        dist = parse_dist(text, k)
        record["describe_raw"] = text[:600]
        record["tv_describe"] = tv(dist, item["pew_dist"]) if dist else None
        json.dump(record, open(out_path, "w"))
        rows.append(record)
        print(f"[{idx+1}/{len(items)}] {item['key']} k={k} "
              f"argyle {record['tv_argyle']:.3f} ppa {record['tv_ppa']:.3f} "
              f"describe {record['tv_describe']}", flush=True)

    import pandas as pd
    from scipy.stats import wilcoxon
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{key}_nonlikert_summary.csv", index=False)
    m = df.dropna(subset=["tv_describe"])
    print(f"\n== {key} non-Likert (n={len(df)}, describe ok {len(m)}) ==")
    print(f"argyle {df.tv_argyle.mean():.3f}  ppa {df.tv_ppa.mean():.3f}  "
          f"describe {m.tv_describe.mean():.3f}")
    print("by k:", df.groupby("k")[["tv_argyle", "tv_ppa"]].mean().round(3).to_dict())
    if len(m) > 8:
        print(f"argyle vs describe p={wilcoxon(m.tv_argyle, m.tv_describe).pvalue:.3g}")
        print(f"argyle vs ppa p={wilcoxon(df.tv_argyle, df.tv_ppa).pvalue:.3g}")


def main(args):
    if getattr(args, "build_items", False):
        build_items()
    else:
        run_benchmark(args.model)
