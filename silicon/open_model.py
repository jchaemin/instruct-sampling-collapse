"""Open-model replication of the paper's applied OpinionQA result.

Runs the exact baseline-Argyle, PPA, and describe pipelines (prompts verbatim
from the paper's API experiments) on local open-weight instruct models over
the same 100 items and the same 100 personas (seed 42). If the KNOWS/DOES gap
and the PPA improvement replicate, the applied claim no longer rests on a
single closed model.

--repair pass — re-parse (and where needed, regenerate greedily with
full-text capture) describe outputs whose JSON parse failed, using a
percentage-list fallback for models that answer the verbatim describe prompt
in prose. The parsing rule is extended; the prompt itself is unchanged.
Updates the per-item results/open_model/{model}_{item}.json files and
rewrites the summary CSV; the `describe_parse` field records per-item
provenance.

Needs: torch + transformers and 1-2 GPUs, OpinionQA data under
data/human_resp/ (item ground truth is recomputed via
silicon.opinionqa.load_100_items), and the original 50-item run's
item_list.json under PROJECT_ROOT.
Outputs: results/open_model/{model}_{item}.json per item and
results/open_model/{model}_summary.csv (shipped combined copy:
data/openmodel/openmodel_main_summary.csv).
Run: CUDA_VISIBLE_DEVICES=0,1 python run.py open-model-replication --model llama31-instruct [--repair]
"""
import json
import os
import random
import re

os.environ.setdefault("USE_TF", "0")

import numpy as np

from silicon import results_dir

ORDERINGS = {
    "ascending":  [1, 2, 3, 4, 5],
    "descending": [5, 4, 3, 2, 1],
    "shuffled_a": [3, 5, 1, 4, 2],
    "shuffled_b": [2, 4, 1, 5, 3],
}
PHRASINGS = {
    "direct":      "{q}",
    "indirect":    "On a scale from 1 to 5, how would you rate the following question: {q}",
    "third_person": "How would you, in your role, respond to this question: {q}",
}
POSITIONS = ["system_then_user", "user_only", "user_with_recap"]

SEED = 42
BATCH = int(os.environ.get("OM_BATCH", "24"))


def build_argyle(item, persona, ord_n="ascending", phr_n="direct", pos_n="system_then_user"):
    labels = item["options"]
    ordering = ORDERINGS[ord_n]
    phrasing = PHRASINGS[phr_n].format(q=item["question"])
    options = "\n".join(f"{i+1}. {labels[ordering[i] - 1]}" for i in range(5))
    user = f"{phrasing}\n\nOptions:\n{options}\n\nRespond with just the number (1-5)."
    p = persona["descrip"]
    if pos_n == "system_then_user":
        return [{"role": "system", "content": p}, {"role": "user", "content": user}], ordering
    if pos_n == "user_only":
        return [{"role": "user", "content": p + "\n\n" + user}], ordering
    return [{"role": "system", "content": p},
            {"role": "user", "content": "Given this background, answer:\n\n" + user}], ordering


def build_describe(item):
    options = "\n".join(f"  {i+1}. {opt}" for i, opt in enumerate(item["options"]))
    user = (
        f"Estimate the distribution of responses to the following Pew survey "
        f"item across a representative sample of U.S. adults. Output a JSON "
        f"object with a single key 'distribution' whose value is a list of "
        f"5 probabilities summing to 1.0, in the order of options 1 through 5.\n\n"
        f"Question: {item['question']}\n\n"
        f"Options:\n{options}"
    )
    return [{"role": "user", "content": user}]


def parse_digit(text):
    if not text: return None
    m = re.search(r"[1-5]", text)
    return int(m.group()) if m else None


def parse_dist(text):
    if not text: return None
    cleaned = re.sub(r"```(json)?", "", text)
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m: return None
    try:
        d = json.loads(m.group())
        dist = d.get("distribution") or list(d.values())[0]
        dist = [float(x) for x in dist][:5]
        s = sum(dist)
        return [x / s for x in dist] if s > 1e-9 else None
    except Exception:
        return None


def parse_dist_ext(text):
    """parse_dist plus a fallback for prose answers: five percentages in order."""
    d = parse_dist(text)
    if d is not None:
        return d
    if not text:
        return None
    pcts = re.findall(r"(\d+(?:\.\d+)?)\s*%", text)
    if len(pcts) >= 5:
        vals = [float(p) for p in pcts[:5]]
        s = sum(vals)
        if s > 1e-9:
            return [v / s for v in vals]
    return None


def empirical(samples, k=5):
    counts = np.bincount([s for s in samples if s], minlength=k + 1)[1:k + 1].astype(float)
    if counts.sum() == 0: return np.full(k, 1.0 / k)
    return counts / counts.sum()


def tv(p, q):
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


def run_replication(key, out_dir):
    from silicon.ladder import MODELS
    from silicon.opinionqa import load_100_items, sample_personas

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
    def gen_batch(message_lists, max_new, sample=True):
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                   for m in message_lists]
        outs = []
        for i in range(0, len(prompts), BATCH):
            chunk = prompts[i:i + BATCH]
            enc = tok(chunk, return_tensors="pt", padding=True).to(model.device)
            torch.manual_seed(SEED + i)
            g = model.generate(**enc, do_sample=sample,
                               temperature=1.0 if sample else None,
                               top_p=1.0 if sample else None,
                               max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
            for row in g:
                outs.append(tok.decode(row[enc.input_ids.shape[1]:], skip_special_tokens=True))
        return outs

    personas = sample_personas(100, seed=SEED)
    items = load_100_items()
    print(f"{key}: {len(items)} items, {len(personas)} personas", flush=True)
    rng = random.Random(SEED)

    rows = []
    for idx, item in enumerate(items):
        out_path = out_dir / f"{key}_{item['key']}.json"
        if out_path.exists():
            rows.append(json.load(open(out_path)))
            continue
        record = {"item_key": item["key"], "model": key, "pew_dist": item["pew_dist"]}

        # baseline Argyle
        msgs, orderings = [], []
        for p in personas:
            m, o = build_argyle(item, p)
            msgs.append(m); orderings.append(o)
        texts = gen_batch(msgs, max_new=8)
        canon = []
        for t, o in zip(texts, orderings):
            d = parse_digit(t)
            canon.append(o[d - 1] if d else None)
        record["argyle_n_valid"] = sum(1 for c in canon if c)
        p_hat = empirical(canon)
        record["argyle_p_hat"] = p_hat.tolist()
        record["tv_argyle"] = tv(p_hat, item["pew_dist"])

        # PPA
        msgs, orderings = [], []
        for p in personas:
            ord_n = rng.choice(list(ORDERINGS))
            phr_n = rng.choice(list(PHRASINGS))
            pos_n = rng.choice(POSITIONS)
            m, o = build_argyle(item, p, ord_n, phr_n, pos_n)
            msgs.append(m); orderings.append(o)
        texts = gen_batch(msgs, max_new=8)
        canon = []
        for t, o in zip(texts, orderings):
            d = parse_digit(t)
            canon.append(o[d - 1] if d else None)
        record["ppa_n_valid"] = sum(1 for c in canon if c)
        p_hat = empirical(canon)
        record["ppa_p_hat"] = p_hat.tolist()
        record["tv_ppa"] = tv(p_hat, item["pew_dist"])

        # describe (greedy)
        text = gen_batch([build_describe(item)], max_new=200, sample=False)[0]
        dist = parse_dist(text)
        record["describe_raw"] = text[:400]
        record["tv_describe"] = tv(dist, item["pew_dist"]) if dist else None

        json.dump(record, open(out_path, "w"))
        rows.append(record)
        if (idx + 1) % 10 == 0:
            done = [r for r in rows if r.get("tv_describe") is not None]
            print(f"[{idx+1}/{len(items)}] running means: "
                  f"argyle {np.mean([r['tv_argyle'] for r in rows]):.3f} "
                  f"ppa {np.mean([r['tv_ppa'] for r in rows]):.3f} "
                  f"describe {np.mean([r['tv_describe'] for r in done]):.3f}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{key}_summary.csv", index=False)
    from scipy.stats import wilcoxon
    m = df.dropna(subset=["tv_describe"])
    print(f"\n== {key} FINAL (n={len(df)}, describe-parseable {len(m)}) ==")
    print(f"mean TV: argyle {df.tv_argyle.mean():.3f}, ppa {df.tv_ppa.mean():.3f}, "
          f"describe {m.tv_describe.mean():.3f}")
    print(f"argyle vs describe Wilcoxon p={wilcoxon(m.tv_argyle, m.tv_describe).pvalue:.3g}")
    print(f"argyle vs ppa Wilcoxon p={wilcoxon(df.tv_argyle, df.tv_ppa).pvalue:.3g}")


def run_repair(key, out_dir):
    from silicon.ladder import MODELS
    from silicon.opinionqa import load_100_items

    items = {it["key"]: it for it in load_100_items()}

    # pass 1: re-parse cached raw text
    todo = []
    for f in sorted(out_dir.glob(f"{key}_*.json")):
        record = json.load(open(f))
        if record.get("tv_describe") is not None:
            continue
        dist = parse_dist_ext(record.get("describe_raw", ""))
        if dist is not None:
            record["tv_describe"] = tv(dist, record["pew_dist"])
            record["describe_parse"] = "percent_fallback_cached"
            json.dump(record, open(f, "w"))
        else:
            todo.append(f)
    print(f"re-parsed from cache; still unparsed: {len(todo)}", flush=True)

    # pass 2: regenerate greedily with a longer text budget
    if todo:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        spec = MODELS[key]
        tok = AutoTokenizer.from_pretrained(spec["hf"])
        model = AutoModelForCausalLM.from_pretrained(
            spec["hf"], torch_dtype=torch.float16, device_map="auto")
        model.eval()
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

        @torch.no_grad()
        def gen(messages, max_new=350):
            prompt = tok.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
            enc = tok(prompt, return_tensors="pt").to(model.device)
            g = model.generate(**enc, do_sample=False,
                               max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
            return tok.decode(g[0][enc.input_ids.shape[1]:], skip_special_tokens=True)

        for f in todo:
            record = json.load(open(f))
            item = items[record["item_key"]]
            text = gen(build_describe(item))
            dist = parse_dist_ext(text)
            record["describe_raw"] = text[:1500]
            record["tv_describe"] = tv(dist, record["pew_dist"]) if dist else None
            record["describe_parse"] = "percent_fallback_regen" if dist else "unparseable"
            json.dump(record, open(f, "w"))
            print(record["item_key"], record["tv_describe"], flush=True)

    import pandas as pd
    from scipy.stats import wilcoxon
    rows = [json.load(open(f)) for f in sorted(out_dir.glob(f"{key}_*.json"))]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{key}_summary.csv", index=False)
    m = df.dropna(subset=["tv_describe"])
    print(f"\n== {key} (n={len(df)}, describe parseable {len(m)}) ==")
    print(f"argyle {df.tv_argyle.mean():.3f}  ppa {df.tv_ppa.mean():.3f}  "
          f"describe {m.tv_describe.mean():.3f}")
    if len(m) > 10:
        print(f"argyle vs describe p={wilcoxon(m.tv_argyle, m.tv_describe).pvalue:.3g}")
        print(f"argyle vs ppa p={wilcoxon(df.tv_argyle, df.tv_ppa).pvalue:.3g}")


def main(args):
    out_dir = results_dir("open_model")
    if getattr(args, "repair", False):
        run_repair(args.model, out_dir)
    else:
        run_replication(args.model, out_dir)
