"""Post-training-ladder replication of the paper's five-target sampling panel,
run locally on open checkpoints.

Isolates WHICH post-training stage induces the sampling collapse (OLMo-2
base -> SFT -> DPO -> Instruct/RLVR, a fully open ladder) and whether
NON-instruction fine-tuning does too (Qwen2.5-Coder-7B continued-pretrain
control). Also replicates the paper's Llama-3.1-8B base-vs-instruct panel
locally, plus Qwen3-8B as a fourth family/generation.

Protocol mirrors the paper's base-vs-instruct appendix:
  V0: plain sampling, T=1.0, top-p 1.0, 200 calls/task.
      Base models: 3-shot cross-domain Q/A format. Instruct: chat template, zero-shot.
  V8: list-of-20 in one call, 10 calls/task. Base models: 1 canonical Q/A exemplar.
  Logit read: T=0 forward pass, within-support first-token distribution.
  Meta probes (instruct only): M1 predict the 200-call histogram;
  M2 vote between deterministic and randomize policies.
Tasks: point_5 control + the paper's five KNOWS/DOES tasks
(verbatim from cross_family.KD_TASKS).

Outputs: results/ladder/{model}_cells.csv (+ raw JSONL); shipped combined
copy: data/ladder/ladder_cells_all.csv (seeds 13 and 47).
Needs torch + transformers; one GPU per 7-8B model. Idempotent: skips cells
whose raw JSONL is complete.
Run: python run.py ladder --models olmo2-base,olmo2-sft [--gpus 0,1] [--seed 13]
"""
import json
import os
import re
import time

from silicon import results_dir

os.environ.setdefault("USE_TF", "0")  # a broken tensorflow install breaks transformers import

MODELS = {
    # OLMo-2 alignment ladder (fully open post-training pipeline)
    "olmo2-base":     {"hf": "allenai/OLMo-2-1124-7B",          "kind": "base"},
    "olmo2-sft":      {"hf": "allenai/OLMo-2-1124-7B-SFT",      "kind": "instruct"},
    "olmo2-dpo":      {"hf": "allenai/OLMo-2-1124-7B-DPO",      "kind": "instruct"},
    "olmo2-instruct": {"hf": "allenai/OLMo-2-1124-7B-Instruct", "kind": "instruct"},
    # Paper's family, local replication
    "llama31-base":     {"hf": "meta-llama/Llama-3.1-8B",          "kind": "base"},
    "llama31-instruct": {"hf": "meta-llama/Llama-3.1-8B-Instruct", "kind": "instruct"},
    # Qwen family + non-instruction fine-tune control
    "qwen25-base":       {"hf": "Qwen/Qwen2.5-7B",          "kind": "base"},
    "qwen25-coder-base": {"hf": "Qwen/Qwen2.5-Coder-7B",    "kind": "base"},
    "qwen25-instruct":   {"hf": "Qwen/Qwen2.5-7B-Instruct", "kind": "instruct"},
    # Fourth family/generation, added during the response period
    "qwen3-8b":          {"hf": "Qwen/Qwen3-8B",             "kind": "instruct",
                          "chat_kwargs": {"enable_thinking": False}},
}


def parse_digit(text):
    if not text: return None
    m = re.search(r"[1-5]", text.strip())
    return int(m.group()) if m else None

def parse_coin(text):
    if not text: return None
    t = text.strip().upper()
    if t.startswith("H") or "HEAD" in t: return "H"
    if t.startswith("T") or "TAIL" in t: return "T"
    return None

def parse_alpha_beta(text):
    if not text: return None
    t = text.strip().lower()
    if "alpha" in t or t.startswith("a"): return "alpha"
    if "beta" in t or t.startswith("b"): return "beta"
    return None

TASKS = {
    "point_5": {
        "target": [0, 0, 0, 0, 1], "parser": parse_digit, "support": [1, 2, 3, 4, 5],
        "v0_prompt": "Always output the digit 5. Reply with just the digit."},
    "uniform_digit": {
        "target": [0.2] * 5, "parser": parse_digit, "support": [1, 2, 3, 4, 5],
        "v0_prompt": "Pick a random digit from 1 to 5 uniformly at random. Reply with just the digit."},
    "fair_coin": {
        "target": [0.5, 0.5], "parser": parse_coin, "support": ["H", "T"],
        "v0_prompt": "Flip a fair coin. Output H or T."},
    "skewed_binary": {
        "target": [0.7, 0.3], "parser": parse_alpha_beta, "support": ["alpha", "beta"],
        "v0_prompt": "Output 'alpha' with probability 70% and 'beta' with probability 30%. Reply with just the word."},
    "bimodal_5way": {
        "target": [0.4, 0.1, 0.0, 0.1, 0.4], "parser": parse_digit, "support": [1, 2, 3, 4, 5],
        "v0_prompt": "Output a digit from 1 to 5 where 1 appears 40%, 2 appears 10%, 3 appears 0%, 4 appears 10%, 5 appears 40%. Reply with just the digit."},
    "skewed_5way": {
        "target": [0.5, 0.2, 0.1, 0.1, 0.1], "parser": parse_digit, "support": [1, 2, 3, 4, 5],
        "v0_prompt": "Output a digit from 1 to 5 where 1 appears 50%, 2 appears 20%, 3 appears 10%, 4 appears 10%, 5 appears 10%. Reply with just the digit."},
}

# Fixed cross-domain 3-shot exemplars for base models: teach the bare-answer
# format without demonstrating any answer from the test task's support.
BASE_SHOTS = (
    "Q: Pick a random letter from the English alphabet uniformly at random. Output only the letter.\n"
    "A: Q\n\n"
    "Q: Pick a random integer uniformly between 1 and 100. Output only the integer.\n"
    "A: 37\n\n"
    "Q: Pick a random color from red, green, blue. Output only the color.\n"
    "A: green\n\n"
)

BASE_V8_SHOT = (
    "Q: Generate a list of 4 answers where 'X' about 2 times, 'Y' about 2 times. "
    "Output as JSON with key 'samples' whose value is a list of 4 strings.\n"
    'A: {"samples": ["X", "Y", "Y", "X"]}\n\n'
)

N_V0 = 200
N_V8_CALLS = 10
V8_LIST_LEN = 20
BATCH = 50
DEFAULT_SEED = 13


def v8_user_prompt(task):
    p = task["target"]
    expected = [round(pi * V8_LIST_LEN) for pi in p]
    freq = ", ".join(f"'{task['support'][i]}' about {expected[i]} times" for i in range(len(p)))
    return (f"Generate a list of {V8_LIST_LEN} answers where {freq}. "
            f"Output as JSON with key 'samples' whose value is a list of {V8_LIST_LEN} strings.")


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


def tv(counts, support, target):
    n = sum(counts.get(str(s), 0) for s in support)
    if n == 0: return None
    emp = [counts.get(str(s), 0) / n for s in support]
    return 0.5 * sum(abs(e - t) for e, t in zip(emp, target))


def main(args):
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    seed = args.seed
    tag = "" if seed == DEFAULT_SEED else f"_seed{seed}"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    raw_dir = results_dir("ladder/raw")
    out_dir = results_dir("ladder")

    for key in args.models.split(","):
        if key not in MODELS:
            raise SystemExit(f"unknown checkpoint key {key!r}; valid keys: {', '.join(MODELS)}")
        spec = MODELS[key]
        print(f"== {key} ({spec['hf']}) ==", flush=True)
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(spec["hf"])
        model = AutoModelForCausalLM.from_pretrained(
            spec["hf"], torch_dtype=torch.float16, device_map="auto")
        model.eval()
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        print(f"loaded in {time.time()-t0:.0f}s", flush=True)

        cells = []

        def build_prompt(task, condition):
            if condition == "V0":
                user = task["v0_prompt"]
                if spec["kind"] == "base":
                    return BASE_SHOTS + f"Q: {user}\nA:"
                return tok.apply_chat_template(
                    [{"role": "user", "content": user}],
                    tokenize=False, add_generation_prompt=True,
                    **spec.get("chat_kwargs", {}))
            else:  # V8
                user = v8_user_prompt(task)
                if spec["kind"] == "base":
                    return BASE_V8_SHOT + f"Q: {user}\nA:"
                return tok.apply_chat_template(
                    [{"role": "user", "content": user}],
                    tokenize=False, add_generation_prompt=True,
                    **spec.get("chat_kwargs", {}))

        @torch.no_grad()
        def generate(prompt, n, max_new):
            outs = []
            enc = tok(prompt, return_tensors="pt").to(model.device)
            for start in range(0, n, BATCH):
                bs = min(BATCH, n - start)
                torch.manual_seed(seed + start)
                gen = model.generate(
                    input_ids=enc.input_ids.repeat(bs, 1),
                    attention_mask=enc.attention_mask.repeat(bs, 1),
                    do_sample=True, temperature=1.0, top_p=1.0,
                    max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
                for row in gen:
                    text = tok.decode(row[enc.input_ids.shape[1]:], skip_special_tokens=True)
                    outs.append(text)
                del gen
                torch.cuda.empty_cache()
            return outs

        @torch.no_grad()
        def logit_read(task):
            """T=0 support distribution via teacher-forced sequence scoring:
            for each support element s, sum log P of the tokens of s (and of
            " s") appended to the prompt; take the better-scoring form."""
            prompt = build_prompt(task, "V0")
            penc = tok(prompt, return_tensors="pt")
            plen = penc.input_ids.shape[1]
            out = {}
            for s in task["support"]:
                best = None
                for form in (str(s), " " + str(s)):
                    ids = tok.encode(form, add_special_tokens=False)
                    if not ids:
                        continue
                    full = torch.cat(
                        [penc.input_ids[0], torch.tensor(ids)]).unsqueeze(0).to(model.device)
                    logits = model(input_ids=full).logits[0].float()
                    lp = 0.0
                    for j, tid in enumerate(ids):
                        lp += torch.log_softmax(logits[plen - 1 + j], dim=-1)[tid].item()
                    best = lp if best is None or lp > best else best
                out[str(s)] = best
            import math
            mx = max(out.values())
            exp = {k: math.exp(v - mx) for k, v in out.items()}
            z = sum(exp.values())
            norm = {k: v / z for k, v in exp.items()}
            return {"raw_logprob": out, "within_support": norm}

        for task_name, task in TASKS.items():
            # V0
            raw_path = raw_dir / f"{key}{tag}_{task_name}_V0.jsonl"
            if raw_path.exists() and sum(1 for _ in open(raw_path)) >= args.n_v0:
                texts = [json.loads(l)["text"] for l in open(raw_path)]
                print(f"[skip] {task_name} V0 (cached)", flush=True)
            else:
                t1 = time.time()
                cut = 4 if task_name != "skewed_binary" else 8
                texts = generate(build_prompt(task, "V0"), args.n_v0, max_new=cut + 8)
                with open(raw_path, "w") as f:
                    for t in texts:
                        f.write(json.dumps({"text": t}) + "\n")
                print(f"{task_name} V0 done in {time.time()-t1:.0f}s", flush=True)
            firstline = [t.split("\n")[0] if t else "" for t in texts]
            parsed = [task["parser"](t) for t in firstline]
            valid = [p for p in parsed if p is not None]
            counts = {}
            for v in valid:
                counts[str(v)] = counts.get(str(v), 0) + 1
            n_valid = len(valid)
            p_mode = max(counts.values()) / n_valid if n_valid else None
            cells.append({
                "model": key, "hf": spec["hf"], "kind": spec["kind"],
                "task": task_name, "condition": "V0",
                "n": len(parsed), "n_valid": n_valid,
                "parse_fail": len(parsed) - n_valid,
                "counts": json.dumps(counts),
                "tv": tv(counts, task["support"], task["target"]),
                "p_mode": p_mode,
                "seed": seed,
            })

            # V8
            raw_path = raw_dir / f"{key}{tag}_{task_name}_V8.jsonl"
            if raw_path.exists() and sum(1 for _ in open(raw_path)) >= N_V8_CALLS:
                texts = [json.loads(l)["text"] for l in open(raw_path)]
                print(f"[skip] {task_name} V8 (cached)", flush=True)
            else:
                t1 = time.time()
                texts = generate(build_prompt(task, "V8"), N_V8_CALLS, max_new=350)
                with open(raw_path, "w") as f:
                    for t in texts:
                        f.write(json.dumps({"text": t}) + "\n")
                print(f"{task_name} V8 done in {time.time()-t1:.0f}s", flush=True)
            all_samples, ok_calls = [], 0
            for t in texts:
                if spec["kind"] == "base":
                    t = t.split("\nQ:")[0]
                s = [x for x in parse_json_list(t, task["parser"]) if x is not None]
                if s:
                    ok_calls += 1
                    all_samples.extend(s)
            counts = {}
            for v in all_samples:
                counts[str(v)] = counts.get(str(v), 0) + 1
            cells.append({
                "model": key, "hf": spec["hf"], "kind": spec["kind"],
                "task": task_name, "condition": "V8",
                "n": N_V8_CALLS, "n_valid": len(all_samples),
                "parse_fail": N_V8_CALLS - ok_calls,
                "counts": json.dumps(counts),
                "tv": tv(counts, task["support"], task["target"]),
                "p_mode": (max(counts.values()) / len(all_samples)) if all_samples else None,
                "seed": seed,
            })

            # Meta-knowledge probes (instruct only): does the model KNOW that
            # repeated independent calls should produce variation?
            if spec["kind"] == "instruct":
                # M1: predict the histogram of 200 independent calls
                m1_user = (f"Suppose 200 independent copies of you are each given this "
                           f"instruction once, with no shared memory: \"{task['v0_prompt']}\" "
                           f"Predict the histogram of the 200 answers. Output JSON: an object "
                           f"whose keys are the possible answers and whose values are expected "
                           f"counts summing to 200.")
                m1_prompt = tok.apply_chat_template(
                    [{"role": "user", "content": m1_user}],
                    tokenize=False, add_generation_prompt=True,
                    **spec.get("chat_kwargs", {}))
                raw_path = raw_dir / f"{key}{tag}_{task_name}_meta_hist.jsonl"
                if raw_path.exists() and sum(1 for _ in open(raw_path)) >= N_V8_CALLS:
                    texts = [json.loads(l)["text"] for l in open(raw_path)]
                else:
                    texts = generate(m1_prompt, N_V8_CALLS, max_new=200)
                    with open(raw_path, "w") as f:
                        for t in texts:
                            f.write(json.dumps({"text": t}) + "\n")
                tvs, ok = [], 0
                for t in texts:
                    m = re.search(r"\{.*\}", re.sub(r"```(json)?", "", t), re.DOTALL)
                    if not m:
                        continue
                    try:
                        d = json.loads(m.group())
                        cnt = {}
                        for k2, v2 in d.items():
                            pk = task["parser"](str(k2))
                            if pk is not None:
                                cnt[str(pk)] = cnt.get(str(pk), 0) + float(v2)
                        t_tv = tv(cnt, task["support"], task["target"])
                        if t_tv is not None:
                            tvs.append(t_tv)
                            ok += 1
                    except Exception:
                        continue
                cells.append({
                    "model": key, "hf": spec["hf"], "kind": spec["kind"],
                    "task": task_name, "condition": "meta_hist",
                    "n": N_V8_CALLS, "n_valid": ok, "parse_fail": N_V8_CALLS - ok,
                    "counts": json.dumps({"mean_tv": (sum(tvs) / len(tvs)) if tvs else None}),
                    "tv": (sum(tvs) / len(tvs)) if tvs else None,
                    "p_mode": None,
                    "seed": seed,
                })
                print(f"{task_name} meta_hist mean TV: "
                      f"{(sum(tvs)/len(tvs)) if tvs else None}", flush=True)

                # M2: which policy is correct?
                m2_user = (f"You will repeatedly receive this instruction, once per "
                           f"independent session with no memory of other sessions: "
                           f"\"{task['v0_prompt']}\" Which response policy is correct? "
                           f"A: In every session, give the single most likely answer. "
                           f"B: Randomize each session so that across sessions the answers "
                           f"match the requested distribution. Reply with just A or B.")
                m2_prompt = tok.apply_chat_template(
                    [{"role": "user", "content": m2_user}],
                    tokenize=False, add_generation_prompt=True,
                    **spec.get("chat_kwargs", {}))
                raw_path = raw_dir / f"{key}{tag}_{task_name}_meta_policy.jsonl"
                if raw_path.exists() and sum(1 for _ in open(raw_path)) >= 20:
                    texts = [json.loads(l)["text"] for l in open(raw_path)]
                else:
                    texts = generate(m2_prompt, 20, max_new=6)
                    with open(raw_path, "w") as f:
                        for t in texts:
                            f.write(json.dumps({"text": t}) + "\n")
                votes = {"A": 0, "B": 0}
                for t in texts:
                    s = t.strip().upper()
                    if s[:1] in votes:
                        votes[s[:1]] += 1
                cells.append({
                    "model": key, "hf": spec["hf"], "kind": spec["kind"],
                    "task": task_name, "condition": "meta_policy",
                    "n": 20, "n_valid": votes["A"] + votes["B"],
                    "parse_fail": 20 - votes["A"] - votes["B"],
                    "counts": json.dumps(votes),
                    "tv": None,
                    "p_mode": (votes["B"] / (votes["A"] + votes["B"])
                               if votes["A"] + votes["B"] else None),
                    "seed": seed,
                })
                print(f"{task_name} meta_policy votes: {votes} "
                      f"(p_mode col = share endorsing B/randomize)", flush=True)

            # logit read
            lr = logit_read(task)
            ws = lr["within_support"]
            cells.append({
                "model": key, "hf": spec["hf"], "kind": spec["kind"],
                "task": task_name, "condition": "logit_t0",
                "n": 1, "n_valid": 1, "parse_fail": 0,
                "counts": json.dumps(ws),
                "tv": (0.5 * sum(abs(ws[str(s)] - task["target"][i])
                                 for i, s in enumerate(task["support"]))
                       if all(ws[str(s)] is not None for s in task["support"]) else None),
                "p_mode": max(v for v in ws.values() if v is not None) if ws else None,
                "seed": seed,
            })
            print(f"{task_name} logit within-support: {ws}", flush=True)

        import csv
        out = out_dir / f"{key}{tag}_cells.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cells[0].keys()))
            w.writeheader()
            w.writerows(cells)
        print(f"[written] {out}", flush=True)

        del model
        torch.cuda.empty_cache()
