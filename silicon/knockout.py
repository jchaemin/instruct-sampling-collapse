"""Instruction-tuning knockout via task arithmetic (the alpha sweep of Sec. 4).

tau = theta_tuned - theta_base is the instruction-tuning update as a weight
vector (Ilharco et al., 2023, "Editing Models with Task Arithmetic").
theta(alpha) = theta_base + alpha * tau dials that update in and out with the
prompt held FIXED (the base 3-shot format of A.15), so only the weights
change. alpha = 0 is the base checkpoint, alpha = 1 the tuned checkpoint,
alpha = 1.25 amplifies the update past the real endpoint.

For each alpha: the five distributional tasks of silicon/ladder.py, N = 200
sampled calls at T = 1.0, plus a teacher-forced T = 0 support read (max
probability + entropy over the support). Shipped output of our run:
data/ladder/knockout_full.json (OLMo-2 base->SFT and Qwen2.5 base->Instruct).

Needs a GPU with torch/transformers; ~30 GB of HF downloads per lineage.
Run: python run.py knockout [--lineage olmo_sft|qwen_instruct]
"""
import gc
import json
import math

from silicon import results_dir
from silicon.ladder import BASE_SHOTS, MODELS, TASKS, tv

SWEEP_TASKS = ["uniform_digit", "fair_coin", "skewed_binary", "bimodal_5way", "skewed_5way"]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
N = 200
SEED = 13

LINEAGES = {
    "olmo_sft": {"label": "OLMo-2 base -> SFT",
                 "base": MODELS["olmo2-base"]["hf"], "tuned": MODELS["olmo2-sft"]["hf"]},
    "qwen_instruct": {"label": "Qwen2.5 base -> Instruct",
                      "base": MODELS["qwen25-base"]["hf"], "tuned": MODELS["qwen25-instruct"]["hf"]},
}


def entropy(probs):
    return -sum(p * math.log(p) for p in probs if p > 0)


def run_lineage(cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"########## {cfg['label']} ##########", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg["base"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["base"], torch_dtype=torch.float16,
                                                 device_map="auto")
    model.eval()
    base_sd = {k: v.detach().to("cpu", torch.float16).clone()
               for k, v in model.state_dict().items()}
    tuned = AutoModelForCausalLM.from_pretrained(cfg["tuned"], torch_dtype=torch.float16)
    tau = {k: v.detach().to("cpu", torch.float16) - base_sd[k]
           for k, v in tuned.state_dict().items()}
    del tuned
    gc.collect()
    assert set(base_sd) == set(tau), "param-name mismatch between base and tuned"

    def set_alpha(alpha):
        with torch.no_grad():
            for name, p in model.named_parameters():
                val = base_sd[name] + (alpha * tau[name].float()).to(torch.float16)
                p.data.copy_(val.to(p.device, p.dtype))

    @torch.no_grad()
    def generate(prompt, n, max_new=6):
        enc = tok(prompt, return_tensors="pt").to(model.device)
        outs = []
        for start in range(0, n, 20):
            bs = min(20, n - start)
            torch.manual_seed(SEED + start)
            gen = model.generate(input_ids=enc.input_ids.repeat(bs, 1),
                                 attention_mask=enc.attention_mask.repeat(bs, 1),
                                 do_sample=True, temperature=1.0, top_p=1.0,
                                 max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
            for row in gen:
                outs.append(tok.decode(row[enc.input_ids.shape[1]:], skip_special_tokens=True))
            del gen
            torch.cuda.empty_cache()
        return outs

    @torch.no_grad()
    def logit_read(prompt, support):
        penc = tok(prompt, return_tensors="pt")
        plen = penc.input_ids.shape[1]
        lp = {}
        for s in support:
            best = None
            for form in (str(s), " " + str(s)):
                ids = tok.encode(form, add_special_tokens=False)
                if not ids:
                    continue
                full = torch.cat([penc.input_ids[0], torch.tensor(ids)]).unsqueeze(0).to(model.device)
                logits = model(input_ids=full).logits[0].float()
                val = sum(torch.log_softmax(logits[plen - 1 + j], dim=-1)[tid].item()
                          for j, tid in enumerate(ids))
                best = val if best is None or val > best else best
            lp[str(s)] = best
        mx = max(lp.values())
        exp = {k: math.exp(v - mx) for k, v in lp.items()}
        z = sum(exp.values())
        return {k: v / z for k, v in exp.items()}

    def measure_all():
        cell = {}
        for tname in SWEEP_TASKS:
            task = TASKS[tname]
            prompt = BASE_SHOTS + f"Q: {task['v0_prompt']}\nA:"
            texts = generate(prompt, N)
            counts, parsed = {}, 0
            for t in texts:
                v = task["parser"](t)
                if v is not None and str(v) in [str(s) for s in task["support"]]:
                    counts[str(v)] = counts.get(str(v), 0) + 1
                    parsed += 1
            sup = logit_read(prompt, [str(s) for s in task["support"]])
            cell[tname] = {"tv": tv(counts, task["support"], task["target"]),
                           "parse": parsed,
                           "logit_maxprob": max(sup.values()),
                           "logit_entropy": entropy(sup.values())}
        tvs = [c["tv"] for c in cell.values() if c["tv"] is not None]
        cell["_mean"] = {"tv": sum(tvs) / len(tvs),
                         "logit_maxprob": sum(c["logit_maxprob"] for c in cell.values() if "logit_maxprob" in c) / len(SWEEP_TASKS),
                         "logit_entropy": sum(c["logit_entropy"] for c in cell.values() if "logit_entropy" in c) / len(SWEEP_TASKS)}
        return cell

    results = {"base_ref": measure_all()}
    print(f"  base_ref meanTV={results['base_ref']['_mean']['tv']:.3f}", flush=True)
    for alpha in ALPHAS:
        set_alpha(alpha)
        cell = measure_all()
        results[f"alpha_{alpha}"] = cell
        m = cell["_mean"]
        print(f"  alpha={alpha:.2f}  meanTV={m['tv']:.3f}  logit_maxp={m['logit_maxprob']:.3f}  "
              f"logit_H={m['logit_entropy']:.3f}", flush=True)
    del model, base_sd, tau
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main(args):
    keys = [args.lineage] if getattr(args, "lineage", None) else list(LINEAGES)
    out = results_dir("knockout") / "knockout_full.json"
    allres = {}
    if out.exists():
        allres = json.load(open(out))
    for key in keys:
        cfg = LINEAGES[key]
        allres[key] = {"label": cfg["label"], "alphas": ALPHAS, "tasks": SWEEP_TASKS,
                       "N": N, "seed": SEED, "results": run_lineage(cfg)}
        json.dump(allres, open(out, "w"), indent=1)
        print(f"[written] {out}", flush=True)
