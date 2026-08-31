"""Open-ended generation under prompt perturbation, survey-domain tasks.

Five tasks drawn from the paper's application domain: two open-ended survey
items of the kind fielded by national polls (most-important-problem, primary
news source), one policy-priority item, and two synthetic-respondent
attributes of the kind persona-based simulation pipelines generate
(occupation, city of residence).

Conditions per task, 200 sampled calls each at T=1.0:
  fixed        one canonical prompt
  nonsemantic  six whitespace/punctuation transforms of the canonical prompt
  semantic     ten paraphrases drawn uniformly per call

Metrics per condition: distinct outputs, top-1 share, Shannon entropy (nats).

Outputs: results/openended/openended_survey_{model}.json + raw JSONL
(shipped grid: data/ladder/openended_survey_*.json).
Needs torch + transformers.
Run: CUDA_VISIBLE_DEVICES=0,1 python run.py openended --model llama31-instruct
"""
import json
import math
import os
import random
import re

os.environ.setdefault("USE_TF", "0")

from silicon import results_dir
from silicon.ladder import MODELS

N = 200
BATCH = 24
SEED = 13

TASKS = {
    "important_problem": {
        "fixed": "In your view, what is the most important problem facing the United States today? Answer with a short phrase only.",
        "semantic": [
            "In your view, what is the most important problem facing the United States today? Answer with a short phrase only.",
            "What do you consider the single biggest problem the country faces right now? Short phrase only.",
            "Name the most pressing issue facing Americans today. Answer in a few words.",
            "If you had to pick one, what is the top problem confronting the United States? Brief phrase only.",
            "What national problem concerns you most at present? Answer with a short phrase.",
            "Which issue facing the country today matters most? A few words only.",
            "What is the most serious challenge the United States faces today? Short answer only.",
            "Identify the most important issue for the nation right now. Phrase only.",
            "In a national survey, you are asked for the country's most important problem. Your short answer:",
            "What single problem should Americans be most worried about today? Brief phrase.",
        ],
    },
    "news_source": {
        "fixed": "Which single news outlet do you rely on most for news? Output only the outlet's name.",
        "semantic": [
            "Which single news outlet do you rely on most for news? Output only the outlet's name.",
            "What is your primary source of news? Name the outlet only.",
            "Name the one news organization you turn to most often. Outlet name only.",
            "Where do you get most of your news? Answer with a single outlet name.",
            "If you could keep only one news source, which would it be? Name only.",
            "Which news outlet do you check first each day? Output the name only.",
            "What news source do you trust most? Outlet name only.",
            "Name your main news outlet. Nothing but the name.",
            "In a media survey, you are asked for your primary news source. Your answer (name only):",
            "Which outlet provides most of the news you consume? Name only.",
        ],
    },
    "policy_priority": {
        "fixed": "Name one policy area Congress should prioritize this year. Answer with a short phrase only.",
        "semantic": [
            "Name one policy area Congress should prioritize this year. Answer with a short phrase only.",
            "Which policy issue should lawmakers focus on first this year? Short phrase only.",
            "If Congress could advance one policy area this session, which should it be? Brief phrase.",
            "What legislative priority matters most right now? Answer in a few words.",
            "Name the policy domain most deserving of congressional attention. Phrase only.",
            "Which area of policy should the federal government prioritize? Short answer only.",
            "Pick one policy area for Congress to act on this year. A few words only.",
            "What should top the legislative agenda this year? Brief phrase only.",
            "In a civic survey, you are asked which policy area Congress should prioritize. Your short answer:",
            "Which single policy issue should receive priority in Washington this year? Phrase only.",
        ],
    },
    "respondent_occupation": {
        "fixed": "Generate a plausible occupation for a synthetic survey respondent. Output only the occupation.",
        "semantic": [
            "Generate a plausible occupation for a synthetic survey respondent. Output only the occupation.",
            "Assign an occupation to a simulated survey participant. Occupation only.",
            "A synthetic respondent needs an occupation field. Provide one. Output the occupation only.",
            "For a simulated panelist profile, generate the occupation entry. Occupation only.",
            "Produce one occupation for an artificial survey respondent. Output only the job title.",
            "Fill in the occupation attribute for a generated respondent. Job title only.",
            "What occupation should a synthetic panel member have? Output the occupation only.",
            "Generate the employment field for a simulated respondent record. Occupation only.",
            "Provide an occupation for a randomly generated survey persona. Occupation only.",
            "Create one plausible occupation entry for a synthetic respondent. Output only the occupation.",
        ],
    },
    "respondent_city": {
        "fixed": "Generate a plausible U.S. city of residence for a synthetic survey respondent. Output only the city name.",
        "semantic": [
            "Generate a plausible U.S. city of residence for a synthetic survey respondent. Output only the city name.",
            "Assign a U.S. city of residence to a simulated survey participant. City name only.",
            "A synthetic respondent needs a city field. Provide a U.S. city. Name only.",
            "For a simulated panelist profile, generate the city of residence. U.S. city name only.",
            "Produce one U.S. city for an artificial survey respondent's location. City only.",
            "Fill in the residence city for a generated U.S. respondent. City name only.",
            "In which U.S. city should a synthetic panel member live? Output the city only.",
            "Generate the location field for a simulated respondent record. U.S. city only.",
            "Provide a U.S. city of residence for a randomly generated survey persona. City only.",
            "Create one plausible U.S. city entry for a synthetic respondent. Output only the city.",
        ],
    },
}

NONSEMANTIC_TRANSFORMS = [
    lambda p: p,
    lambda p: p.replace(". ", ".  "),
    lambda p: p.replace(". ", " - ", 1),
    lambda p: " " + p + " ",
    lambda p: p.replace(". ", ";\n", 1),
    lambda p: p.replace(".", "...", 1),
]


def parse_answer(text):
    if not text:
        return None
    line = text.strip().split("\n")[0]
    line = re.sub(r"[\"'.!]", "", line).strip()
    return line.title() if line else None


def main(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = results_dir("openended")
    raw_dir = results_dir("openended/raw")

    key = args.model
    if key not in MODELS:
        raise SystemExit(f"unknown checkpoint key {key!r}; valid keys: {', '.join(MODELS)}")
    spec = MODELS[key]
    tok = AutoTokenizer.from_pretrained(spec["hf"])
    model = AutoModelForCausalLM.from_pretrained(
        spec["hf"], torch_dtype=torch.float16, device_map="auto")
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # decoder-only generation needs left padding

    @torch.no_grad()
    def generate(users, max_new=16):
        chat_kwargs = spec.get("chat_kwargs", {})
        prompts = [tok.apply_chat_template([{"role": "user", "content": u}],
                                           tokenize=False, add_generation_prompt=True,
                                           **chat_kwargs)
                   for u in users]
        outputs = []
        for i in range(0, len(prompts), BATCH):
            enc = tok(prompts[i:i + BATCH], return_tensors="pt", padding=True).to(model.device)
            torch.manual_seed(SEED + i)
            gen = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0,
                                 max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
            for row in gen:
                outputs.append(tok.decode(row[enc.input_ids.shape[1]:],
                                          skip_special_tokens=True))
        return outputs

    rng = random.Random(SEED)
    summary = []
    for task, task_spec in TASKS.items():
        conditions = {
            "fixed": [task_spec["fixed"]] * N,
            "nonsemantic": [rng.choice(NONSEMANTIC_TRANSFORMS)(task_spec["fixed"])
                            for _ in range(N)],
            "semantic": [rng.choice(task_spec["semantic"]) for _ in range(N)],
        }
        for condition, users in conditions.items():
            texts = generate(users)
            with open(raw_dir / f"oes_{key}_{task}_{condition}.jsonl", "w") as f:
                for u, t in zip(users, texts):
                    f.write(json.dumps({"user": u, "text": t}) + "\n")
            values = [v for v in (parse_answer(t) for t in texts) if v]
            counts = {}
            for v in values:
                counts[v] = counts.get(v, 0) + 1
            total = len(values)
            entropy = -sum((c / total) * math.log(c / total)
                           for c in counts.values()) if total else None
            top = max(counts.items(), key=lambda kv: kv[1]) if counts else ("", 0)
            row = {"model": key, "task": task, "condition": condition,
                   "n_valid": total, "distinct": len(counts), "top1": top[0],
                   "top1_share": round(top[1] / total, 3) if total else None,
                   "entropy_nats": round(entropy, 3) if entropy is not None else None}
            summary.append(row)
            print(row, flush=True)

    with open(out_dir / f"openended_survey_{key}.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("[written]", out_dir / f"openended_survey_{key}.json")
