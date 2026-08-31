"""100-item OpinionQA expansion: item selection, pipeline runs, and analysis.

Default stage — run baseline + PPA + describe + VS on the 50 new expansion
items (the second half of the 100-item set), N=100 personas
(sample_personas seed=42), the same prompts as the earlier robustness runs;
this module is the canonical shipped source for those prompts. Reads
data/item_list_v2.json. Idempotent: skips cells whose
results/expansion/{key}_{pipeline}.json exists.
  baseline: 100 calls, ascending order, direct phrasing, system+user
  ppa:      100 calls, randomized order x phrasing x position
  describe: 1 call, JSON {distribution: [...]};  vs: 1 call, JSON {responses: [...]}

--select-items stage — rebuild the 50-item expansion set by the rule fixed
before selection:
  1. Same 7 waves as the original set (W26, W27, W34, W42, W43, W50, W54).
  2. 5-pt Likert with usable Pew ground truth.
  3. Exclude the original 50 items (13 curated + 37 earlier additions).
  4. Dedup-by-key (skip multi-statement matrix items).
  5. Sort by key ascending; take first 50.
  6. If <50 remain, extend the wave list (W92, W36, ...) until 50 reached.
Domain labels use the same programmatic FACTS/VALUES rule as the original
set; the original set's entropy median is preserved as the split threshold.
Outputs of this stage are shipped (data/item_list_v2.json,
item_labels_v2.json, item_entropy_v2.json), so a rerun is only needed to
rebuild the item set from scratch. Needs OpinionQA data under
data/human_resp/ and the original 50-item run's item files under
PROJECT_ROOT.

--analyze stage — combine the two 50-item batches into the 100-item tables:
primary pipeline table, domain-by-entropy 2x2, bootstrap CIs, and the
decision checks fixed before the expansion was run. Reads the original-50
per-item table (per_item_50.csv, not part of this archive; available on
request), the describe/VS rerun summaries under PROJECT_ROOT (not
redistributed), and the per-item JSONs from the default stage. No API key.

Outputs: results/expansion/.
Run: OPENAI_API_KEY=... python run.py expansion [--select-items | --analyze]
"""
import ast
import json
import os
import random
import zlib
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd

from silicon import DATA, PROJECT, results_dir, require_env

client = None
MODEL = os.environ.get("SIM_MODEL", "gpt-4o")
N_THREADS = 16
N_PERSONAS = 100
SEED = 42
N_BOOT = 1000

ORIG50_DIR = PROJECT / "paper_extensions/robustness/exp_e6_50items"
VS_RERUN_DIR = PROJECT / "paper_extensions/robustness/exp_e9_vs"

ORDERINGS = {
    "ascending":  [1, 2, 3, 4, 5],
    "descending": [5, 4, 3, 2, 1],
    "shuffled_a": [3, 5, 1, 4, 2],
    "shuffled_b": [2, 4, 1, 5, 3],
}
PHRASINGS = {
    "direct":      "{q}",
    "indirect":    "On a scale from 1 to 5, how would you rate the following question: {q}",
    "third_person":"How would you, in your role, respond to this question: {q}",
}
POSITIONS = ["system_then_user", "user_only", "user_with_recap"]


def _ensure_client():
    global client
    if client is None:
        from openai import OpenAI
        client = OpenAI(api_key=require_env("OPENAI_API_KEY"))


# ---- prompt builders (canonical shipped source for the Argyle/PPA/describe/VS prompts) ----

def build_argyle(item, persona, ord_n="ascending", phr_n="direct",
                  pos_n="system_then_user"):
    labels = item["likert_labels"]
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
    options = "\n".join(f"  {i+1}. {opt}" for i, opt in enumerate(item["likert_to_pew"]))
    user = (
        f"Estimate the distribution of responses to the following Pew survey "
        f"item across a representative sample of U.S. adults. Output a JSON "
        f"object with a single key 'distribution' whose value is a list of "
        f"5 probabilities summing to 1.0, in the order of options 1 through 5.\n\n"
        f"Question: {item['question']}\n\n"
        f"Options:\n{options}"
    )
    return [{"role": "user", "content": user}]


def build_vs(item):
    options = "\n".join(f"  - {opt}" for opt in item["likert_to_pew"])
    user = (
        "For the following survey question, list each possible response option "
        "along with the proportion of US adults who would select it. "
        "Output as JSON: a single key 'responses' whose value is a list of "
        "objects, each with 'response' (string) and 'probability' (number). "
        "Probabilities must sum to 1. Be precise; this is being used for "
        "downstream estimation.\n\n"
        f"Question: {item['question']}\n\nResponse options:\n{options}"
    )
    return [{"role": "user", "content": user}]


# ---- helpers ----

def parse_digit(text):
    if not text: return None
    m = re.search(r"[1-5]", text)
    return int(m.group()) if m else None


def displayed_to_canonical(d, ordering):
    if d is None or d < 1 or d > 5: return None
    return ordering[d - 1]


def call_text(messages, temperature=1.0, max_tokens=10):
    for attempt in range(4):
        try:
            r = client.chat.completions.create(model=MODEL, messages=messages,
                                                temperature=temperature, max_tokens=max_tokens)
            return r.choices[0].message.content
        except Exception:
            if attempt == 3: raise
            time.sleep(2 ** attempt)


def call_json(messages, max_tokens=600):
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


def empirical(samples, k=5):
    counts = np.bincount([s for s in samples if s is not None], minlength=k + 1)[1:k + 1].astype(float)
    if counts.sum() == 0: return np.full(k, 1.0 / k)
    return counts / counts.sum()


def tv(p, q):
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


def load_new_50():
    item_list = json.load(open(DATA / "item_list_v2.json"))
    items = []
    for d in item_list["new_item_details"]:
        items.append({
            "id": d["key"], "key": d["key"],
            "wave": f"American_Trends_Panel_{d['wave']}",
            "question": d["question"],
            "likert_to_pew": d["options_pew_order"],
            "likert_labels": d["options_pew_order"],
            "pew_dist": d["pew_dist"],
        })
    return items


# ---- default stage: run the four pipelines on the 50 new items ----

def run_baseline(item, personas, out_dir, raw_dir):
    key = item["key"]
    out_path = out_dir / f"{key}_baseline.json"
    if out_path.exists(): return json.load(open(out_path))

    def worker(persona):
        msgs, ordering = build_argyle(item, persona)
        txt = call_text(msgs, temperature=1.0, max_tokens=10)
        d = parse_digit(txt)
        return {"persona_id": persona["id"], "raw": txt,
                "displayed": d, "canonical": displayed_to_canonical(d, ordering)}
    with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
        per = list(pool.map(worker, personas))
    with open(raw_dir / f"{key}_baseline_raw.jsonl", "w") as f:
        for c in per: f.write(json.dumps(c) + "\n")
    samples = [c["canonical"] for c in per]
    p_hat = empirical(samples)
    summary = {"item_key": key, "pipeline": "baseline", "n": len(personas),
                "n_valid": sum(1 for s in samples if s is not None),
                "p_hat": p_hat.tolist(), "pew_dist": item["pew_dist"],
                "tv": tv(p_hat, item["pew_dist"])}
    json.dump(summary, open(out_path, "w"), indent=2)
    return summary


def run_ppa(item, personas, rng, out_dir, raw_dir):
    key = item["key"]
    out_path = out_dir / f"{key}_ppa.json"
    if out_path.exists(): return json.load(open(out_path))

    ord_names = list(ORDERINGS); phr_names = list(PHRASINGS); pos_names = POSITIONS
    def worker(persona):
        ord_n = rng.choice(ord_names)
        phr_n = rng.choice(phr_names)
        pos_n = rng.choice(pos_names)
        msgs, ordering = build_argyle(item, persona, ord_n, phr_n, pos_n)
        txt = call_text(msgs, temperature=1.0, max_tokens=10)
        d = parse_digit(txt)
        return {"persona_id": persona["id"],
                "perturbation": {"order": ord_n, "phrasing": phr_n, "position": pos_n},
                "raw": txt, "displayed": d,
                "canonical": displayed_to_canonical(d, ordering)}
    with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
        per = list(pool.map(worker, personas))
    with open(raw_dir / f"{key}_ppa_raw.jsonl", "w") as f:
        for c in per: f.write(json.dumps(c) + "\n")
    samples = [c["canonical"] for c in per]
    p_hat = empirical(samples)
    summary = {"item_key": key, "pipeline": "ppa", "n": len(personas),
                "n_valid": sum(1 for s in samples if s is not None),
                "p_hat": p_hat.tolist(), "pew_dist": item["pew_dist"],
                "tv": tv(p_hat, item["pew_dist"])}
    json.dump(summary, open(out_path, "w"), indent=2)
    return summary


def run_describe(item, out_dir):
    key = item["key"]
    out_path = out_dir / f"{key}_describe.json"
    if out_path.exists(): return json.load(open(out_path))
    text = call_json(build_describe(item))
    try:
        d = json.loads(text)
        dist = d.get("distribution") or list(d.values())[0]
        dist = [float(x) for x in dist]
        s = sum(dist)
        dist = [x / s for x in dist] if s > 1e-9 else None
    except Exception:
        dist = None
    summary = {"item_key": key, "pipeline": "describe", "raw": text,
                "p_hat": dist, "pew_dist": item["pew_dist"],
                "tv": tv(dist, item["pew_dist"]) if dist and len(dist) == 5 else None}
    json.dump(summary, open(out_path, "w"), indent=2)
    return summary


def run_vs(item, out_dir):
    key = item["key"]
    out_path = out_dir / f"{key}_vs.json"
    if out_path.exists(): return json.load(open(out_path))
    text = call_json(build_vs(item))
    try:
        d = json.loads(text)
        entries = d.get("responses") or list(d.values())[0]
        prob_map = {}
        for entry in entries:
            if isinstance(entry, dict):
                k = entry.get("response") or entry.get("option") or list(entry.values())[0]
                v = entry.get("probability") or entry.get("p") or 0.0
                prob_map[str(k).strip().lower()] = float(v)
        dist = []
        for opt in item["likert_to_pew"]:
            ko = opt.strip().lower()
            if ko in prob_map:
                dist.append(prob_map[ko])
            else:
                hit = next((v for k, v in prob_map.items()
                             if k.startswith(ko[:max(1, len(ko) - 2)])), None)
                dist.append(hit if hit is not None else 0.0)
        s = sum(dist)
        dist = [x / s for x in dist] if s > 1e-9 else None
    except Exception:
        dist = None
    summary = {"item_key": key, "pipeline": "vs", "raw": text,
                "p_hat": dist, "pew_dist": item["pew_dist"],
                "tv": tv(dist, item["pew_dist"]) if dist and len(dist) == 5 else None}
    json.dump(summary, open(out_path, "w"), indent=2)
    return summary


def run_pipelines():
    from silicon.opinionqa import sample_personas

    _ensure_client()
    out_dir = results_dir("expansion")
    raw_dir = results_dir("expansion/raw")
    items = load_new_50()
    personas = sample_personas(N_PERSONAS, seed=SEED)
    print(f"Loaded {len(items)} new items, {len(personas)} personas")

    print("\nBASELINE Argyle (50 items x 100 personas)")
    for it in items:
        s = run_baseline(it, personas, out_dir, raw_dir)
        print(f"  {s['item_key']:25s}  TV={s['tv']:.3f}  n_valid={s['n_valid']}")

    print("\nPPA (50 items x 100 personas)")
    for it in items:
        rng = random.Random(SEED + zlib.crc32(it["key"].encode()) % 100000)  # stable across processes (PYTHONHASHSEED-independent)
        s = run_ppa(it, personas, rng, out_dir, raw_dir)
        print(f"  {s['item_key']:25s}  TV={s['tv']:.3f}  n_valid={s['n_valid']}")

    print("\nDESCRIBE pathway (50 items x 1 call)")
    for it in items:
        s = run_describe(it, out_dir)
        print(f"  {s['item_key']:25s}  TV={s['tv']}")

    print("\nVS (50 items x 1 call)")
    for it in items:
        s = run_vs(it, out_dir)
        print(f"  {s['item_key']:25s}  TV={s['tv']}")

    meta = {"timestamp": datetime.utcnow().isoformat() + "Z",
            "model_string": MODEL, "n_personas": N_PERSONAS,
            "n_items_new": len(items)}
    json.dump(meta, open(out_dir / "run_meta.json", "w"), indent=2)


# ---- --select-items stage ----

PRIMARY_WAVES = [f"American_Trends_Panel_W{w}"
                 for w in [26, 27, 34, 42, 43, 50, 54]]
EXTENSION_WAVES = [f"American_Trends_Panel_W{w}"
                   for w in [92, 36, 41, 49, 82, 45, 32, 29]]

# Programmatic FACTS/VALUES rule (identical to the original 50-item selection).
VALUES_PATTERNS = [
    "good or bad", "more or less", "agree or disagree", "favor or oppose",
    "do you think", "would you support", "should", "right or wrong",
    "moral", "fairness", "discrimin", "racis", "gender", "religion",
    "abortion", "lgbt", "gay", "homosexual", "interracial", "immig",
    "fair", "wealth", "rich and poor", "rich", "poor", "fair share",
    "diverse", "majority", "minority", "society",
]
FACTS_PATTERNS = [
    "how often", "how much", "how many", "frequency", "how safe",
    "how worried", "how concerned", "how likely", "in the past",
    "the last time", "have you", "do you ", "everyday", "daily",
    "weekly", "do you eat", "how regularly", "consume", "use of",
]


def parse_option_mapping(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        return {}


def is_5pt_likert(option_mapping):
    if not isinstance(option_mapping, dict): return False
    keys_substantive = set()
    for k, v in option_mapping.items():
        if isinstance(k, (int, float)) and k in (1, 2, 3, 4, 5, 1.0, 2.0, 3.0, 4.0, 5.0):
            if v != "Refused":
                keys_substantive.add(int(k) if not isinstance(k, int) else k)
    return keys_substantive == {1, 2, 3, 4, 5}


def collect_from_waves(wave_list, exclude_keys):
    rows = []
    for wave in wave_list:
        info_path = DATA / "human_resp" / wave / "info.csv"
        if not info_path.exists(): continue
        info = pd.read_csv(info_path)
        key_counts = info["key"].value_counts()
        seen = set()
        for _, row in info.iterrows():
            key = row.get("key")
            if not isinstance(key, str): continue
            if key in seen: continue
            seen.add(key)
            if key in exclude_keys: continue
            if key_counts.get(key, 0) > 1: continue
            opt_map = parse_option_mapping(row.get("option_mapping"))
            if not is_5pt_likert(opt_map): continue
            options_in_order = [opt_map[float(i)] if float(i) in opt_map else opt_map[i] for i in range(1, 6)]
            question = row.get("question") if isinstance(row.get("question"), str) else ""
            rows.append({"key": key, "wave": wave,
                          "question": question,
                          "options_pew_order": options_in_order})
    return rows


def compute_pew_dist(item):
    resp_path = DATA / "human_resp" / item["wave"] / "responses.csv"
    if not resp_path.exists(): return None
    df = pd.read_csv(resp_path, low_memory=False)
    if item["key"] not in df.columns: return None
    vals = df[item["key"]].astype(str)
    options = item["options_pew_order"]
    counts = np.zeros(5)
    for v in vals:
        v = v.strip()
        for i, opt in enumerate(options):
            if v == opt:
                counts[i] += 1; break
    if counts.sum() < 10: return None
    return (counts / counts.sum()).tolist()


def shannon_entropy_normalized(p, K=5):
    p = np.array([x for x in p if x > 0])
    if len(p) == 0: return 0.0
    h = -np.sum(p * np.log(p))
    return float(h / np.log(K))


def domain_label(question):
    q = (question or "").lower()
    v_hits = sum(1 for p in VALUES_PATTERNS if p in q)
    f_hits = sum(1 for p in FACTS_PATTERNS if p in q)
    if f_hits > v_hits: return "FACTS"
    return "VALUES"


def select_items():
    out_dir = results_dir("expansion")

    if not (ORIG50_DIR / "item_list.json").exists():
        raise SystemExit(
            f"--select-items needs the original experiment tree under {ORIG50_DIR} "
            "(set PROJECT_ROOT). The selected items already ship as "
            "data/item_list_v2.json; rerunning selection is only needed to "
            "audit the selection rule itself.")
    orig50_item_list = json.load(open(ORIG50_DIR / "item_list.json"))
    orig50_keys = set(orig50_item_list["all_50"])
    orig50_labels = json.load(open(ORIG50_DIR / "item_labels.json"))
    orig50_entropy = json.load(open(ORIG50_DIR / "item_entropy.json"))
    orig50_median = orig50_entropy["median"]

    print("Collecting from primary waves:", [w[-3:] for w in PRIMARY_WAVES])
    candidates = collect_from_waves(PRIMARY_WAVES, orig50_keys)
    print(f"  candidates after dedup + original-50-exclude: {len(candidates)}")
    usable = []
    for c in candidates:
        d = compute_pew_dist(c)
        if d is None: continue
        c["pew_dist"] = d
        c["pew_entropy_normalized"] = shannon_entropy_normalized(d)
        usable.append(c)
    print(f"  after filtering for usable Pew GT: {len(usable)}")

    waves_used = PRIMARY_WAVES.copy()
    if len(usable) < 50:
        for w in EXTENSION_WAVES:
            print(f"  <50 — extending to {w[-3:]}")
            extra = collect_from_waves([w], orig50_keys)
            for c in extra:
                d = compute_pew_dist(c)
                if d is None: continue
                c["pew_dist"] = d
                c["pew_entropy_normalized"] = shannon_entropy_normalized(d)
                usable.append(c)
            waves_used.append(w)
            print(f"    now {len(usable)} candidates")
            if len(usable) >= 50: break

    usable.sort(key=lambda x: x["key"])
    new_items = usable[:50]
    print("\nSelected first 50 new items.")
    print(f"  Keys: {[it['key'] for it in new_items]}")

    item_labels_v2 = {}
    for it in new_items:
        item_labels_v2[it["key"]] = {"domain": domain_label(it["question"]),
                                       "wave": it["wave"][-3:],
                                       "question_excerpt": (it["question"] or "")[:100]}
    with open(out_dir / "item_labels_v2.json", "w") as f:
        json.dump(item_labels_v2, f, indent=2)
    print(f"Wrote item_labels_v2.json ({len(item_labels_v2)} new items)")

    new_entropies = {it["key"]: it["pew_entropy_normalized"] for it in new_items}
    # The original 50-item median is the preserved threshold.
    combined_entropies = {**orig50_entropy["per_item"], **new_entropies}
    new_median_100 = sorted(combined_entropies.values())[len(combined_entropies) // 2]
    with open(out_dir / "item_entropy_v2.json", "w") as f:
        json.dump({
            "per_item_new": new_entropies,
            "preserved_threshold_e6_median": orig50_median,
            "fresh_median_across_100": new_median_100,
        }, f, indent=2)
    print(f"Wrote item_entropy_v2.json. Preserved original-50 median={orig50_median:.4f}, "
          f"fresh 100-item median={new_median_100:.4f}")

    item_list_v2 = {
        "new_50_keys": [it["key"] for it in new_items],
        "all_100_keys": orig50_item_list["all_50"] + [it["key"] for it in new_items],
        "waves_scanned": [w[-3:] for w in waves_used],
        "waves_scanned_note": ("waves scanned for candidate items; a scanned wave "
                               "can contribute zero items with usable 5-point "
                               "ground truth"),
        "new_item_details": [
            {"key": it["key"], "wave": it["wave"][-3:],
              "question": it["question"][:200],
              "options_pew_order": it["options_pew_order"],
              "pew_dist": it["pew_dist"],
              "pew_entropy_normalized": it["pew_entropy_normalized"]}
            for it in new_items
        ]
    }
    with open(out_dir / "item_list_v2.json", "w") as f:
        json.dump(item_list_v2, f, indent=2)
    print(f"Wrote item_list_v2.json ({len(item_list_v2['new_50_keys'])} new keys)")

    print("\nPRE-REGISTERED 100-ITEM 2x2 (using preserved original-50 median)")
    cells = {}
    for k in item_list_v2["all_100_keys"]:
        dom = (orig50_labels.get(k) or item_labels_v2.get(k) or {}).get("domain", "?")
        ent = combined_entropies.get(k, 0)
        bucket = "HI_ENT" if ent >= orig50_median else "LO_ENT"
        cells.setdefault((dom, bucket), []).append(k)
    for (dom, bucket), keys in sorted(cells.items()):
        print(f"  {dom:7s} x {bucket:7s}: n={len(keys)}")

    wave_counts = {}
    for it in new_items:
        w = it["wave"][-3:]
        wave_counts[w] = wave_counts.get(w, 0) + 1
    print("\nWave distribution (new 50):")
    for w in sorted(wave_counts):
        print(f"  {w}: {wave_counts[w]}")


# ---- --analyze stage ----

def load_original_50():
    p50 = DATA / "ppa_per_item" / "per_item_50.csv"
    if not p50.exists():
        raise SystemExit(
            "expansion --analyze needs the original-50 per-item table "
            "(ppa_per_item/per_item_50.csv), which is not part of this "
            "archive; it is available from the authors on request. The "
            "authoritative 100-item numbers ship in per_item_100.csv and "
            "are re-checked by `python analyze.py verify`.")
    df = pd.read_csv(p50)
    desc_path = VS_RERUN_DIR / "results" / "describe_summary.csv"
    vs_path = VS_RERUN_DIR / "results" / "vs_summary.csv"
    if desc_path.exists() and vs_path.exists():
        desc = pd.read_csv(desc_path).rename(columns={"tv": "describe_tv_rerun"})
        vs = pd.read_csv(vs_path).rename(columns={"tv": "vs_tv"})
        df = df.merge(desc[["item_key", "describe_tv_rerun"]], on="item_key", how="left")
        df = df.merge(vs[["item_key", "vs_tv"]], on="item_key", how="left")
        df["describe_tv"] = df["describe_tv_rerun"].combine_first(df["describe_tv"])
    else:
        print(f"note: original-50 rerun summaries not found under {VS_RERUN_DIR}; "
              "using the describe_tv column of per_item_50.csv and leaving vs_tv "
              "empty for those items (the shipped per_item_100.csv carries the "
              "full four-pipeline table). Numbers printed below come from this "
              "partial rebuild and are NOT the shipped results; "
              "`python analyze.py verify` re-checks the authoritative numbers.")
        df["vs_tv"] = float("nan")
    df["source"] = "E6_50"
    return df[["item_key", "baseline_tv", "ppa_tv", "describe_tv", "vs_tv",
                "domain", "pew_entropy", "entropy_bucket", "source"]]


def load_new_50_results(results):
    labels = json.load(open(DATA / "item_labels_v2.json"))
    entropy = json.load(open(DATA / "item_entropy_v2.json"))
    orig50_median = entropy["preserved_threshold_e6_median"]
    new_keys = json.load(open(DATA / "item_list_v2.json"))["new_50_keys"]

    rows = []
    for key in new_keys:
        bp = results / f"{key}_baseline.json"
        pp = results / f"{key}_ppa.json"
        dp = results / f"{key}_describe.json"
        vp = results / f"{key}_vs.json"
        if not (bp.exists() and pp.exists() and dp.exists() and vp.exists()):
            continue
        b = json.load(open(bp))
        p = json.load(open(pp))
        d = json.load(open(dp))
        v = json.load(open(vp))
        ent = entropy["per_item_new"][key]
        rows.append({
            "item_key": key,
            "baseline_tv": b["tv"], "ppa_tv": p["tv"],
            "describe_tv": d["tv"], "vs_tv": v["tv"],
            "domain": labels[key]["domain"],
            "pew_entropy": ent,
            "entropy_bucket": "HI_ENT" if ent >= orig50_median else "LO_ENT",
            "source": "E13_50",
        })
    return pd.DataFrame(rows)


def ci(arr):
    arr = np.asarray(arr); arr = arr[~np.isnan(arr)]
    if len(arr) == 0: return (np.nan, np.nan, np.nan)
    return float(np.percentile(arr, 2.5)), float(np.median(arr)), float(np.percentile(arr, 97.5))


def analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    out_dir = results_dir("expansion")
    df_orig = load_original_50()
    df_new = load_new_50_results(out_dir)
    df = pd.concat([df_orig, df_new], ignore_index=True)
    print(f"Loaded {len(df_orig)} original items + {len(df_new)} new items = {len(df)} total")
    df.to_csv(out_dir / "per_item_100.csv", index=False)

    df_complete = df.dropna(subset=["baseline_tv", "ppa_tv", "describe_tv", "vs_tv"])
    print(f"  {len(df_complete)} items have all four pipelines")

    print("\n--- Sanity: marginal means by source ---")
    for src in ["E6_50", "E13_50"]:
        s = df_complete[df_complete["source"] == src]
        print(f"  {src}: n={len(s)}  baseline={s['baseline_tv'].mean():.4f}  "
              f"ppa={s['ppa_tv'].mean():.4f}  describe={s['describe_tv'].mean():.4f}  "
              f"vs={s['vs_tv'].mean():.4f}")

    print("\nPRIMARY TABLE — 100 items")
    primary_rows = []
    for pipe in ["baseline", "ppa", "describe", "vs"]:
        col = f"{pipe}_tv"
        valid = df_complete[col].dropna()
        primary_rows.append({"pipeline": pipe, "n": len(valid),
                              "mean_tv": float(valid.mean()),
                              "sem_tv": float(valid.sem())})
    primary_df = pd.DataFrame(primary_rows)
    print(primary_df.to_string(index=False))
    primary_df.to_csv(out_dir / "primary_table_100.csv", index=False)

    print("\n--- Wilcoxon paired tests ---")
    paired = df_complete.dropna(subset=["baseline_tv", "ppa_tv"])
    diffs_ppa = paired["ppa_tv"] - paired["baseline_tv"]
    stat_ppa, p_two_ppa = stats.wilcoxon(diffs_ppa)
    _, p_one_ppa = stats.wilcoxon(diffs_ppa, alternative="less")
    print(f"  PPA vs baseline (N={len(diffs_ppa)}): "
          f"delta={diffs_ppa.mean():+.4f}  stat={stat_ppa}  p_two={p_two_ppa:.4g}  "
          f"p_one(PPA<base)={p_one_ppa:.4g}")

    paired_dv = df_complete.dropna(subset=["describe_tv", "vs_tv"])
    diffs_dv = paired_dv["describe_tv"] - paired_dv["vs_tv"]
    stat_dv, p_two_dv = stats.wilcoxon(diffs_dv)
    print(f"  describe vs VS (N={len(diffs_dv)}): "
          f"delta={diffs_dv.mean():+.4f}  stat={stat_dv}  p_two={p_two_dv:.4g}")

    print("\n2x2 PRE-REGISTERED SCOPE ANALYSIS")
    cells = []
    for (dom, ent), grp in df_complete.groupby(["domain", "entropy_bucket"]):
        d = grp["ppa_tv"] - grp["baseline_tv"]
        try:
            wp = float(stats.wilcoxon(d, alternative="less").pvalue) if len(d) >= 6 else None
        except Exception:
            wp = None
        cells.append({
            "domain": dom, "entropy": ent, "n": len(grp),
            "mean_baseline_tv": float(grp["baseline_tv"].mean()),
            "mean_ppa_tv": float(grp["ppa_tv"].mean()),
            "mean_delta": float(d.mean()),
            "wilcoxon_p_one_sided": wp})
    cells_df = pd.DataFrame(cells)
    print(cells_df.to_string(index=False))
    cells_df.to_csv(out_dir / "twoxtwo_100.csv", index=False)

    facts = df_complete[df_complete["domain"] == "FACTS"]
    values = df_complete[df_complete["domain"] == "VALUES"]
    facts_delta = (facts["ppa_tv"] - facts["baseline_tv"]).mean()
    values_delta = (values["ppa_tv"] - values["baseline_tv"]).mean()
    print(f"\nMarginal FACTS delta: {facts_delta:+.4f} (n={len(facts)})")
    print(f"Marginal VALUES delta: {values_delta:+.4f} (n={len(values)})")
    print(f"FACTS - VALUES gap: {facts_delta - values_delta:+.4f}")

    rng = np.random.default_rng(SEED)
    n = len(df_complete)
    baseline_means = np.zeros(N_BOOT)
    ppa_means = np.zeros(N_BOOT)
    describe_means = np.zeros(N_BOOT)
    vs_means = np.zeros(N_BOOT)
    ratios = np.zeros(N_BOOT)
    deltas = np.zeros(N_BOOT)

    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        s = df_complete.iloc[idx]
        baseline_means[b] = s["baseline_tv"].mean()
        ppa_means[b] = s["ppa_tv"].mean()
        describe_means[b] = s["describe_tv"].mean()
        vs_means[b] = s["vs_tv"].mean()
        ratios[b] = baseline_means[b] / describe_means[b] if describe_means[b] > 0 else np.nan
        deltas[b] = (s["ppa_tv"] - s["baseline_tv"]).mean()

    # Per-cell bootstrap (preserve cell n).
    cell_deltas = {}
    cell_wilcoxon_p = {}
    for (dom, ent), grp in df_complete.groupby(["domain", "entropy_bucket"]):
        cell_key = f"{dom}_{ent}"
        cn = len(grp)
        arr_d = np.zeros(N_BOOT)
        arr_p = np.full(N_BOOT, np.nan)
        for b in range(N_BOOT):
            idx = rng.integers(0, cn, size=cn)
            s = grp.iloc[idx]
            d_vals = (s["ppa_tv"] - s["baseline_tv"]).values
            arr_d[b] = d_vals.mean()
            if cn >= 6:
                try:
                    arr_p[b] = stats.wilcoxon(d_vals, alternative="less").pvalue
                except Exception:
                    pass
        cell_deltas[cell_key] = arr_d
        cell_wilcoxon_p[cell_key] = arr_p

    print("\nBOOTSTRAP CIs")
    print(f"\n{'Statistic':30s}  {'Observed':>10s}  {'Median':>10s}  {'CI_lo':>10s}  {'CI_hi':>10s}")
    obs = {
        "Mean baseline TV":          (baseline_means, df_complete["baseline_tv"].mean()),
        "Mean PPA TV":               (ppa_means, df_complete["ppa_tv"].mean()),
        "Mean describe TV":          (describe_means, df_complete["describe_tv"].mean()),
        "Mean VS TV":                (vs_means, df_complete["vs_tv"].mean()),
        "Ratio baseline/describe":   (ratios, df_complete["baseline_tv"].mean() / df_complete["describe_tv"].mean()),
        "PPA delta vs baseline":     (deltas, (df_complete["ppa_tv"] - df_complete["baseline_tv"]).mean()),
    }
    ci_rows = []
    for name, (arr, observed) in obs.items():
        lo, med, hi = ci(arr)
        print(f"{name:30s}  {observed:>10.4f}  {med:>10.4f}  [{lo:>7.4f}, {hi:>7.4f}]")
        ci_rows.append({"statistic": name.lower().replace(" ", "_"),
                         "observed": observed, "boot_median": med,
                         "ci_lo": lo, "ci_hi": hi})

    print(f"\n{'Cell':25s}  {'n':>3s}  {'Observed d':>12s}  {'Median d':>10s}  {'CI_lo':>10s}  {'CI_hi':>10s}  {'frac p<0.05':>12s}")
    for cell_key, arr in cell_deltas.items():
        parts = cell_key.rsplit("_", 2)
        dom = parts[0]; ent = "_".join(parts[1:])
        cell = df_complete[(df_complete["domain"] == dom) & (df_complete["entropy_bucket"] == ent)]
        observed_d = (cell["ppa_tv"] - cell["baseline_tv"]).mean()
        lo, med, hi = ci(arr)
        wp_arr = cell_wilcoxon_p[cell_key]
        frac_sig = float(np.mean(wp_arr < 0.05)) if not np.all(np.isnan(wp_arr)) else np.nan
        print(f"  {dom + ' x ' + ent:23s}  {len(cell):3d}  {observed_d:>+12.4f}  {med:>+10.4f}  "
              f"[{lo:>+7.4f}, {hi:>+7.4f}]  {frac_sig:>11.1%}")
        ci_rows.append({"statistic": f"delta_{cell_key}",
                         "observed": float(observed_d), "boot_median": med,
                         "ci_lo": lo, "ci_hi": hi,
                         "frac_wilcoxon_p_lt_05": frac_sig})

    pd.DataFrame(ci_rows).to_csv(out_dir / "ci_summary.csv", index=False)
    pd.DataFrame({
        "baseline_mean": baseline_means, "ppa_mean": ppa_means,
        "describe_mean": describe_means, "vs_mean": vs_means,
        "ratio": ratios, "ppa_delta": deltas,
        **{f"delta_{k}": v for k, v in cell_deltas.items()},
    }).to_csv(out_dir / "bootstrap_100.csv", index=False)

    print("\nPRE-REGISTERED DECISION RULES")
    decisions = []

    ratio_obs = obs["Ratio baseline/describe"][1]
    ratio_lo, _, ratio_hi = ci(ratios)
    if ratio_lo >= 2.0:
        d = f"Ratio: observed {ratio_obs:.2f}, 95% CI [{ratio_lo:.2f}, {ratio_hi:.2f}] EXCLUDES 2.0 -> keep Section 5 framing."
    elif ratio_obs >= 2.0:
        d = f"Ratio: observed {ratio_obs:.2f}, CI [{ratio_lo:.2f}, {ratio_hi:.2f}] crosses 2.0 — report wider CI."
    else:
        d = f"Ratio: observed {ratio_obs:.2f} BELOW 2.0 -> update Section 5 headline."
    print(d); decisions.append(d)

    delta_obs = obs["PPA delta vs baseline"][1]
    if delta_obs <= -0.05 and p_one_ppa < 0.01:
        d = f"PPA delta: {delta_obs:+.4f} with p={p_one_ppa:.4g} -> keep Section 6 framing."
    elif delta_obs < -0.05:
        d = f"PPA delta: {delta_obs:+.4f} OK direction, p={p_one_ppa:.4g} less stringent — note p value."
    elif delta_obs < 0:
        d = f"PPA delta: {delta_obs:+.4f} weakens — soften Section 6 framing."
    else:
        d = f"PPA delta: {delta_obs:+.4f} (no longer negative) — DROP PPA."
    print(d); decisions.append(d)

    confirmed_cell = next((c for c in cells if c["domain"] == "FACTS" and c["entropy"] == "LO_ENT"), None)
    if confirmed_cell and confirmed_cell["n"] >= 20 and abs(facts_delta - values_delta) >= 0.10:
        d = f"FACTS vs VALUES gap: |{facts_delta - values_delta:+.4f}| >= 0.10 with confirmed cell n={confirmed_cell['n']} -> CONFIRMATORY."
    else:
        d = (f"FACTS vs VALUES gap: |{facts_delta - values_delta:+.4f}| (cell n={confirmed_cell['n'] if confirmed_cell else '?'}) "
             f"— stays exploratory.")
    print(d); decisions.append(d)

    if confirmed_cell and confirmed_cell["wilcoxon_p_one_sided"] is not None and confirmed_cell["wilcoxon_p_one_sided"] < 0.05:
        d = (f"Confirmed cell (FACTS_LO_ENT, n={confirmed_cell['n']}): "
              f"delta={confirmed_cell['mean_delta']:+.4f}, p={confirmed_cell['wilcoxon_p_one_sided']:.4g} -> SURVIVES.")
    else:
        d = "Confirmed cell loses significance — remove cell-level confirmatory claim."
    print(d); decisions.append(d)

    if p_two_dv > 0.05:
        d = f"VS vs describe: paired p={p_two_dv:.4g} > 0.05 -> Section 5 equivalence ROBUST on 100 items."
    else:
        d = f"VS vs describe: paired p={p_two_dv:.4g} <= 0.05 -> divergence; investigate per-item."
    print(d); decisions.append(d)

    md_lines = ["# 100-item expansion — decision\n\n"]
    for d in decisions:
        md_lines.append(f"- {d}\n")
    with open(out_dir / "decision.md", "w") as f:
        f.writelines(md_lines)

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    panels = [
        ("Mean baseline TV", baseline_means, obs["Mean baseline TV"][1]),
        ("Mean PPA TV", ppa_means, obs["Mean PPA TV"][1]),
        ("Mean describe TV", describe_means, obs["Mean describe TV"][1]),
        ("Ratio baseline/describe", ratios, ratio_obs),
        ("PPA delta", deltas, delta_obs),
        ("FACTS_LO_ENT delta", cell_deltas.get("FACTS_LO_ENT", np.array([])),
         confirmed_cell["mean_delta"] if confirmed_cell else np.nan),
    ]
    for ax, (name, arr, observed) in zip(axes.flat, panels):
        a = arr[~np.isnan(arr)]
        if len(a) == 0: continue
        ax.hist(a, bins=40, color="#3366cc", edgecolor="white", linewidth=0.4)
        ax.axvline(observed, color="black", linestyle="--", lw=1.2,
                    label=f"observed={observed:.3f}")
        lo, hi = np.percentile(a, [2.5, 97.5])
        ax.axvline(lo, color="#dd5500", linestyle=":", lw=1.0)
        ax.axvline(hi, color="#dd5500", linestyle=":", lw=1.0)
        ax.set_title(name, fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_dir / "fig_bootstrap_100.pdf", dpi=200)
    fig.savefig(out_dir / "fig_bootstrap_100.png", dpi=200)
    plt.close(fig)
    print("\nWrote fig_bootstrap_100.pdf")


def main(args):
    global MODEL
    if getattr(args, "model", None):
        MODEL = args.model
    if getattr(args, "select_items", False):
        select_items()
    elif getattr(args, "analyze", False):
        analyze()
    else:
        run_pipelines()
