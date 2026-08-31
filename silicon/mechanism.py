"""Mechanism tests: why does PPA work?

--entropy-correlation stage — does PPA's per-call entropy increase explain
its TV improvement? 20 items x 5 personas x 50 calls x 2 conditions = 10,000
calls. Decision rule fixed before running: OLS regression
delta_TV ~ delta_H + H_Pew; the OpinionQA framing is kept only if the delta_H
coefficient is positive with p < 0.05.
Outputs: results/mechanism/entropy_correlation/ (per_item_data.csv,
per_cell_data.csv, regression_results.txt, raw_responses.jsonl, decision.txt,
scatter plot). Shipped copies: data/aggregates/entropy_correlation_per_item.csv
and entropy_correlation_regression_results.txt.

Default stage — mechanism tests 1 and 3: does raising per-call entropy
WITHOUT semantic prompt variation (non-semantic perturbation, or temperature)
reproduce PPA's TV improvement? Re-uses the 20-item subset and 5 personas
from the entropy-correlation stage; 50 calls per cell per condition.
Conditions: N1 whitespace-only, N2 punctuation-only, N3 prefix-only,
N4 all non-semantic, T15 standard Argyle at T=1.5
(5 x 20 x 5 x 50 = 25,000 calls).
Needs the entropy-correlation outputs plus, for the delta-TV columns, the
13-item non-semantic summary and the temperature-sweep per-item CSV under
PROJECT_ROOT (not redistributed). Shipped copies of the outputs:
data/aggregates/mechanism_test1_per_cell.csv (fresh runs also write
summary and per-item tables under results/).

Both stages need OpinionQA data under data/human_resp/, the original
50-item run's item files under PROJECT_ROOT, and statsmodels.
Run: OPENAI_API_KEY=... python run.py mechanism [--entropy-correlation]
"""
import json
import os
import random
import zlib
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from silicon import DATA, PROJECT, results_dir, require_env
from silicon.pew_items import ITEMS as ORIGINAL_13, _BASE

client = None
MODEL = os.environ.get("SIM_MODEL", "gpt-4o")
N_CALLS = 50
N_PERSONAS = 5
N_THREADS = 16
SEED = 42

ORIG50_DIR = PROJECT / "paper_extensions/robustness/exp_e6_50items"

# PPA perturbation grid (identical to the applied PPA runs).
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

# Non-semantic perturbation specs (same grid as the 13-item non-semantic run).
WHITESPACE_VARIANTS = [("none", lambda s: s), ("trail", lambda s: s + "\n"),
                        ("double_ws", lambda s: s.replace(" ", "  ", 3)),
                        ("lead", lambda s: " " + s)]
PUNCT_VARIANTS = [("plain", lambda s: s), ("dot", lambda s: s + "."),
                   ("colon", lambda s: s + ":"), ("dash", lambda s: s.replace(" - ", " -- "))]
PREFIX_VARIANTS = [("", ""), ("ok", "OK.\n"), ("sure", "Sure.\n"), ("ans", "Answer:\n")]


def _ensure_client():
    global client
    if client is None:
        from openai import OpenAI
        client = OpenAI(api_key=require_env("OPENAI_API_KEY"))


def _select_5_personas():
    from silicon.opinionqa import sample_personas
    all_p = sample_personas(100, seed=SEED)
    rng = random.Random(SEED)
    indices = rng.sample(range(100), N_PERSONAS)
    return [all_p[i] for i in indices]


def _curated_items():
    items = {}
    for it in ORIGINAL_13:
        path = os.path.join(_BASE, it["wave"], "responses.csv")
        df = pd.read_csv(path, low_memory=False)
        if it["key"] not in df.columns: continue
        vals = df[it["key"]].astype(str)
        counts = np.zeros(5)
        for v in vals:
            v = v.strip()
            for i, opt in enumerate(it["likert_to_pew"]):
                if v == opt: counts[i] += 1; break
        if counts.sum() < 10: continue
        items[it["key"]] = {"key": it["key"], "wave": it["wave"],
                             "question": it["question"], "options": it["likert_to_pew"],
                             "pew_dist": (counts / counts.sum()).tolist()}
    for src in [ORIG50_DIR / "item_list.json", DATA / "item_list_v2.json"]:
        for d in json.load(open(src)).get("new_item_details", []):
            items[d["key"]] = {"key": d["key"], "wave": f"American_Trends_Panel_{d['wave']}",
                                "question": d["question"], "options": d["options_pew_order"],
                                "pew_dist": d["pew_dist"]}
    return items


def parse_digit(text):
    if not text: return None
    m = re.search(r"[1-5]", text.strip())
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
            if attempt == 3: return None
            time.sleep(2 ** attempt)


def empirical(samples, k=5):
    counts = np.bincount([s for s in samples if s is not None], minlength=k+1)[1:k+1].astype(float)
    if counts.sum() == 0: return np.full(k, 1.0/k)
    return counts / counts.sum()


def shannon_entropy(p):
    p = np.asarray(p)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def tv(p, q):
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


def build_standard(item, persona):
    labels = item["options"]
    ordering = ORDERINGS["ascending"]
    options = "\n".join(f"{i+1}. {labels[ordering[i]-1]}" for i in range(5))
    user = f"{item['question']}\n\nOptions:\n{options}\n\nRespond with just the number (1-5)."
    return [{"role": "system", "content": persona["descrip"]},
            {"role": "user", "content": user}], ordering


def build_ppa(item, persona, rng):
    labels = item["options"]
    ord_n = rng.choice(list(ORDERINGS))
    phr_n = rng.choice(list(PHRASINGS))
    pos_n = rng.choice(POSITIONS)
    ordering = ORDERINGS[ord_n]
    phrasing = PHRASINGS[phr_n].format(q=item["question"])
    options = "\n".join(f"{i+1}. {labels[ordering[i] - 1]}" for i in range(5))
    user = f"{phrasing}\n\nOptions:\n{options}\n\nRespond with just the number (1-5)."
    p = persona["descrip"]
    if pos_n == "system_then_user":
        msgs = [{"role": "system", "content": p}, {"role": "user", "content": user}]
    elif pos_n == "user_only":
        msgs = [{"role": "user", "content": p + "\n\n" + user}]
    else:
        msgs = [{"role": "system", "content": p},
                {"role": "user", "content": "Given this background, answer:\n\n" + user}]
    return msgs, ordering


def build_nonsemantic(item, persona, ws_n, punct_n, prefix_n):
    labels = item["options"]
    ordering = ORDERINGS["ascending"]
    options = "\n".join(f"{i+1}. {labels[ordering[i]-1]}" for i in range(5))
    base_user = f"{item['question']}\n\nOptions:\n{options}\n\nRespond with just the number (1-5)."
    user_text = dict(PUNCT_VARIANTS)[punct_n](base_user)
    user_text = dict(WHITESPACE_VARIANTS)[ws_n](user_text)
    prefix = dict(PREFIX_VARIANTS)[prefix_n]
    user_text = prefix + user_text
    return [{"role": "system", "content": persona["descrip"]},
            {"role": "user", "content": user_text}], ordering


# ---- --entropy-correlation stage ----

def _load_all_100():
    from silicon.opinionqa import load_100_items
    return load_100_items()


def _get_labels():
    labels = json.load(open(ORIG50_DIR / "item_labels.json"))
    labels.update(json.load(open(DATA / "item_labels_v2.json")))
    entropy = json.load(open(DATA / "item_entropy_v2.json"))
    per_item = {**entropy.get("per_item", {}), **entropy.get("per_item_new", {})}
    median = entropy["preserved_threshold_e6_median"]
    return labels, per_item, median


def _select_20_items(all_items, labels, per_item_ent, median):
    """Five items per (domain, entropy-bucket) cell, seeded shuffle."""
    rng = random.Random(SEED)
    cells = {}
    for it in all_items:
        dom = labels.get(it["key"], {}).get("domain", "VALUES")
        ent = per_item_ent.get(it["key"], 0)
        bucket = "HI" if ent >= median else "LO"
        cells.setdefault((dom, bucket), []).append(it)
    selected = []
    for cell_key in [("FACTS", "HI"), ("FACTS", "LO"), ("VALUES", "HI"), ("VALUES", "LO")]:
        pool = cells.get(cell_key, [])
        rng.shuffle(pool)
        selected.extend(pool[:5])
    return selected


def _run_cell(item, persona, condition, ppa_rng):
    def worker(_):
        if condition == "standard":
            msgs, ordering = build_standard(item, persona)
        else:
            msgs, ordering = build_ppa(item, persona, ppa_rng)
        text = call_text(msgs)
        d = parse_digit(text)
        canonical = displayed_to_canonical(d, ordering)
        return {"item_id": item["key"], "persona_id": persona["id"],
                "condition": condition, "raw": text, "canonical": canonical}
    with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
        responses = list(pool.map(worker, range(N_CALLS)))
    return responses


def run_entropy_correlation():
    from scipy import stats as scipy_stats

    out_dir = results_dir("mechanism/entropy_correlation")
    print("Loading items and personas...")
    all_items = _load_all_100()
    labels, per_item_ent, median = _get_labels()
    items = _select_20_items(all_items, labels, per_item_ent, median)
    personas = _select_5_personas()
    print(f"Selected {len(items)} items, {len(personas)} personas")
    print(f"Items: {[it['key'] for it in items]}")
    print(f"Personas: {[p['id'] for p in personas]}")

    cell_rows = []
    raw_file = open(out_dir / "raw_responses.jsonl", "w")
    total_calls = 0

    for cond in ["standard", "ppa"]:
        print(f"\nCondition: {cond}")
        for item in items:
            for persona in personas:
                ppa_rng = random.Random(SEED + zlib.crc32(item["key"].encode()) + persona["id"])  # stable across processes (PYTHONHASHSEED-independent)
                responses = _run_cell(item, persona, cond, ppa_rng)
                total_calls += len(responses)
                for r in responses:
                    raw_file.write(json.dumps(r, default=str) + "\n")
                canonical = [r["canonical"] for r in responses]
                n_valid = sum(1 for c in canonical if c is not None)
                if n_valid < 40:
                    print(f"  WARNING: {item['key']}/{persona['id']}/{cond} only {n_valid}/50 valid")
                p_hat = empirical(canonical)
                cell_rows.append({
                    "item_id": item["key"], "persona_id": persona["id"],
                    "condition": cond, "entropy": shannon_entropy(p_hat),
                    "n_valid": n_valid,
                    "distribution": p_hat.tolist(),
                })
            print(f"  {item['key']:25s} done ({cond})")

    raw_file.close()
    print(f"\nTotal API calls: {total_calls}")

    cell_df = pd.DataFrame(cell_rows)
    cell_df.to_csv(out_dir / "per_cell_data.csv", index=False)

    item_rows = []
    for item in items:
        dom = labels.get(item["key"], {}).get("domain", "VALUES")
        ent_val = per_item_ent.get(item["key"], 0)
        bucket = "HI" if ent_val >= median else "LO"
        H_Pew = shannon_entropy(item["pew_dist"])

        for cond in ["standard", "ppa"]:
            sub = cell_df[(cell_df["item_id"] == item["key"]) & (cell_df["condition"] == cond)]
            H_mean = sub["entropy"].mean()
            # Aggregate distribution: pool all responses, weighted by n_valid.
            all_canonical = []
            for _, row in sub.iterrows():
                dist = row["distribution"]
                if isinstance(dist, str):
                    dist = json.loads(dist)
                for i, p in enumerate(dist):
                    all_canonical.extend([i+1] * int(round(p * row["n_valid"])))
            agg_dist = empirical(all_canonical)
            TV = tv(agg_dist, item["pew_dist"])
            if cond == "standard":
                H_std = H_mean; TV_std = TV
            else:
                H_ppa = H_mean; TV_ppa = TV

        item_rows.append({
            "item_id": item["key"], "domain": dom, "entropy_bin": bucket,
            "H_standard": H_std, "H_PPA": H_ppa,
            "delta_H": H_ppa - H_std,
            "TV_standard": TV_std, "TV_PPA": TV_ppa,
            "delta_TV": TV_std - TV_ppa,  # positive = PPA improves
            "H_Pew": H_Pew,
        })

    item_df = pd.DataFrame(item_rows)
    item_df.to_csv(out_dir / "per_item_data.csv", index=False)

    print("\nPER-ITEM SUMMARY")
    print(item_df[["item_id", "domain", "entropy_bin", "H_standard", "H_PPA",
                     "delta_H", "TV_standard", "TV_PPA", "delta_TV", "H_Pew"]].round(4).to_string(index=False))

    import statsmodels.api as sm

    X = item_df[["delta_H", "H_Pew"]]
    X = sm.add_constant(X)
    y = item_df["delta_TV"]
    model = sm.OLS(y, X).fit()

    print("\nOLS REGRESSION: delta_TV ~ delta_H + H_Pew")
    print(model.summary())

    with open(out_dir / "regression_results.txt", "w") as f:
        f.write(str(model.summary()))

    coeff_dH = model.params["delta_H"]
    ci_dH = model.conf_int().loc["delta_H"]
    p_dH = model.pvalues["delta_H"]

    r, p_corr = scipy_stats.pearsonr(item_df["delta_H"], item_df["delta_TV"])
    print(f"\nPearson r(delta_H, delta_TV) = {r:.4f}, p = {p_corr:.4g}")

    print("\nDECISION")
    if coeff_dH > 0 and p_dH < 0.05 and ci_dH[0] > 0:
        decision = f"KEEP_OPINIONQA: coefficient = {coeff_dH:.4f}, p = {p_dH:.4g}, 95% CI [{ci_dH[0]:.4f}, {ci_dH[1]:.4f}]"
    else:
        decision = f"DROP_OPINIONQA: coefficient = {coeff_dH:.4f}, p = {p_dH:.4g}, 95% CI [{ci_dH[0]:.4f}, {ci_dH[1]:.4f}]"
    print(f"-> {decision}")

    with open(out_dir / "decision.txt", "w") as f:
        f.write(decision + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(item_df["delta_H"], item_df["delta_TV"], c="#3366cc", s=50, zorder=3)
    for _, row in item_df.iterrows():
        ax.annotate(row["item_id"][:12], (row["delta_H"], row["delta_TV"]),
                     fontsize=6, alpha=0.7)
    x_range = np.linspace(item_df["delta_H"].min() - 0.05, item_df["delta_H"].max() + 0.05, 100)
    y_pred = model.params["const"] + model.params["delta_H"] * x_range + model.params["H_Pew"] * item_df["H_Pew"].mean()
    ax.plot(x_range, y_pred, "r--", lw=1.5, label=f"OLS fit (beta={coeff_dH:.3f}, p={p_dH:.3f})")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.axvline(0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("dH (PPA entropy - Standard entropy)")
    ax.set_ylabel("dTV (Standard TV - PPA TV; positive = PPA helps)")
    ax.set_title("Entropy-correlation test: does PPA's entropy increase predict TV improvement?")
    ax.legend(loc="best", fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_dir / "scatter_plot.pdf", dpi=200)
    fig.savefig(out_dir / "scatter_plot.png", dpi=200)
    plt.close()
    print("\nWrote scatter_plot.pdf")
    print("DONE.")


# ---- default stage: mechanism tests 1 and 3 ----

def _entropy_run_path(name):
    """Locate an entropy-correlation output: fresh run first, then PROJECT."""
    local = results_dir("mechanism/entropy_correlation") / name
    if local.exists():
        return local
    return PROJECT / "entropy_correlation_results" / name


def _run_condition(items, personas, cond_name):
    """50 calls per (item, persona) cell under one control condition."""
    ws_names = [n for n, _ in WHITESPACE_VARIANTS]
    punct_names = [n for n, _ in PUNCT_VARIANTS]
    prefix_names = [n for n, _ in PREFIX_VARIANTS]
    rng = random.Random(SEED + zlib.crc32(cond_name.encode()) % 100000)  # stable across processes (PYTHONHASHSEED-independent)

    rows = []
    for item in items:
        for persona in personas:
            def worker(_):
                if cond_name == "T15":
                    msgs, ordering = build_standard(item, persona)
                    text = call_text(msgs, temperature=1.5)
                elif cond_name == "N1_whitespace":
                    ws = rng.choice(ws_names)
                    msgs, ordering = build_nonsemantic(item, persona, ws, "plain", "")
                    text = call_text(msgs)
                elif cond_name == "N2_punctuation":
                    punct = rng.choice(punct_names)
                    msgs, ordering = build_nonsemantic(item, persona, "none", punct, "")
                    text = call_text(msgs)
                elif cond_name == "N3_prefix":
                    prefix = rng.choice(prefix_names)
                    msgs, ordering = build_nonsemantic(item, persona, "none", "plain", prefix)
                    text = call_text(msgs)
                elif cond_name == "N4_all":
                    ws = rng.choice(ws_names)
                    punct = rng.choice(punct_names)
                    prefix = rng.choice(prefix_names)
                    msgs, ordering = build_nonsemantic(item, persona, ws, punct, prefix)
                    text = call_text(msgs)
                else:
                    msgs, ordering = build_standard(item, persona)
                    text = call_text(msgs)
                return parse_digit(text)

            with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
                results = list(pool.map(worker, range(N_CALLS)))

            p_hat = empirical(results)
            rows.append({"condition": cond_name, "item_id": item["key"],
                          "persona_id": persona["id"],
                          "entropy": shannon_entropy(p_hat),
                          "n_valid": sum(1 for r in results if r is not None),
                          "distribution": p_hat.tolist()})
        print(f"  {cond_name:15s} {item['key']:25s} done"); sys.stdout.flush()
    return rows


def run_controls():
    t1_dir = results_dir("mechanism/test1_nonsemantic_entropy")
    t3_dir = results_dir("mechanism/test3_temperature_analogue")

    entropy_items = pd.read_csv(_entropy_run_path("per_item_data.csv"))
    item_keys = entropy_items["item_id"].tolist()

    all_items = _curated_items()
    items = [all_items[k] for k in item_keys if k in all_items]
    personas = _select_5_personas()
    print(f"Items: {len(items)}, Personas: {len(personas)}")
    print("Conditions: N1, N2, N3, N4, T15")
    print(f"Total calls: 5 x {len(items)} x {len(personas)} x {N_CALLS} = {5*len(items)*len(personas)*N_CALLS}\n")

    all_rows = []
    cache_path = t1_dir / "all_conditions_per_cell.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        done_conds = set(cached["condition"].unique())
        all_rows = cached.to_dict("records")
        print(f"Resuming: {done_conds} already done")
    else:
        done_conds = set()

    for cond in ["N1_whitespace", "N2_punctuation", "N3_prefix", "N4_all", "T15"]:
        if cond in done_conds:
            print(f"  SKIP {cond} (cached)"); sys.stdout.flush(); continue
        print(f"\nCondition: {cond}"); sys.stdout.flush()
        rows = _run_condition(items, personas, cond)
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(cache_path, index=False)
        print(f"  Saved {len(all_rows)} rows to cache"); sys.stdout.flush()

    df = pd.DataFrame(all_rows)

    # Combine with the standard/PPA cells from the entropy-correlation run.
    existing = pd.read_csv(_entropy_run_path("per_cell_data.csv"))
    combined = pd.concat([existing, df], ignore_index=True)
    combined.to_csv(t1_dir / "all_conditions_per_cell.csv", index=False)

    print("\nTEST 1: NON-SEMANTIC ENTROPY")

    item_means = combined.groupby(["condition", "item_id"])["entropy"].mean().reset_index()
    item_means = item_means.rename(columns={"entropy": "mean_entropy"})
    std_ent = item_means[item_means["condition"] == "standard"].set_index("item_id")["mean_entropy"]

    nonsem_summary = pd.read_csv(
        PROJECT / "paper_extensions/robustness/exp_e2c_nonsemantic/results/nonsem_summary.csv")
    cond_map = {"standard": "N0_baseline", "N1_whitespace": "N1_whitespace",
                 "N2_punctuation": "N2_punctuation", "N3_prefix": "N3_prefix",
                 "N4_all": "N4_all_nonsem"}

    summary_rows = []
    for cond in ["standard", "ppa", "N1_whitespace", "N2_punctuation", "N3_prefix", "N4_all", "T15"]:
        sub = item_means[item_means["condition"] == cond].set_index("item_id")["mean_entropy"]
        common = std_ent.index.intersection(sub.index)
        delta_H = (sub[common] - std_ent[common]).mean()
        mean_H = sub[common].mean()

        # delta-TV per condition: non-semantic conditions from the 13-item
        # non-semantic summary; ppa and T15 from the earlier ablation and
        # temperature-sweep runs.
        if cond in cond_map:
            ns_row = nonsem_summary[nonsem_summary["condition"] == cond_map[cond]]
            delta_TV = float(ns_row["delta_vs_baseline"].iloc[0]) if len(ns_row) and cond != "standard" else 0.0
        elif cond == "ppa":
            delta_TV = -0.1064  # from the earlier PPA ablation run
        elif cond == "T15":
            delta_TV = -0.0164  # from the temperature sweep
        else:
            delta_TV = 0

        ratio = abs(delta_TV) / max(abs(delta_H), 1e-6) if delta_H != 0 else None
        summary_rows.append({"condition": cond, "mean_H": round(mean_H, 4),
                              "delta_H": round(delta_H, 4),
                              "delta_TV": round(delta_TV, 4) if delta_TV else None,
                              "abs_deltaTV_over_deltaH": round(ratio, 4) if ratio else None})

    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))
    summary.to_csv(t1_dir / "summary.csv", index=False)

    print("\nTEST 3: TEMPERATURE ANALOGUE")

    t15_ent = item_means[item_means["condition"] == "T15"].set_index("item_id")["mean_entropy"]
    common = std_ent.index.intersection(t15_ent.index)

    t3_rows = []
    for iid in common:
        ec_row = entropy_items[entropy_items["item_id"] == iid]
        if ec_row.empty: continue
        ec = ec_row.iloc[0]
        t3_rows.append({
            "item_id": iid,
            "deltaH_PPA": ec["delta_H"],
            "deltaTV_PPA": ec["delta_TV"],
            "deltaH_T15": float(t15_ent[iid] - std_ent[iid]),
            "H_std": float(std_ent[iid]),
            "H_T15": float(t15_ent[iid]),
        })
    t3_df = pd.DataFrame(t3_rows)

    # delta-TV for T=1.5 comes from the 13-item temperature sweep, which is
    # keyed by short item id; map long keys through ORIGINAL_13.
    temp_sweep = pd.read_csv(
        PROJECT / "paper_extensions/robustness/exp_r7_fpa_baselines/results/temp_sweep.csv")
    t10 = temp_sweep[temp_sweep["temperature"] == 1.0].set_index("item_id")["tv"]
    t15_tv = temp_sweep[temp_sweep["temperature"] == 1.5].set_index("item_id")["tv"]
    id_to_key = {it["id"]: it["key"] for it in ORIGINAL_13}
    for iid in t3_df["item_id"]:
        for short_id, key in id_to_key.items():
            if key == iid and short_id in t10.index and short_id in t15_tv.index:
                idx = t3_df[t3_df["item_id"] == iid].index[0]
                t3_df.loc[idx, "deltaTV_T15"] = float(t10[short_id] - t15_tv[short_id])
                break

    t3_df.to_csv(t3_dir / "per_item.csv", index=False)
    valid_t3 = t3_df.dropna(subset=["deltaTV_T15", "deltaH_T15"])
    print(f"\nItems with both dH and dTV for T=1.5: {len(valid_t3)}")
    if len(valid_t3) >= 5:
        import statsmodels.api as sm
        X = sm.add_constant(valid_t3["deltaH_T15"])
        model = sm.OLS(valid_t3["deltaTV_T15"], X).fit()
        print(f"  T=1.5 regression: beta_dH = {model.params.iloc[1]:.4f}, p = {model.pvalues.iloc[1]:.4g}")
        print("  PPA regression:   beta_dH = 0.2664, p = 0.0003")
        print(f"\n  Mean dH (PPA): {t3_df['deltaH_PPA'].mean():.4f}")
        print(f"  Mean dH (T=1.5): {t3_df['deltaH_T15'].mean():.4f}")
        with open(t3_dir / "summary.md", "w") as f:
            f.write("# Test 3: Temperature analogue\n\n")
            f.write(f"Items with both metrics: {len(valid_t3)}\n")
            f.write(f"T=1.5 beta_dH: {model.params.iloc[1]:.4f} (p={model.pvalues.iloc[1]:.4g})\n")
            f.write("PPA beta_dH: 0.2664 (p=0.0003)\n")
            f.write(f"Mean dH PPA: {t3_df['deltaH_PPA'].mean():.4f}\n")
            f.write(f"Mean dH T=1.5: {t3_df['deltaH_T15'].mean():.4f}\n")
    else:
        print("  Too few items for T=1.5 regression (temp sweep was 13-item only)")

    print("\nTEST 1 INTERPRETATION")
    ppa_row = summary[summary["condition"] == "ppa"].iloc[0]
    n4_row = summary[summary["condition"] == "N4_all"].iloc[0]
    print(f"PPA:  dH={ppa_row['delta_H']:.4f}, dTV={ppa_row['delta_TV']:.4f}")
    print(f"N4:   dH={n4_row['delta_H']:.4f}, dTV={n4_row['delta_TV']:.4f}")
    if abs(n4_row["delta_H"]) > 0 and abs(ppa_row["delta_H"]) > 0:
        h_ratio = n4_row["delta_H"] / ppa_row["delta_H"]
        tv_ratio = n4_row["delta_TV"] / ppa_row["delta_TV"] if ppa_row["delta_TV"] != 0 else 0
        print(f"  N4 dH is {h_ratio*100:.1f}% of PPA's dH")
        print(f"  N4 dTV is {tv_ratio*100:.1f}% of PPA's dTV")
        if h_ratio > 0.5 and tv_ratio < 0.3:
            verdict = "Entropy mechanism not sufficient: N4 raises entropy comparably but TV barely improves."
        elif h_ratio < 0.3:
            verdict = "Entropy mechanism supported: N4 raises entropy much less than PPA."
        else:
            verdict = "Inconclusive"
        print(f"  VERDICT: {verdict}")
        with open(t1_dir / "interpretation.md", "w") as f:
            f.write(f"# Test 1 Interpretation\n\n{verdict}\n\n")
            f.write(f"PPA dH={ppa_row['delta_H']:.4f}, dTV={ppa_row['delta_TV']:.4f}\n")
            f.write(f"N4 dH={n4_row['delta_H']:.4f}, dTV={n4_row['delta_TV']:.4f}\n")
            f.write(f"N4 captures {h_ratio*100:.1f}% of PPA's dH but {tv_ratio*100:.1f}% of PPA's dTV\n")

    print("\nALL DONE.")


def main(args):
    global MODEL
    if getattr(args, "model", None):
        MODEL = args.model
    _ensure_client()
    if getattr(args, "entropy_correlation", False):
        run_entropy_correlation()
    else:
        run_controls()
