#!/usr/bin/env python3
"""Entry point for the shipped-data analyses.

    python analyze.py knows-does    # per-item describe-error vs Argyle-error
    python analyze.py subgroups     # PPA subgroup analyses (needs unshipped raw logs)
    python analyze.py verify        # re-check every shipped headline number, PASS/FAIL

`verify` and `knows-does` run on the shipped
data/ files alone and need only numpy, pandas, and scipy. `subgroups` needs
the per-call raw JSONL from the expansion run (results/expansion/raw/ or
PROJECT_ROOT) and the OpinionQA waves under data/human_resp/.
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

from silicon import DATA, PROJECT, RESULTS, results_dir


def _expansion_results_dir():
    local = RESULTS / "expansion"
    if any(local.glob("*_baseline.json")):
        return local
    return PROJECT / "paper_extensions/expansion_to_100/results"


def knows_does(_args=None):
    """What happens on items where the model cannot describe the distribution?

    Primary (shipped data): correlation of describe error with standard-Argyle
    error across the 100 items, worst-describe-quartile comparison, and the
    2x2 at TV 0.20. Secondary (needs the per-item expansion JSONs, regenerable
    with `run.py expansion`): the 50-item join against paraphrase-stability
    data, Q1-Q3, and the joined per-item table.
    """
    from scipy.stats import pearsonr, spearmanr, wilcoxon

    e2 = pd.read_csv(DATA / "matched_argyle_per_item.csv")

    lines = []
    def report(s):
        print(s)
        lines.append(str(s))

    report("PRIMARY (100 items, matched_argyle_per_item.csv): describe vs standard-Argyle")
    m100 = e2.dropna(subset=["TV_describe", "TV_standard_argyle"])
    rho, p = spearmanr(m100["TV_describe"], m100["TV_standard_argyle"])
    report(f"  spearman(describe TV, argyle TV) = {rho:.3f} (p={p:.3g}, n={len(m100)})")
    q4 = m100.copy()
    q4["dq"] = pd.qcut(q4["TV_describe"], 4, labels=["best", "q2", "q3", "worst"])
    g4 = q4.groupby("dq", observed=True).agg(
        n=("item_key", "count"),
        mean_tv_describe=("TV_describe", "mean"),
        mean_tv_argyle=("TV_standard_argyle", "mean"))
    g4["argyle_wins"] = q4.groupby("dq", observed=True).apply(
        lambda s: int((s.TV_standard_argyle < s.TV_describe).sum()), include_groups=False)
    report(g4.to_string())
    w100 = q4[q4.dq == "worst"]
    ws = wilcoxon(w100["TV_describe"], w100["TV_standard_argyle"])
    report(f"  worst-describe quartile Wilcoxon: p={ws.pvalue:.3g}; "
           f"argyle wins {(w100.TV_standard_argyle < w100.TV_describe).sum()}/{len(w100)}")
    knows_ok = m100.TV_describe <= 0.2
    does_ok = m100.TV_standard_argyle <= 0.2
    report(f"  2x2 @0.2: K+D+ {int((knows_ok & does_ok).sum())} | K+D- {int((knows_ok & ~does_ok).sum())} | "
           f"K-D+ {int((~knows_ok & does_ok).sum())} | K-D- {int((~knows_ok & ~does_ok).sum())}")

    res = _expansion_results_dir()
    if not any(res.glob("*_baseline.json")):
        report("")
        report(f"SECONDARY skipped: no per-item expansion JSONs under {res}")
        report("(regenerate with `python run.py expansion`, or request the raw logs)")
        return

    rows = []
    for f in sorted(res.glob("*_baseline.json")):
        key = f.name.replace("_baseline.json", "")
        row = {"item_key": key}
        for pipe in ["baseline", "describe", "ppa", "vs"]:
            p_path = res / f"{key}_{pipe}.json"
            if p_path.exists():
                row[f"tv_{pipe}"] = json.load(open(p_path)).get("tv")
        rows.append(row)
    exp = pd.DataFrame(rows)

    df = exp.merge(e2[["item_key", "TV_matched_argyle"]], on="item_key", how="left")
    report(f"items joined: {len(df)}")

    report("")
    report("SECONDARY (50-item expansion subset with paraphrase-stability data)")
    report("Q2. Correlation between KNOWS error (describe TV) and DOES error (Argyle TV)")
    for col in ["tv_baseline", "TV_matched_argyle"]:
        m = df[["tv_describe", col]].dropna()
        rho, p = spearmanr(m["tv_describe"], m[col])
        r, pp = pearsonr(m["tv_describe"], m[col])
        report(f"  describe vs {col}: spearman rho={rho:.3f} (p={p:.3g}), pearson r={r:.3f} (p={pp:.3g}), n={len(m)}")

    report("")
    report("Q1. On items where describe is WORST, does sampling do better?")
    q = df.dropna(subset=["tv_describe", "tv_baseline"]).copy()
    q["describe_quartile"] = pd.qcut(q["tv_describe"], 4, labels=["best", "q2", "q3", "worst"])
    g = q.groupby("describe_quartile", observed=True).agg(
        n=("item_key", "count"),
        mean_tv_describe=("tv_describe", "mean"),
        mean_tv_argyle=("tv_baseline", "mean"),
        argyle_wins=("item_key", lambda ix: int((q.loc[ix.index, "tv_baseline"] < q.loc[ix.index, "tv_describe"]).sum())),
    )
    report(g.to_string())
    worst = q[q["describe_quartile"] == "worst"]
    w_stat = wilcoxon(worst["tv_describe"], worst["tv_baseline"])
    report(f"  worst-quartile Wilcoxon describe-vs-argyle: stat={w_stat.statistic:.1f}, p={w_stat.pvalue:.3g}")
    report(f"  overall: argyle beats describe on {(q.tv_baseline < q.tv_describe).sum()}/{len(q)} items")
    report(f"  worst-describe-quartile: argyle beats describe on "
           f"{(worst.tv_baseline < worst.tv_describe).sum()}/{len(worst)} items")

    report("")
    report("Q3. Self-diagnostic: does paraphrase INSTABILITY (observable, no ground")
    report("    truth needed) predict describe ERROR (needs ground truth)?")
    m = df.dropna(subset=["mean_pairwise_TV", "mean_TV_to_Pew"])
    for signal in ["mean_pairwise_TV", "max_pairwise_TV"]:
        rho, p = spearmanr(m[signal], m["mean_TV_to_Pew"])
        report(f"  spearman({signal}, describe error) = {rho:.3f} (p={p:.3g}, n={len(m)})")
    rho2, p2 = spearmanr(m["mean_pairwise_TV"], m["tv_describe"])
    report(f"  spearman(mean_pairwise_TV, single-call describe TV) = {rho2:.3f} (p={p2:.3g})")

    thr = m["mean_pairwise_TV"].quantile(2 / 3)
    flagged = m["mean_pairwise_TV"] > thr
    hi_err = m["mean_TV_to_Pew"] > m["mean_TV_to_Pew"].median()
    precision = (flagged & hi_err).sum() / max(flagged.sum(), 1)
    recall = (flagged & hi_err).sum() / max(hi_err.sum(), 1)
    report(f"  rule 'flag top-tertile instability': precision {precision:.2f}, recall {recall:.2f} for above-median describe error")

    report("")
    report("2x2 at TV threshold 0.20 (KNOWS ok = describe TV<=0.2; DOES ok = argyle TV<=0.2)")
    knows_ok = q.tv_describe <= 0.2
    does_ok = q.tv_baseline <= 0.2
    report(f"  KNOWS ok / DOES ok:   {int((knows_ok & does_ok).sum()):3d}")
    report(f"  KNOWS ok / DOES bad:  {int((knows_ok & ~does_ok).sum()):3d}")
    report(f"  KNOWS bad / DOES ok:  {int((~knows_ok & does_ok).sum()):3d}   <- 'can sample what it can't describe'")
    report(f"  KNOWS bad / DOES bad: {int((~knows_ok & ~does_ok).sum()):3d}")

    out = results_dir("knows_does")
    df.to_csv(out / "per_item_joined.csv", index=False)
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"\n[written] {out}/per_item_joined.csv, summary.txt")


# ---- subgroups ----

def subgroups(_args=None):
    """Is PPA-induced variation demographically meaningful?

    T1 subgroup-level TV against Pew party-subgroup ground truth,
    T2 Dem-Rep gap alignment across items, T3 chi-square association of
    persona party with answer, each for baseline vs PPA.

    Needs the per-call raw JSONL from the expansion run
    ({key}_{pipeline}_raw.jsonl under results/expansion/raw/ or PROJECT_ROOT;
    regenerate with `run.py expansion`) and the OpinionQA waves under
    data/human_resp/. Shipped outputs: data/subgroup_analysis/.
    """
    from scipy.stats import chi2_contingency, pearsonr, spearmanr, wilcoxon

    from silicon.opinionqa import sample_personas

    if not (DATA / "human_resp").exists():
        print("data/human_resp/ not found; this analysis needs the public "
              "OpinionQA waves (see the README's External data section). "
              "Shipped outputs: data/subgroup_analysis/")
        return

    raw_dir = RESULTS / "expansion" / "raw"
    if not (raw_dir.exists() and any(raw_dir.glob("*_baseline_raw.jsonl"))):
        alt = PROJECT / "paper_extensions/expansion_to_100/raw"
        if alt.exists():
            raw_dir = alt
    out = results_dir("subgroups")

    personas = sample_personas(100, seed=42)
    party_col = "F_PARTYSUM_FINAL"
    party_levels = ["Dem/Lean Dem", "Rep/Lean Rep"]

    items = {d["key"]: d for d in json.load(open(DATA / "item_list_v2.json"))["new_item_details"]}

    resp_cache = {}
    def load_wave(wave):
        if wave not in resp_cache:
            resp_cache[wave] = pd.read_csv(
                DATA / "human_resp" / f"American_Trends_Panel_{wave}" / "responses.csv",
                low_memory=False)
        return resp_cache[wave]

    def pew_subgroup_dist(item, level):
        df = load_wave(item["wave"])
        if party_col not in df.columns:
            return None
        sub = df[df[party_col] == level]
        counts = sub[item["key"]].value_counts()
        p = np.array([counts.get(opt, 0) for opt in item["options_pew_order"]], float)
        return p / p.sum() if p.sum() >= 30 else None  # need enough respondents

    def model_dist(calls, ids):
        answers = [c["canonical"] for c in calls if c["persona_id"] in ids and c.get("canonical")]
        if len(answers) < 10:
            return None, 0
        p = np.zeros(5)
        for a in answers:
            p[int(a) - 1] += 1
        return p / p.sum(), len(answers)

    def mean_likert(dist):
        return float(sum((i + 1) * dist[i] for i in range(5)))

    party_ids = {lvl: {p["id"] for p in personas if p["demo"][party_col] == lvl}
                 for lvl in party_levels}
    print("persona party counts:", {k: len(v) for k, v in party_ids.items()})

    rows, gaps, chis = [], [], []
    for key, item in items.items():
        calls = {}
        for pipe in ["baseline", "ppa"]:
            f = raw_dir / f"{key}_{pipe}_raw.jsonl"
            if not f.exists():
                continue
            calls[pipe] = [json.loads(l) for l in open(f)]
        if len(calls) < 2:
            continue

        # T1 subgroup TV + T2 gap
        pewd, modeld = {}, {"baseline": {}, "ppa": {}}
        for lvl in party_levels:
            pg = pew_subgroup_dist(item, lvl)
            pewd[lvl] = pg
            for pipe in ["baseline", "ppa"]:
                mg, n = model_dist(calls[pipe], party_ids[lvl])
                modeld[pipe][lvl] = mg
                if pg is not None and mg is not None:
                    rows.append({"item": key, "party": lvl, "pipeline": pipe,
                                 "n_calls": n,
                                 "tv_subgroup": 0.5 * np.abs(mg - pg).sum()})
        if all(pewd[l] is not None for l in party_levels):
            pew_gap = mean_likert(pewd["Dem/Lean Dem"]) - mean_likert(pewd["Rep/Lean Rep"])
            row = {"item": key, "pew_gap": pew_gap}
            for pipe in ["baseline", "ppa"]:
                if all(modeld[pipe][l] is not None for l in party_levels):
                    row[f"{pipe}_gap"] = (mean_likert(modeld[pipe]["Dem/Lean Dem"])
                                           - mean_likert(modeld[pipe]["Rep/Lean Rep"]))
            gaps.append(row)

        # T3 chi-square: persona party x answer
        for pipe in ["baseline", "ppa"]:
            tab = np.zeros((2, 5))
            for c in calls[pipe]:
                if not c.get("canonical"):
                    continue
                for gi, lvl in enumerate(party_levels):
                    if c["persona_id"] in party_ids[lvl]:
                        tab[gi, int(c["canonical"]) - 1] += 1
            keep = tab.sum(axis=0) > 0
            if keep.sum() >= 2 and tab.sum() > 0:
                chi2, p, _, _ = chi2_contingency(tab[:, keep])
                chis.append({"item": key, "pipeline": pipe, "chi2": chi2, "p": p,
                             "cramers_v": np.sqrt(chi2 / tab.sum() / 1)})

    if not rows:
        print(f"No raw call logs found under {raw_dir}; "
              "regenerate with `python run.py expansion` (shipped outputs: data/subgroup_analysis/)")
        return

    sub = pd.DataFrame(rows)
    gap = pd.DataFrame(gaps).dropna()
    chi = pd.DataFrame(chis)

    lines = []
    def report(s):
        print(s)
        lines.append(str(s))

    report("T1. Subgroup-level TV to Pew subgroup ground truth (party subgroups)")
    piv = sub.pivot_table(index=["item", "party"], columns="pipeline", values="tv_subgroup")
    piv = piv.dropna()
    report(f"  n (item x subgroup cells): {len(piv)}")
    report(f"  mean subgroup TV baseline: {piv['baseline'].mean():.4f}")
    report(f"  mean subgroup TV PPA:      {piv['ppa'].mean():.4f}  "
           f"(delta {piv['ppa'].mean() - piv['baseline'].mean():+.4f})")
    w = wilcoxon(piv["baseline"], piv["ppa"])
    report(f"  Wilcoxon paired: stat={w.statistic:.0f}, p={w.pvalue:.3g}")
    report(f"  PPA improves subgroup TV on {(piv['ppa'] < piv['baseline']).sum()}/{len(piv)} cells")

    report("")
    report("T2. Dem-Rep gap alignment with Pew (mean Likert difference per item)")
    report(f"  n items: {len(gap)}")
    for pipe in ["baseline", "ppa"]:
        r, p = pearsonr(gap["pew_gap"], gap[f"{pipe}_gap"])
        rho, sp = spearmanr(gap["pew_gap"], gap[f"{pipe}_gap"])
        sign = (np.sign(gap["pew_gap"]) == np.sign(gap[f"{pipe}_gap"])).mean()
        report(f"  {pipe:9s}: pearson r={r:.3f} (p={p:.3g}), spearman rho={rho:.3f} "
               f"(p={sp:.3g}), sign agreement {sign:.2f}")
    report(f"  mean |model gap|: baseline {gap['baseline_gap'].abs().mean():.3f}, "
           f"ppa {gap['ppa_gap'].abs().mean():.3f}, pew {gap['pew_gap'].abs().mean():.3f}")

    report("")
    report("T3. Persona-party x answer association per item (chi-square)")
    for pipe in ["baseline", "ppa"]:
        c = chi[chi.pipeline == pipe]
        report(f"  {pipe:9s}: significant (p<0.05) on {(c.p < 0.05).sum()}/{len(c)} items; "
               f"median Cramer's V {c.cramers_v.median():.3f}")

    sub.to_csv(out / "subgroup_tv.csv", index=False)
    gap.to_csv(out / "gap_alignment.csv", index=False)
    chi.to_csv(out / "chi_square.csv", index=False)
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"\n[written] {out}/")


ATTRIBUTION_BASELINES = {"ordering": "ascending", "phrasing": "direct",
                         "position": "system_then_user"}


def _check(label, computed, expected):
    ok = computed == expected
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}: {computed}" + ("" if ok else f"  (expected {expected})"))
    return ok


def verify(_args=None):
    """Re-check every shipped headline number against the data/ files.
    Needs only numpy, pandas, and scipy; runs in about a minute."""
    from scipy.stats import pearsonr, spearmanr, wilcoxon

    all_ok = True

    print("[1/11] Headline pipeline means over the 100 items "
          "(data/ppa_per_item/per_item_100.csv)")
    d = pd.read_csv(DATA / "ppa_per_item" / "per_item_100.csv")
    c = d.dropna(subset=["baseline_tv", "ppa_tv", "describe_tv", "vs_tv"])
    all_ok &= _check("items with all four pipelines", len(c), 100)
    for pipe, expected in [("baseline", 0.459), ("ppa", 0.361),
                           ("describe", 0.215), ("vs", 0.228)]:
        all_ok &= _check(f"mean TV {pipe}", round(float(c[f"{pipe}_tv"].mean()), 3), expected)

    print("\n[2/11] KNOWS/DOES per-item correlation "
          "(data/matched_argyle_per_item.csv)")
    m = pd.read_csv(DATA / "matched_argyle_per_item.csv")
    r = spearmanr(m.TV_describe, m.TV_standard_argyle)
    all_ok &= _check("spearman rho (describe, argyle)", round(float(r.statistic), 3), 0.008)
    all_ok &= _check("spearman p", round(float(r.pvalue), 3), 0.936)
    all_ok &= _check("n items", len(m), 100)
    w = m[m.TV_describe >= m.TV_describe.quantile(.75)]
    all_ok &= _check("worst-describe quartile: mean argyle TV",
                     round(float(w.TV_standard_argyle.mean()), 3), 0.494)
    all_ok &= _check("worst-describe quartile: mean describe TV",
                     round(float(w.TV_describe.mean()), 3), 0.339)
    all_ok &= _check("worst-describe quartile: Wilcoxon p",
                     round(float(wilcoxon(w.TV_standard_argyle, w.TV_describe).pvalue), 4),
                     0.0018)

    print("\n[3/11] Post-training-ladder means, both seeds "
          "(data/ladder/ladder_cells_all.csv; V0, non-point tasks)")
    lad = pd.read_csv(DATA / "ladder" / "ladder_cells_all.csv")
    expected_means = {
        13: {"llama31-base": 0.188, "llama31-instruct": 0.336,
             "olmo2-base": 0.236, "olmo2-sft": 0.321, "olmo2-dpo": 0.411,
             "olmo2-instruct": 0.442, "qwen25-base": 0.203,
             "qwen25-coder-base": 0.168, "qwen25-instruct": 0.540,
             "qwen3-8b": 0.639},
        47: {"llama31-base": 0.214, "llama31-instruct": 0.331,
             "olmo2-base": 0.244, "olmo2-sft": 0.327, "olmo2-dpo": 0.420,
             "olmo2-instruct": 0.441, "qwen25-base": 0.221,
             "qwen25-coder-base": 0.211, "qwen25-instruct": 0.539,
             "qwen3-8b": 0.640},
    }
    for seed, expected in expected_means.items():
        sel = lad[(lad.seed == seed) & (lad.condition == "V0") & (lad.task != "point_5")]
        computed = {k: round(float(v), 3)
                    for k, v in sel.groupby("model").tv.mean().items()}
        all_ok &= _check(f"seed {seed} mean V0 TV per model", computed, expected)

    print("\n[4/11] Dem-Rep gap alignment, baseline vs PPA "
          "(data/subgroup_analysis/gap_alignment.csv)")
    g = pd.read_csv(DATA / "subgroup_analysis" / "gap_alignment.csv")
    for col, exp_r, exp_sign in [("baseline_gap", 0.573, 0.64), ("ppa_gap", 0.678, 0.76)]:
        all_ok &= _check(f"{col}: pearson r",
                         round(float(pearsonr(g.pew_gap, g[col]).statistic), 3), exp_r)
        all_ok &= _check(f"{col}: sign agreement",
                         round(float((np.sign(g.pew_gap) == np.sign(g[col])).mean()), 2),
                         exp_sign)

    print("\n[5/11] Open-model applied replication "
          "(data/openmodel/openmodel_main_summary.csv)")
    om = pd.read_csv(DATA / "openmodel" / "openmodel_main_summary.csv")
    expected_om = {
        "llama31-instruct": (0.436, 0.276, 0.293, 96, 1.25e-08),
        "qwen25-instruct": (0.525, 0.345, 0.241, 82, 7.57e-14),
        "qwen3-8b": (0.465, 0.304, 0.240, 100, 3.54e-13),
    }
    for model, (ea, ep, ed, en, epval) in expected_om.items():
        g = om[om.model == model]
        gd = g.dropna(subset=["tv_describe"])
        all_ok &= _check(f"{model}: mean TV argyle",
                         round(float(g.tv_argyle.mean()), 3), ea)
        all_ok &= _check(f"{model}: mean TV ppa",
                         round(float(g.tv_ppa.mean()), 3), ep)
        all_ok &= _check(f"{model}: mean TV describe (parsed items)",
                         round(float(gd.tv_describe.mean()), 3), ed)
        all_ok &= _check(f"{model}: parsed describe items", len(gd), en)
        p = float(wilcoxon(gd.tv_argyle, gd.tv_describe).pvalue)
        all_ok &= _check(f"{model}: Wilcoxon argyle vs describe p",
                         float(f"{p:.3g}"), epval)

    print("\n[6/11] Matched Argyle vs matched+PPA "
          "(data/openmodel/openmodel_matched_summary.csv)")
    mt = pd.read_csv(DATA / "openmodel" / "openmodel_matched_summary.csv")
    expected_mt = {
        "llama31-instruct": (0.426, 0.285, 1.37e-14),
        "qwen25-instruct": (0.543, 0.346, 2.69e-15),
        "qwen3-8b": (0.491, 0.315, 2.11e-14),
        "gpt-4o": (0.482, 0.394, 2.14e-07),
    }
    for model, (em, emp, epval) in expected_mt.items():
        g = mt[mt.model == model]
        all_ok &= _check(f"{model}: mean TV matched",
                         round(float(g.tv_matched.mean()), 3), em)
        all_ok &= _check(f"{model}: mean TV matched+PPA",
                         round(float(g.tv_matched_ppa.mean()), 3), emp)
        all_ok &= _check(f"{model}: n items", len(g), 100)
        p = float(wilcoxon(g.tv_matched, g.tv_matched_ppa).pvalue)
        all_ok &= _check(f"{model}: Wilcoxon p", float(f"{p:.3g}"), epval)

    print("\n[7/11] Non-Likert benchmark "
          "(data/openmodel/openmodel_nonlikert_summary.csv)")
    nl = pd.read_csv(DATA / "openmodel" / "openmodel_nonlikert_summary.csv")
    expected_nl = {
        "llama31-instruct": (0.254, 0.184, 0.0249),
        "qwen25-instruct": (0.299, 0.199, 0.0059),
        "qwen3-8b": (0.362, 0.202, 0.0010),
    }
    for model, (ea, ed, epval) in expected_nl.items():
        g = nl[nl.model == model].dropna(subset=["tv_describe"])
        all_ok &= _check(f"{model}: mean TV argyle",
                         round(float(g.tv_argyle.mean()), 3), ea)
        all_ok &= _check(f"{model}: mean TV describe",
                         round(float(g.tv_describe.mean()), 3), ed)
        all_ok &= _check(f"{model}: n items", len(g), 24)
        p = float(wilcoxon(g.tv_argyle, g.tv_describe).pvalue)
        all_ok &= _check(f"{model}: Wilcoxon argyle vs describe p",
                         round(p, 4), epval)

    print("\n[8/11] Open-ended survey grid, both seeds "
          "(data/ladder/openended_survey_*.json)")
    oe_models = ["llama31-instruct", "olmo2-instruct",
                 "qwen25-instruct", "qwen3-8b"]
    expected_oe = {"": (0.983, 0.013, -0.755, 0.00012),
                   "_s47": (1.011, 0.021, -0.761, 0.000098)}
    for tag, (esem, enon, erho, ep) in expected_oe.items():
        cells = []
        for model in oe_models:
            rows = json.load(open(DATA / "ladder" /
                                  f"openended_survey_{model}{tag}.json"))
            by = {(r["task"], r["condition"]): r["entropy_nats"] for r in rows}
            for task in sorted(set(t for t, _ in by)):
                cells.append((by[(task, "fixed")],
                              by[(task, "semantic")] - by[(task, "fixed")],
                              by[(task, "nonsemantic")] - by[(task, "fixed")]))
        fixed = [c[0] for c in cells]
        sem = [c[1] for c in cells]
        non = [c[2] for c in cells]
        label = "seed 47" if tag else "seed 13"
        all_ok &= _check(f"{label}: n cells", len(cells), 20)
        all_ok &= _check(f"{label}: mean semantic entropy gain",
                         round(float(np.mean(sem)), 3), esem)
        all_ok &= _check(f"{label}: mean non-semantic entropy gain",
                         round(float(np.mean(non)), 3), enon)
        r = spearmanr(fixed, sem)
        all_ok &= _check(f"{label}: spearman rho (baseline, gain)",
                         round(float(r.statistic), 3), erho)
        all_ok &= _check(f"{label}: spearman p", float(f"{r.pvalue:.2g}"), ep)

    print("\n[9/11] Mistral-7B-v0.3 base vs instruct, both seeds "
          "(data/ladder/mistral_table3_cells.csv; Table 3 reproduction)")
    mdf = pd.read_csv(DATA / "ladder" / "mistral_table3_cells.csv")
    expected_m = {("mistral-base", 13): 0.195, ("mistral-base", 47): 0.164,
                  ("mistral-instruct", 13): 0.453, ("mistral-instruct", 47): 0.448}
    for (model, seed), exp in expected_m.items():
        sub = mdf[(mdf.model == model) & (mdf.seed == seed)]
        all_ok &= _check(f"{model} seed {seed}: mean TV over 5 targets",
                         round(float(sub.tv.mean()), 3), exp)
    for seed in (13, 47):
        b = mdf[(mdf.model == "mistral-base") & (mdf.seed == seed)].tv.mean()
        i = mdf[(mdf.model == "mistral-instruct") & (mdf.seed == seed)].tv.mean()
        all_ok &= _check(f"seed {seed}: instruct worse than base", bool(i > b), True)

    print("\n[10/11] Instruction-tuning knockout, alpha sweep "
          "(data/ladder/knockout_full.json)")
    ko = json.load(open(DATA / "ladder" / "knockout_full.json"))
    expected_ko = {
        "olmo_sft": {"0.0": 0.236, "1.0": 0.317, "1.25": 0.41},
        "qwen_instruct": {"0.0": 0.198, "1.0": 0.459, "1.25": 0.519},
    }
    for lin, exp in expected_ko.items():
        res = ko[lin]["results"]
        for al, ev in exp.items():
            got = round(res[f"alpha_{al}"]["_mean"]["tv"], 3)
            all_ok &= _check(f"{lin}: mean TV at alpha={al}", got, ev)
        seq = [res[f"alpha_{a}"]["_mean"]["tv"] for a in ["0.0", "0.25", "0.5", "0.75", "1.0", "1.25"]]
        mono = all(b >= a - 0.011 for a, b in zip(seq, seq[1:]))
        all_ok &= _check(f"{lin}: TV monotone in alpha (0.011 tol)", bool(mono), True)
        mp = [res[f"alpha_{a}"]["_mean"]["logit_maxprob"] for a in ["0.0", "1.0"]]
        all_ok &= _check(f"{lin}: T=0 max-prob rises base->tuned", bool(mp[1] > mp[0]), True)
        all_ok &= _check(f"{lin}: alpha=0 reproduces base_ref",
                         round(res["alpha_0.0"]["_mean"]["tv"], 3),
                         round(res["base_ref"]["_mean"]["tv"], 3))

    print("\n[11/11] PPA converges toward the model's own described distribution "
          "(data/openmodel/ppa_describe_convergence.csv)")
    cv = pd.read_csv(DATA / "openmodel" / "ppa_describe_convergence.csv")
    expected_cv = {"qwen25-instruct": (64, 0.556, 0.375, 9.1e-11, 0.448),
                   "qwen3-8b": (100, 0.507, 0.291, 2.2e-17, 0.36)}
    for m, (n, ea, ep, epv, eu) in expected_cv.items():
        sub = cv[cv.model == m]
        all_ok &= _check(f"{m}: n items (describe-parseable)", len(sub), n)
        all_ok &= _check(f"{m}: mean TV(argyle, describe)", round(sub.tv_argyle_describe.mean(), 3), ea)
        all_ok &= _check(f"{m}: mean TV(ppa, describe)", round(sub.tv_ppa_describe.mean(), 3), ep)
        p = wilcoxon(sub.tv_argyle_describe, sub.tv_ppa_describe).pvalue
        all_ok &= _check(f"{m}: Wilcoxon p", float(f"{p:.2g}"), epv)
        all_ok &= _check(f"{m}: mean TV(ppa, uniform)", round(sub.tv_ppa_uniform.mean(), 3), eu)
        closer = (sub.tv_ppa_describe.mean() < sub.tv_ppa_uniform.mean())
        all_ok &= _check(f"{m}: PPA closer to describe than to uniform", bool(closer), True)

    print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
    return 0 if all_ok else 1


def main():
    parser = argparse.ArgumentParser(
        prog="analyze.py",
        description="Analyses of the shipped data/ files (no API key needed).")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("knows-does",
                   help="per-item describe-error vs Argyle-error (primary: shipped data)")
    sub.add_parser("subgroups",
                   help="PPA subgroup analyses (needs unshipped raw call logs)")
    sub.add_parser("verify",
                   help="re-check every shipped headline number; prints PASS/FAIL per block")
    args = parser.parse_args()

    commands = {
                "knows-does": knows_does, "subgroups": subgroups, "verify": verify}
    rc = commands[args.command](args)
    sys.exit(rc if isinstance(rc, int) else 0)


if __name__ == "__main__":
    main()
