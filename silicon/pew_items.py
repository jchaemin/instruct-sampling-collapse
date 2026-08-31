"""The 13 curated Pew ATP items (across 6 waves) with ground-truth loaders.

All items are 5-point ordinal scales presented to the LLM as a Likert-style
"1 (most negative) to 5 (most positive)" battery. Pew's internal option order
differs across surveys (e.g. "Neither" sits at position 4 or 5 in some waves),
so each item carries an explicit remap to a consistent Likert structure:
position 1 = most negative / least frequent, position 3 = neutral,
position 5 = most positive / most frequent. "P(neutral)" is therefore always
position 3 across items.

10 political items + 3 non-political items; the non-political items act as a
politicization control (if Argyle's pathology were specific to politically
coded topics, they should show smaller describe-vs-Argyle gaps).

Requires the OpinionQA waves at data/human_resp/ (not redistributed; see the
README's "External data" section).
"""
import os

import numpy as np
import pandas as pd

from silicon import DATA

_BASE = str(DATA / "human_resp")

ITEMS = [
    # W92 Society* items: Pew order is good -> bad with neutral in the middle.
    {
        "id": "TRANS", "key": "SOCIETY_TRANS_W92", "wave": "American_Trends_Panel_W92",
        "topic": "political",
        "question": ("Do you think greater social acceptance of people who are "
                     "transgender (people who identify as a gender that is "
                     "different from the sex they were assigned at birth) is "
                     "generally good or bad for our society?"),
        "likert_to_pew": ["Very bad for society", "Somewhat bad for society",
                            "Neither good nor bad for society",
                            "Somewhat good for society", "Very good for society"],
        "likert_labels": ["Very bad for society", "Somewhat bad for society",
                            "Neither good nor bad for society",
                            "Somewhat good for society", "Very good for society"],
    },
    {
        "id": "RELG", "key": "SOCIETY_RELG_W92", "wave": "American_Trends_Panel_W92",
        "topic": "political",
        "question": ("Do you think a decline in the share of Americans belonging "
                     "to an organized religion is generally good or bad for "
                     "our society?"),
        "likert_to_pew": ["Very bad for society", "Somewhat bad for society",
                            "Neither good nor bad for society",
                            "Somewhat good for society", "Very good for society"],
        "likert_labels": ["Very bad for society", "Somewhat bad for society",
                            "Neither good nor bad for society",
                            "Somewhat good for society", "Very good for society"],
    },
    {
        "id": "GUNS", "key": "SOCIETY_GUNS_W92", "wave": "American_Trends_Panel_W92",
        "topic": "political",
        "question": ("Do you think an increase in the number of guns in the "
                     "U.S. is generally good or bad for our society?"),
        "likert_to_pew": ["Very bad for society", "Somewhat bad for society",
                            "Neither good nor bad for society",
                            "Somewhat good for society", "Very good for society"],
        "likert_labels": ["Very bad for society", "Somewhat bad for society",
                            "Neither good nor bad for society",
                            "Somewhat good for society", "Very good for society"],
    },
    {
        "id": "SSM", "key": "SOCIETY_SSM_W92", "wave": "American_Trends_Panel_W92",
        "topic": "political",
        "question": ("Do you think same-sex marriages being legal in the U.S. "
                     "is generally good or bad for our society?"),
        "likert_to_pew": ["Very bad for society", "Somewhat bad for society",
                            "Neither good nor bad for society",
                            "Somewhat good for society", "Very good for society"],
        "likert_labels": ["Very bad for society", "Somewhat bad for society",
                            "Neither good nor bad for society",
                            "Somewhat good for society", "Very good for society"],
    },
    {
        "id": "IMMIG", "key": "LEGALIMMIGAMT_W92", "wave": "American_Trends_Panel_W92",
        "topic": "political",
        "question": ("Do you think the number of legal immigrants the U.S. "
                     "admits should..."),
        # Pew: [Increase a lot, Increase a little, Stay, Decrease a little, Decrease a lot].
        # Likert direction: 1 = decrease a lot, 5 = increase a lot.
        "likert_to_pew": ["Decrease a lot", "Decrease a little",
                            "Stay about the same", "Increase a little",
                            "Increase a lot"],
        "likert_labels": ["Decrease a lot", "Decrease a little",
                            "Stay about the same", "Increase a little",
                            "Increase a lot"],
    },
    # W41: Pew order is good -> bad with "Neither" at the end (position 4).
    {
        "id": "MAJMIN", "key": "ETHNCMAJMOD_W41", "wave": "American_Trends_Panel_W41",
        "topic": "political",
        "question": ("According to the U.S. Census Bureau, by the year 2050, "
                     "a majority of the population will be made up of Black, "
                     "Asian, Hispanic and other racial minority groups. In "
                     "terms of its impact on the country, do you think this "
                     "is..."),
        "likert_to_pew": ["A very bad thing", "A somewhat bad thing",
                            "Neither a good nor bad thing",
                            "A somewhat good thing", "A very good thing"],
        "likert_labels": ["A very bad thing for the country",
                            "A somewhat bad thing for the country",
                            "Neither a good nor bad thing for the country",
                            "A somewhat good thing for the country",
                            "A very good thing for the country"],
    },
    {
        "id": "INTRMAR", "key": "INTRMAR_W41", "wave": "American_Trends_Panel_W41",
        "topic": "political",
        "question": ("According to the U.S. Census Bureau, more people of "
                     "different races are marrying each other these days "
                     "than in the past. In terms of its impact on society, "
                     "do you think this is..."),
        "likert_to_pew": ["A very bad thing", "A somewhat bad thing",
                            "Neither a good nor bad thing",
                            "A somewhat good thing", "A very good thing"],
        "likert_labels": ["A very bad thing for society",
                            "A somewhat bad thing for society",
                            "Neither a good nor bad thing for society",
                            "A somewhat good thing for society",
                            "A very good thing for society"],
    },
    # W43: "helps a lot / hurts a lot / Neither" with "Neither" at position 4.
    {
        "id": "RACE_BLACK", "key": "RACESURV5b_W43", "wave": "American_Trends_Panel_W43",
        "topic": "political",
        "question": ("Overall, how does being Black affect people's ability "
                     "to get ahead in our country these days? Does it help "
                     "or hurt?"),
        "likert_to_pew": ["Hurts a lot", "Hurts a little",
                            "Neither helps nor hurts",
                            "Helps a little", "Helps a lot"],
        "likert_labels": ["Hurts a lot", "Hurts a little",
                            "Neither helps nor hurts",
                            "Helps a little", "Helps a lot"],
    },
    # W54: "helping / hurting / Neither" with "Neither" at position 4.
    {
        "id": "ECON_POOR", "key": "ECON5_d_W54", "wave": "American_Trends_Panel_W54",
        "topic": "political",
        "question": ("Do you think the country's current economic conditions "
                     "are helping or hurting people who are poor?"),
        "likert_to_pew": ["Hurting a lot", "Hurting a little",
                            "Neither helping nor hurting",
                            "Helping a little", "Helping a lot"],
        "likert_labels": ["Hurting a lot", "Hurting a little",
                            "Neither helping nor hurting",
                            "Helping a little", "Helping a lot"],
    },
    {
        "id": "ECON_RICH", "key": "ECON5_b_W54", "wave": "American_Trends_Panel_W54",
        "topic": "political",
        "question": ("Do you think the country's current economic conditions "
                     "are helping or hurting people who are wealthy?"),
        "likert_to_pew": ["Hurting a lot", "Hurting a little",
                            "Neither helping nor hurting",
                            "Helping a little", "Helping a lot"],
        "likert_labels": ["Hurting a lot", "Hurting a little",
                            "Neither helping nor hurting",
                            "Helping a little", "Helping a lot"],
    },
    # Non-political items.
    {
        "id": "EAT_HEALTHY", "key": "EAT1_W34", "wave": "American_Trends_Panel_W34",
        "topic": "non_political",
        "question": ("How often do you choose foods to eat because they are "
                     "healthy and nutritious?"),
        # Pew: [All of the time, More than half, About half, Less than half, Never].
        # Likert position 1 = Never (low frequency), 5 = All of the time.
        "likert_to_pew": ["Never", "Less than half of the time",
                            "About half of the time",
                            "More than half of the time", "All of the time"],
        "likert_labels": ["Never", "Less than half of the time",
                            "About half of the time",
                            "More than half of the time", "All of the time"],
    },
    {
        "id": "EAT_CONVENIENT", "key": "EAT2_W34", "wave": "American_Trends_Panel_W34",
        "topic": "non_political",
        "question": ("How often do you choose foods to eat because they are "
                     "easy and most convenient?"),
        "likert_to_pew": ["Never", "Less than half of the time",
                            "About half of the time",
                            "More than half of the time", "All of the time"],
        "likert_labels": ["Never", "Less than half of the time",
                            "About half of the time",
                            "More than half of the time", "All of the time"],
    },
    {
        "id": "PRIV_FREQ", "key": "PP1_W49", "wave": "American_Trends_Panel_W49",
        "topic": "non_political",
        "question": ("How often are you asked to agree to the terms and "
                     "conditions of a company's privacy policy?"),
        # Pew: [Almost daily, About once a week, About once a month, Less frequently, Never].
        "likert_to_pew": ["Never", "Less frequently",
                            "About once a month",
                            "About once a week", "Almost daily"],
        "likert_labels": ["Never", "Less frequently than monthly",
                            "About once a month",
                            "About once a week", "Almost daily"],
    },
]

_RESP_CACHE = {}


def _load_responses(wave_dir):
    if wave_dir not in _RESP_CACHE:
        _RESP_CACHE[wave_dir] = pd.read_csv(
            os.path.join(_BASE, wave_dir, "responses.csv"), low_memory=False)
    return _RESP_CACHE[wave_dir]


def pew_distribution(item_id, subgroup=None):
    """Pew empirical distribution in Likert order (index 0..4 = Likert 1..5).
    Returns a length-5 numpy array, renormalized after excluding Refused/NaN.
    """
    item = next(x for x in ITEMS if x["id"] == item_id)
    df = _load_responses(item["wave"])
    if subgroup is not None:
        for k, v in subgroup.items():
            df = df[df[k] == v]
    counts = df[item["key"]].value_counts()
    p = np.zeros(5, dtype=float)
    for likert_pos in range(5):
        pew_option = item["likert_to_pew"][likert_pos]
        p[likert_pos] = counts.get(pew_option, 0)
    s = p.sum()
    if s == 0: return None
    return p / s


def pew_demographic_proportions(cols, wave_dir="American_Trends_Panel_W92"):
    """Joint demographic proportions over `cols` in a single wave."""
    df = _load_responses(wave_dir)
    sub = df[cols].copy()
    for c in cols:
        sub = sub[~sub[c].isin(["Refused", "DK/Refused/No lean", "DK/Ref"])]
        sub = sub[sub[c].notna()]
    cnt = sub.groupby(cols).size().reset_index(name="n")
    cnt["p"] = cnt["n"] / cnt["n"].sum()
    return cnt


if __name__ == "__main__":
    print(f"{len(ITEMS)} Pew items\n")
    print(f"  {'id':<14} {'topic':<14} {'mean':>5}  {'P(neut)':>8}  question")
    for it in ITEMS:
        p = pew_distribution(it["id"])
        mean_answer = float(sum((i + 1) * p[i] for i in range(5)))
        print(f"  {it['id']:<14} {it['topic']:<14} {mean_answer:5.2f}   {p[2]*100:5.1f}%   "
              f"{it['question'][:70]}")
    print("\nFull distributions (Likert 1-5; 1=most negative, 3=neutral, 5=most positive):\n")
    for it in ITEMS:
        p = pew_distribution(it["id"])
        bar = " ".join(f"{q:5.3f}" for q in p)
        print(f"  {it['id']:<14}  [{bar}]")
