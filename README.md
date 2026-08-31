# Instruction-Tuned Language Models Cannot Sample from Distributions They Can Describe

Code and data for the paper.

## Verify the reported numbers

```
pip install -r requirements.txt
python3 analyze.py verify
```

`verify` recomputes the applied results and the PPA/alignment analyses from
`data/` and checks each against the paper (no model calls). A few appendix
tables (RNG-exclusion, logit-gap, CoT-bridge, V8 N-sweep, decoding sweeps,
V8-structure) are not stored as data and are regenerated with `run.py`.

## Experiments

`run.py <cmd>` reruns an experiment; `analyze.py` analyzes the shipped results.

| Command | Experiment | Tables |
|---|---|---|
| `corrections [--extended\|--stronger]` | correction strategies V1-V9 on the synthetic targets | 5, 19-22 |
| `ladder --models ...` | post-training ladder (base, SFT, DPO, instruct) + Coder control | 3, 4, 10 |
| `knockout [--lineage ...]` | task-arithmetic update scaling | 11 |
| `expansion [--select-items\|--analyze]` | 100-item OpinionQA: Argyle / PPA / describe / VS + PPA analysis | 6, 7, 24, 25 |
| `opinionqa` | demographically-matched Argyle baseline | App. C |
| `open-model-replication --model ...` | 100-item comparison on open 7-8B models | 12 |
| `matched-ppa --model ... [--api]` | matched Argyle + PPA (open models and gpt-4o) | 32 |
| `nonlikert --model ...` | 24-item non-Likert benchmark | 33 |
| `openended --model ...` | open-ended generation under prompt perturbation | 34 |
| `mechanism [--entropy-correlation]` | entropy regression and mechanism controls | 8, 9, 28 |
| `cross-family [--pilot]` | replication on five other model families (OpenRouter) | 36-39 |

`analyze.py knows-does` and `analyze.py subgroups` cover Sec. 6.2 and Tables
29-30. `client.py`, `pew_items.py`, and `correction_strategies_prompts.json`
are support code.

## Data

```
data/
  aggregates/         RNG probe, seven-target panel, corrections, cross-family, entropy regression, mechanism cells
  ladder/             post-training ladder, Mistral Table 3, knockout sweep, open-ended grid
  openmodel/          open-model 100-item, matched+PPA, non-Likert, PPA-convergence
  ppa_per_item/       per-item TVs, 2x2 breakdown, bootstrap CIs
  subgroup_analysis/  gap alignment, per-item chi-square
  item_list_v2.json   100-item list (+ _labels_v2, _entropy_v2 for domain/entropy labels)
  matched_argyle_per_item.csv   matched-Argyle per-item TVs
```

## Rerunning

Requires `OPENAI_API_KEY` (`SIM_MODEL` defaults to `gpt-4o`); `cross-family`
uses `OPENROUTER_API_KEY`; the local-model commands need `torch`,
`transformers`, and a GPU.

The Pew ATP data is not redistributed. Obtain it from OpinionQA
(https://github.com/tatsu-lab/opinions_qa) and place the waves under
`data/human_resp/`; the shipped aggregates are enough for `verify`.

Prompts are in `prompts/prompts_appendix.tex` (A.1-A.16), mapped to source in
`prompts/prompt_sources.md`.
