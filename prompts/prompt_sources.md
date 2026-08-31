# Prompt family -> source mapping

Verification map for `prompts_appendix.tex`. Line numbers refer to the
files in this archive; where a line number drifts, the named function/
constant in each row is authoritative.

| Appendix | Prompt family | Source (module :: function / lines) |
|---|---|---|
| A.1 | Demographic persona template + category mappings | `silicon/opinionqa.py:46-82` (PEW_TO_PROMPT_* maps + PERSONA_COLS), `:84-92` (`_describe_row`, template string at `:92`), `:95` (`sample_personas`) |
| A.2 | Argyle baseline prompt, 100-item expansion | `silicon/expansion.py:86-100` (`build_argyle`); baseline condition fixed at `:193-216` (`run_baseline` calls with defaults). Item texts: `data/item_list_v2.json` |
| A.3 | PPA perturbation grid (4 orderings x 3 phrasings x 3 positions) | `silicon/expansion.py:64-69` (ORDERINGS), `:70-74` (PHRASINGS), `:75` (POSITIONS), `:93-100` (position message construction), `:218-250` (per-call random draw, `run_ppa`) |
| A.4 | Describe pathway, 100-item expansion | `silicon/expansion.py:103-114` (`build_describe`) |
| A.5 | Verbalized sampling (VS) | `silicon/expansion.py:116-121` (`build_vs`) |
| A.6 | Matched-Argyle persona (Experiment 2) | `silicon/opinionqa.py:84-92` (`_describe_row`, the persona description), `:108-124` (`sample_matched_personas`, weighted respondent sampling), `:207` (fixed user prompt in the `run_matched_argyle` worker) |
| A.7 | Cross-family RNG probe, 5 conditions | `silicon/cross_family.py:215-226` (RNG_CONDITIONS) |
| A.8 | Cross-family seven-target panel, 7 prompts | `silicon/cross_family.py:228-243` (SEVEN_TARGETS) |
| A.9 | KNOWS/DOES V0 prompts, 5 tasks | `silicon/cross_family.py:245-256` (KD_TASKS `v0_prompt` fields) |
| A.10 | V7 algorithmic CoT (inverse-CDF), cross-family | `silicon/cross_family.py:259-270` (`build_v7_messages`) |
| A.11 | V8 list-of-N, cross-family | `silicon/cross_family.py:273-280` (`build_v8_messages`); 20-per-call / 10-call logic `:342-374` (`run_cell_v8`) |
| A.12 | V9 parametric spec, cross-family | `silicon/cross_family.py:283-289` (`build_v9_messages`); external sampling `:377-401` (`run_cell_v9`) |
| A.13 | Correction strategies V1-V6 (uniform digit + coin) | `silicon/corrections.py:100-103` (V1), `:106-110` (V2), `:113-117` (V3), `:120-124` (V4), `:127-134` (V5 system), `:137-149` (V6), `:162-181` (`build_messages`: V5 20-shot turns, V6 user suffix). Task parameters + ICL sequences: `silicon/correction_strategies_prompts.json:1-30` |
| A.13 (ext.) | V1-V6 on 3 non-uniform tasks (extended non-uniform correction runs) | `silicon/corrections.py:341-398` (EXTENDED_TASKS: V0 prompts, target phrases, ICL examples, V6 reasoning lines), `:404-442` (ext_V1-ext_V6 builders), `:445-457` (`ext_build` message construction) |
| A.14 | Stronger corrections V7-V9 (follow-up) | `silicon/corrections.py:530-543` (STRONGER_TASKS incl. base user prompts), `:546-564` (`variant_V7_cot_explicit_prob` systems), `:576-588` (`variant_V8_list_of_N`), `:618-628` (`variant_V9_sampler_spec`) |
| A.15 | Alignment-ladder V0 tasks + base-model few-shot format | `silicon/ladder.py:73-93` (TASKS `v0_prompt` fields), `:96-104` (BASE_SHOTS), `:106-110` (BASE_V8_SHOT), `:118-123` (`v8_user_prompt`), `:173-189` (`build_prompt`: chat-template vs. Q/A-prefix assembly) |
| A.16 | Open-ended survey-domain tasks (fixed prompts + paraphrases) | `silicon/openended.py:36-112` (TASKS), `:114-121` (NONSEMANTIC_TRANSFORMS) |
| Non-Likert k-option templates (not a lettered appendix section; see appendix intro note) | `silicon/nonlikert.py:123` (build_argyle), `:138` (build_describe) |

## Provenance notes

- The earlier verbalized-sampling and 50-item robustness experiments
  (original repository directories `robustness/exp_e9_vs` and
  `robustness/exp_e6_50items`; not shipped as code) reuse the same
  Argyle/PPA/describe/VS prompts; `silicon/expansion.py`, the canonical
  shipped source, states this in its module docstring.
- The extended correction variants (`silicon/corrections.py --extended`)
  instantiate the original correction-strategy templates (same file,
  `variant_V1_verbatim`..`variant_V6_reasoning`) for each new task; the
  ext_V1-ext_V4 builders are byte-identical to those templates, with a
  generic V5 system prompt and per-task V6 reasoning lines (all extracted
  verbatim in A.13).
- STRONGER_TASKS defines a `target_str` field per task
  (`silicon/corrections.py:535`, `:541`) that is never interpolated into any
  prompt; it is omitted from the appendix.
