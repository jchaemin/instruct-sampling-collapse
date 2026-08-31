#!/usr/bin/env python3
"""Entry point for the experiment runners. One subcommand per experiment
family; each dispatches to a module under silicon/ (see the module docstring
for inputs, outputs, and requirements).

    python run.py ladder --models olmo2-base,olmo2-sft --seed 13
    python run.py opinionqa --model gpt-4o
    python run.py open-model-replication --model llama31-instruct [--repair]
    python run.py matched-ppa --model qwen25-instruct | --api
    python run.py nonlikert --model qwen25-instruct | --build-items
    python run.py openended --model llama31-instruct
    python run.py knockout [--lineage olmo_sft|qwen_instruct]
    python run.py cross-family [--pilot]
    python run.py corrections [--extended | --stronger]
    python run.py mechanism [--entropy-correlation]
    python run.py expansion [--select-items | --analyze]

API-based subcommands need OPENAI_API_KEY (cross-family: OPENROUTER_API_KEY);
--model on those sets the chat model (default gpt-4o, the paper's main model). Local-model
subcommands take a checkpoint key from silicon/ladder.py MODELS and need
torch + transformers.
"""
import argparse
import importlib

LOCAL_MODEL_KEYS = ("olmo2-base, olmo2-sft, olmo2-dpo, olmo2-instruct, "
                    "llama31-base, llama31-instruct, qwen25-base, "
                    "qwen25-coder-base, qwen25-instruct, qwen3-8b")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Experiment runners for the supplement (one subcommand per family).")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ladder", help="post-training-ladder sampling panel on local checkpoints")
    p.add_argument("--models", required=True,
                   help=f"comma-separated checkpoint keys: {LOCAL_MODEL_KEYS}")
    p.add_argument("--gpus", default=None, help="CUDA_VISIBLE_DEVICES value")
    p.add_argument("--n-v0", dest="n_v0", type=int, default=200,
                   help="V0 calls per task (default 200)")
    p.add_argument("--seed", type=int, default=13,
                   help="generation seed (13 = primary run, 47 = stability rerun)")

    p = sub.add_parser("opinionqa", help="demographically-matched Argyle baseline (Appendix C)")
    p.add_argument("--model", default=None, help="chat model (default gpt-4o, the paper's main model)")

    p = sub.add_parser("open-model-replication",
                       help="Argyle/PPA/describe on local open models, 100 items")
    p.add_argument("--model", required=True, help="checkpoint key from silicon/ladder.py")
    p.add_argument("--repair", action="store_true",
                   help="describe-parse post-pass (re-parse + greedy regeneration)")

    p = sub.add_parser("matched-ppa", help="matched Argyle vs matched+PPA")
    p.add_argument("--model", default=None, help="checkpoint key (local run)")
    p.add_argument("--api", action="store_true",
                   help="run on the API model (SIM_MODEL, default gpt-4o) instead")

    p = sub.add_parser("nonlikert", help="non-Likert (2-4 option) benchmark on local open models")
    p.add_argument("--model", default=None, help="checkpoint key from silicon/ladder.py")
    p.add_argument("--build-items", action="store_true",
                   help="rebuild the 24-item non-Likert set instead of running the benchmark")

    p = sub.add_parser("openended", help="open-ended survey-domain generation under perturbation")
    p.add_argument("--model", required=True, help="checkpoint key from silicon/ladder.py")

    p = sub.add_parser("knockout", help="instruction-tuning knockout (task-arithmetic alpha sweep)")
    p.add_argument("--lineage", choices=["olmo_sft", "qwen_instruct"], default=None,
                   help="run one lineage only (default: both)")

    p = sub.add_parser("cross-family", help="cross-family replication via OpenRouter")
    p.add_argument("--pilot", action="store_true", help="30-call feasibility pilot only")

    p = sub.add_parser("corrections", help="correction strategies on synthetic sampling tasks")
    p.add_argument("--model", default=None, help="chat model (default gpt-4o, the paper's main model)")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--extended", action="store_true",
                       help="V1-V6 on the three non-uniform targets")
    group.add_argument("--stronger", action="store_true",
                       help="V7-V9 (CoT / list-of-N / sampler spec)")

    p = sub.add_parser("mechanism", help="mechanism tests (non-semantic controls, temperature)")
    p.add_argument("--model", default=None, help="chat model (default gpt-4o, the paper's main model)")
    p.add_argument("--entropy-correlation", action="store_true",
                   help="run the delta-H vs delta-TV entropy-correlation experiment instead")

    p = sub.add_parser("expansion", help="100-item expansion: selection, pipelines, analysis")
    p.add_argument("--model", default=None, help="chat model (default gpt-4o, the paper's main model)")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--select-items", action="store_true",
                       help="rebuild the 50-item expansion set")
    group.add_argument("--analyze", action="store_true",
                       help="100-item tables, 2x2, bootstrap CIs (no API key)")

    return parser


MODULES = {
    "ladder": "silicon.ladder",
    "opinionqa": "silicon.opinionqa",
    "open-model-replication": "silicon.open_model",
    "matched-ppa": "silicon.matched_ppa",
    "nonlikert": "silicon.nonlikert",
    "openended": "silicon.openended",
    "knockout": "silicon.knockout",
    "cross-family": "silicon.cross_family",
    "corrections": "silicon.corrections",
    "mechanism": "silicon.mechanism",
    "expansion": "silicon.expansion",
}


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "matched-ppa" and not args.api and not args.model:
        parser.error("matched-ppa requires --model <checkpoint key> or --api")
    if args.command == "nonlikert" and not args.build_items and not args.model:
        parser.error("nonlikert requires --model <checkpoint key> or --build-items")
    module = importlib.import_module(MODULES[args.command])
    module.main(args)


if __name__ == "__main__":
    main()
