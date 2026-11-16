"""Command line entry point for calibration pairs and schedule optimization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ConfigError, load_config


def _config(args: argparse.Namespace):
    return load_config(
        args.config,
        task=args.task,
        budget=getattr(args, "budget", 4),
        run_root=args.run_root,
        device=args.device,
        micro_batch=getattr(args, "micro_batch", None),
    )


def _samples(config):
    from .data import build_dataset

    dataset = build_dataset(config.data)
    return [dataset.load_sample(record) for record in dataset.records], dataset.manifest()


def cmd_config(args: argparse.Namespace) -> None:
    import yaml

    config = _config(args)
    print(yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True))


def cmd_pairs(args: argparse.Namespace) -> None:
    from .data import build_pair_set
    from .policies import build_policy

    config = _config(args)
    policy = build_policy(config.policy, config.data)
    samples, manifest = _samples(config)
    result = build_pair_set(
        policy,
        samples,
        config.pairs_dir,
        data_fingerprint=config.data_fingerprint,
        dataset_manifest=manifest,
        overwrite=args.overwrite,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    print(f"Wrote {result['num_pairs']} calibration pairs to {config.pairs_dir}")


def cmd_solve(args: argparse.Namespace) -> None:
    from .config.base import fingerprint
    from .data import build_pair_set, load_pairs
    from .deploy import knot_axis_from_metadata, schedule_document
    from .objective import build_objective
    from .policies import ContextSource, LatentBank, build_policy
    from .run.artifacts import write_json
    from .run.bank import EndpointBank
    from .search import solve

    config = _config(args)
    samples, dataset_manifest = _samples(config)
    if config.solver.context_cache < len(samples):
        raise ValueError("context_cache must hold every calibration observation")
    policy = build_policy(config.policy, config.data)
    contexts = ContextSource(policy, samples, cache_size=config.solver.context_cache)
    manifest = build_pair_set(
        policy,
        samples,
        config.pairs_dir,
        data_fingerprint=config.data_fingerprint,
        dataset_manifest=dataset_manifest,
        get_context=contexts.get,
    )
    pairs = load_pairs(config.pairs_dir)
    latents = LatentBank(
        policy, noise_offset=config.method.noise_offset, n_latents=config.method.n_latents
    )
    objective = build_objective(
        policy,
        pairs,
        contexts,
        latents,
        config.method,
        budget=config.budget,
        micro_batch=config.solver.micro_batch,
        score_horizon=int(getattr(policy, "score_horizon", 0) or 0),
        score_slice=getattr(policy, "score_slice", None),
        scratch=config.endpoints_dir,
    )
    identity = fingerprint(
        {
            "schema": "gys-rollout",
            "pairs": manifest["sha256"],
            "config": config.to_dict(),
            "policy": policy.metadata(),
        }
    )
    bank = EndpointBank(config.endpoints_dir, identity=identity)
    solution = solve(
        objective,
        bank,
        budget=config.budget,
        method=config.method,
        solver=config.solver,
        logger=sys.stderr,
    )
    write_json(
        config.schedule_path,
        schedule_document(
            solution.nodes,
            knot_axis=knot_axis_from_metadata(policy.metadata()),
            fingerprint=config.fingerprint,
            provenance={
                "task": config.task,
                "budget": config.budget,
                "order": solution.order,
                "observation_weight": solution.observation_weight,
                "loss": solution.loss,
            },
        ),
    )
    print(
        json.dumps(
            {
                "schedule": str(config.schedule_path),
                "nodes": solution.nodes,
                "loss": solution.loss,
                "order": solution.order,
                "evaluations": solution.evaluations,
            }
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gys", description="Ground Your Steps")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("config", cmd_config), ("pairs", cmd_pairs), ("solve", cmd_solve)):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--task", help="task name used to resolve {task} in the data path")
        command.add_argument("--run-root", type=Path)
        command.add_argument("--device")
        if name != "pairs":
            command.add_argument("--budget", type=int, required=True)
        if name == "pairs":
            command.add_argument("--overwrite", action="store_true")
        if name == "solve":
            command.add_argument("--micro-batch", type=int)
        command.set_defaults(handler=handler)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.handler(args)
    except (ConfigError, FileNotFoundError, ValueError, FloatingPointError) as exc:
        build_parser().exit(2, f"gys: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
