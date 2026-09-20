"""policyci CLI.

  policyci battery --task aloha --n 200 --base-seed 0 -o battery.jsonl
  policyci run --battery battery.jsonl --policy-name act_v18 [--state-bias X] [--drop-norm]
               --out runs/a1 --policy-seed 0 [--render]
  policyci diff runs/a1/run_manifest.json runs/b1/run_manifest.json
               [--noise runs/a1/run_manifest.json runs/a2/run_manifest.json]
               [-o report.md] [--allow-pin-mismatch]

`run` needs the simulator and policy extras and a GPU: it is meant for the box.
`battery` and `diff` run anywhere.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_battery(args) -> int:
    from policyci.scenario import sample_seed_battery
    bat = sample_seed_battery(name=args.name, task="aloha_transfer_cube",
                              backend_id="gym_aloha.transfer_cube.v0",
                              n=args.n, base_seed=args.base_seed)
    path = bat.save(args.out)
    print(f"battery {bat.battery_hash[:16]} with {len(bat)} scenarios -> {path}")
    return 0


def _cmd_run(args) -> int:
    from policyci.scenario import Battery
    from policyci.backends.aloha import AlohaTransferCubeBackend
    from policyci.policies.act_aloha import ACTAlohaPolicy
    from policyci.runner import run_battery

    battery = Battery.load(args.battery)
    variant = {}
    if args.state_bias:
        variant["state_bias"] = args.state_bias
    if args.drop_norm:
        variant["drop_norm"] = True
    backend = AlohaTransferCubeBackend(render=args.render, max_steps=args.max_steps)
    try:
        policy = ACTAlohaPolicy(name=args.policy_name, device=args.device, variant=variant)
        manifest = run_battery(policy, backend, battery, out_dir=args.out,
                               run_id=args.run_id or args.policy_name,
                               policy_seed=args.policy_seed,
                               video_failures=args.render,
                               video_pass_every=args.video_pass_every if args.render else 0,
                               progress_every=args.progress_every,
                               shard=args.shard, num_shards=args.num_shards)
        print(f"manifest -> {manifest}")
    finally:
        backend.close()
    return 0


def _cmd_merge(args) -> int:
    from policyci.scenario import Battery
    from policyci.runner import merge_shards
    battery = Battery.load(args.battery)
    out = merge_shards(args.shards, battery, args.out)
    print(f"manifest -> {out}")
    return 0


def _cmd_diff(args) -> int:
    from policyci.regression import load_manifest, diff_runs, ComparabilityError
    from policyci.report import render_diff, VERDICT_TEXT
    a = load_manifest(args.a)
    b = load_manifest(args.b)
    noise_ref = None
    if args.noise:
        noise_ref = (load_manifest(args.noise[0]), load_manifest(args.noise[1]))
    try:
        d = diff_runs(a, b, noise_ref=noise_ref, alpha=args.alpha,
                      allow_pin_mismatch=args.allow_pin_mismatch)
    except ComparabilityError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    out = args.out or "policyci_diff.md"
    render_diff(d, a, b, out)
    if args.html:
        from policyci.browser import render_browser
        page = render_browser(d, a, b, args.battery, args.html, video_rel=args.video_rel)
        print(f"  browser -> {page}")
    print(f"{d.b_name} vs {d.a_name}: {VERDICT_TEXT.get(d.sequential.decision)}")
    print(f"  A {d.a_rate}")
    print(f"  B {d.b_rate}")
    nb = len(d.newly_broken)
    if d.noise_flips is not None:
        print(f"  newly broken {nb} (noise floor {d.noise_flips}, beyond noise {d.significant_regressions}), fixed {len(d.fixed)}")
    else:
        print(f"  newly broken {nb} (no noise floor measured), fixed {len(d.fixed)}")
    print(f"  report -> {out}")
    # CI semantics: exit 1 when B is worse
    return 1 if d.sequential.decision == "A_better" else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="policyci", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("battery", help="sample a content-addressed scenario battery")
    pb.add_argument("--name", default="aloha_transfer_cube_b1")
    pb.add_argument("--n", type=int, default=200)
    pb.add_argument("--base-seed", type=int, default=0)
    pb.add_argument("-o", "--out", default="battery.jsonl")
    pb.set_defaults(fn=_cmd_battery)

    pr = sub.add_parser("run", help="run one policy over a battery (GPU box)")
    pr.add_argument("--battery", required=True)
    pr.add_argument("--policy-name", required=True)
    pr.add_argument("--run-id", default=None)
    pr.add_argument("--out", required=True)
    pr.add_argument("--policy-seed", type=int, default=0)
    pr.add_argument("--state-bias", type=float, default=0.0)
    pr.add_argument("--drop-norm", action="store_true")
    pr.add_argument("--device", default="cuda")
    pr.add_argument("--max-steps", type=int, default=400)
    pr.add_argument("--render", action="store_true", help="save replay videos for failures")
    pr.add_argument("--video-pass-every", type=int, default=50)
    pr.add_argument("--progress-every", type=int, default=25)
    pr.add_argument("--shard", type=int, default=0, help="this worker's index")
    pr.add_argument("--num-shards", type=int, default=1, help="total workers over the battery")
    pr.set_defaults(fn=_cmd_run)

    pm = sub.add_parser("merge", help="recombine shard manifests into one run manifest")
    pm.add_argument("shards", nargs="+", help="each shard's run_manifest.json")
    pm.add_argument("--battery", required=True)
    pm.add_argument("--out", required=True)
    pm.set_defaults(fn=_cmd_merge)

    pd = sub.add_parser("diff", help="regression report between two runs")
    pd.add_argument("a")
    pd.add_argument("b")
    pd.add_argument("--noise", nargs=2, metavar=("REF_A", "REF_B"),
                    help="two runs of the SAME policy on the same battery, for the noise floor")
    pd.add_argument("--alpha", type=float, default=0.05)
    pd.add_argument("--allow-pin-mismatch", action="store_true")
    pd.add_argument("-o", "--out", default=None)
    pd.add_argument("--html", default=None, help="also write the scenario browser page here")
    pd.add_argument("--battery", default=None, help="battery file, so the page can show each seed")
    pd.add_argument("--video-rel", default="videos", help="videos dir relative to the html page")
    pd.set_defaults(fn=_cmd_diff)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
