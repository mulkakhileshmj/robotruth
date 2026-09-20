# policyci

Policy CI for learned robot policies, built on [robotruth](..). It answers one question:

> Is policy B better, worse, or unsafe compared with policy A, and where?

It does this with deterministic scenario batteries (content-addressed scenes), a policy
runner against a simulator backend, a versioned evaluator that writes robotruth episode
records, and a regression engine that never reports a diff without a noise floor and an
interval behind it.

Design: [docs/DESIGN_POLICY_CI.md](../docs/DESIGN_POLICY_CI.md).

First cell: ALOHA transfer-cube in gym-aloha (MuJoCo) with the public
`lerobot/act_aloha_sim_transfer_cube_human` ACT checkpoint, unmodified weights.
All simulation runs on a rented GPU box; this machine orchestrates and analyses.

```
policyci battery  # sample a content-addressed scenario battery
policyci run      # run one policy over a battery (on the box)
policyci diff     # regression report between two runs, with noise floor
```
