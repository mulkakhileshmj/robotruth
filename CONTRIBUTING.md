# Contributing

Thanks for helping make robot evaluation honest.

## Ground rules

- A number without an interval is not a result. Any new metric that reports a rate must carry a confidence interval, and the report layer must refuse to print it otherwise.
- Fail closed. If a check cannot verify something that changes robot behaviour, it fails, it does not warn.
- No framework imports in adapters. Extractors and ingest read files (JSON, parquet, safetensors headers). This keeps robotruth installable next to any policy stack.
- Cite the evidence. When a threshold or design choice comes from a paper, name it in the docstring.
- Plain English docstrings.

## Setup

```bash
uv venv .venv && uv pip install -e ".[dev]" opencv-python-headless
.venv/bin/python -m pytest -q
```

## Adding a checkpoint family

Add a `_family(src, spec)` function in `src/robotruth/contract/extract.py` that fills the ExecSpec from files only, lists everything it cannot read in `spec.unverified`, and add a detection rule in `detect_family`. Add a fixture test in `tests/test_contract.py`. If you can, run `ops/remote_checkpoint_probe.sh` against a public checkpoint of that family and commit the probe (hashes and configs only, never weights) under `examples/probe/`.

## Adding a dataset format

Extend `src/robotruth/schema/lerobot_ingest.py` or add a sibling module that yields `EpisodeRecord` objects. Success labels and intervention flags are the two fields that matter most.

## Tests

Statistical code is tested by simulation (coverage of intervals, false alarm rates of sequential tests). Keep those tests; tighten tolerances only with more simulations, not fewer.
