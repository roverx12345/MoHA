# MoHA

MoHA is a model-conditioned harness for video-agent calibration. The
repository contains the implementation, unit tests, configuration template,
and a small offline demo. Experiment outputs, datasets, credentials, model
weights, and service logs are intentionally kept outside this repository.

## Layout

```text
src/moha/          calibration, runtime, evidence, and evaluation logic
tests/             offline unit and integration tests
scripts/           optional local service launchers
calibrated/        reviewed, credential-free calibration profiles
config.example.json
pyproject.toml
```

MoHA uses the separately versioned Flat runtime for media loading, model
adapters, and token accounting. Set `runtime.root` and the exact
`runtime.commit` in a copy of `config.example.json`; do not copy Flat into
this repository.

## Offline demo and tests

The demo has no network, credentials, video files, or model dependencies:

```bash
PYTHONPATH=src python -m moha demo --output /tmp/moha-demo
```

For the full offline test suite, use Python 3.11 or newer and the pinned Flat
runtime on `PYTHONPATH`:

```bash
PYTHONPATH=src:/path/to/flat/src python -m unittest discover -s tests -v
```

## Running an experiment

1. Copy `config.example.json` outside the repository.
2. Replace the placeholder media, manifest, endpoint, credential, and Flat
   runtime settings.
3. Keep the source and Flat worktrees committed and clean.
4. Run a configuration check before any model call:

```bash
PYTHONPATH=src:/path/to/flat/src python -m moha doctor \
  --config /path/to/config.json
```

Calibration and evaluation outputs must also live outside the source
worktree. The command records source/runtime commits, input hashes, provider
identities, and measured costs in the output directory.

The configuration supports planner, judge, Omni observer, optional OCR, and
optional Whisper ASR services. Credentials are referenced by environment
variable or an external credential file; inline secrets are rejected.

## Reproducibility

The implementation is deterministic about catalog IDs, split checks, endpoint
lane assignment, budget accounting, and calibration promotion. A completed
calibration can be exported with:

```bash
PYTHONPATH=src:/path/to/flat/src python -m moha export \
  --run /path/to/completed/run --output calibrated
```

Review an exported profile before committing it. It must not contain API keys,
credential paths, datasets, videos, logs, or per-sample traces.
