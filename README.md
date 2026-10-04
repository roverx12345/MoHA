# MoHA

Model-conditioned harness for video-agent calibration and evaluation.

## Prerequisites

- Python 3.11 or newer.
- A pinned Flat source tree. Set `runtime.root` and its exact `runtime.commit` in the external config.
- External model endpoints and credential references. Do not store credentials in this repository.
- A clean MoHA checkout for real model calls.

## Setup

```bash
cd /path/to/moha
PY=/path/to/python
export PYTHONPATH="$PWD/src:/path/to/pinned-flat/src"
```

Copy `config.example.json` outside the repository and fill in the media paths, manifests, model endpoints, credentials, and pinned runtime commit.

Check the configuration before making model calls:

```bash
$PY -m moha doctor --config /path/to/config.json
```

## Calibration and resume

```bash
$PY -m moha run --config /path/to/config.json --output /path/to/run
$PY -m moha resume --config /path/to/config.json --output /path/to/run
```

For shared calibration, provide the same ordered stack list to every command:

```bash
$PY -m moha shared-doctor \
  --stack model_a=/path/to/model_a.json \
  --stack model_b=/path/to/model_b.json

$PY -m moha shared-run \
  --stack model_a=/path/to/model_a.json \
  --stack model_b=/path/to/model_b.json \
  --output /path/to/shared-run
```

## Evaluation

Use a frozen harness and a test manifest:

```bash
$PY -m moha evaluate \
  --config /path/to/config.json \
  --output /path/to/evaluation \
  --frozen /path/to/frozen.json \
  --manifest /path/to/test.manifest.json
```

Add `--resume` to continue an interrupted evaluation. Keep outputs outside the source checkout.

## Services and tests

Generate or launch the configured observer service with:

```bash
$PY -m moha.serving --config /path/to/config.json --gpu 0,1 --host 0.0.0.0 --dry-run
```

Run the offline test suite with the same pinned runtime:

```bash
PYTHONPATH="$PWD/src:/path/to/pinned-flat/src" \
  $PY -m unittest discover -s tests -v
```
