# Calibration profiles

This directory is reserved for reviewed, credential-free planner and observer
profiles exported from completed calibrations. Keep datasets, videos, logs,
credentials, endpoint secrets, and per-sample traces outside the repository.

Export a completed run with:

```bash
PYTHONPATH=src:/path/to/flat/src python -m moha export \
  --run /path/to/completed/run --output calibrated
```

Review the result before committing it. A profile is not a replacement for the
exact Flat runtime commit, input manifests, or model service configuration used
by the run.
