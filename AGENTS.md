# MoHA working agreement

- This directory is the independent MoHA Git repository. Keep one current
  implementation under `src/moha`, tests under `tests`, and one README.
- Use Git history for previous code. Do not add backup copies, dated source
  directories, alternate-version scripts, or experiment outputs in this tree.
- Video OS is an external, read-only runtime dependency, selected explicitly by
  `config.runtime.root` and its exact `commit`. Do not discover versions by name,
  fall back to the parent directory, or duplicate its tools/model adapters here.
- Preserve the paper's shared H0, catalog-constrained single-coordinate proposals,
  calibration/validation separation, fixed-support observer probes, and validation
  gate. Diagnoses are hypotheses, not fixed intervention-routing rules.
- Use trace-local Judge proposals, one unweighted vote per failed sample, stable
  catalog-ID tie breaks, and validation-only promotion. Do not reintroduce a
  global LLM selector or confidence weights. Observer proposals follow probes;
  freeze votes within a round and regenerate them after a harness change.
- Keep all original observation claims and conflicting evidence available. Do not
  assume newer observations are correct or infer hidden model reasoning.
- Commit coherent changes before real model calls. Both this checkout and the
  pinned runtime must be clean. Record source/runtime commits, input/config hashes,
  provider identities, command, environment, and output location for every run.
- Store actual run configs, credentials, logs, checkpoints, and bulk run outputs
  outside this repository. User-requested `calibrated/` holds only checked,
  credential-free planner/observer profiles exported from completed calibrations;
  keep one current profile per pair and its provenance. Never label an unfinished
  run or a template as calibrated. Use Git history for older profiles.
- Never delete completed experiments to make the source look clean.
- Existing runs use their frozen source; do not edit their code or artifacts in place.
- Verify with `PYTHONPATH=src:<runtime-root> python -m unittest discover -s tests -v`.
  Also run the pinned runtime's full test suite when integration changes warrant it.
