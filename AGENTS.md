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
- Keep two planner tools. `observe(start_seconds, end_seconds, goal)` selects
  exact valid source-time support, independent of search candidate IDs. Planner
  controls temporal scope; harness controls sampling and observer routing. Keep
  the thin window adapter in `tools.py`, reusing the pinned execution path.
  Receipts and probes must retain the selected window. Interface changes require
  a new full calibration from H0; never reuse candidate-only episode caches.
- Use trace-local Judge proposals, one unweighted vote per failed sample, stable
  catalog-ID tie breaks, and validation-only promotion. Do not reintroduce a
  global LLM selector or confidence weights. Observer proposals follow probes;
  freeze votes within a round and regenerate them after a harness change.
- Keep all original observation claims and conflicting evidence available. Do not
  assume newer observations are correct or infer hidden model reasoning.
- Project planner input through `context.py` before bounded history selection.
  Keep actionable candidates, search/visited state, and retained observation text;
  leave audit-only observation histories in full logs. Do not restore evicted
  facts outside the explicit memory module. A changed projection needs a fresh
  full calibration, including H0; never relabel cached old-policy episodes.
- Memory keeps whole scoped observations and their missing/uncertainty fields,
  without summarizing, clipping claims, or merging same-ID conflicts. Its bounded
  view reserves part of the existing history allowance. Compaction must not
  declare newer evidence authoritative. Reserve the final existing planner call
  for an answer/abstention; never execute tools on it or add a free extra call.
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
- Keep observer output recovery in `observer.py`: unchanged first request, one
  same-media/goal retry for unusable observation JSON or window-relative times,
  capped at 2048 output tokens (or an existing lower cap). Charge and audit both
  attempts; never expose invalid facts as evidence. Exhaustion is explicit tool
  feedback and an inconclusive probe, while infrastructure errors remain fatal.
  Do not post-process valid claims or merge genuine repeated events. Reuse old
  episodes across this change only after verifying the recovery path would not
  activate and recording their original source and artifact hashes in a new run.
- Use one episode batch path for one or multiple endpoint lanes. Keep a separate
  service/client per lane, stable sample-index assignment, and at most one active
  episode per lane. Only the coordinator makes calibration decisions. Stop new
  submissions on errors, preserve completed caches, and probe on the sample's lane.
- Use bounded, independent Judge workers for trace diagnoses (default four).
  Keep probe services/verdict clients serial within the original episode lane;
  restore manifest order before the coordinator freezes votes and checks health.
  Cache each successful diagnosis and stop new submissions on unexpected errors.
- Verify with `PYTHONPATH=src:<runtime-root> python -m unittest discover -s tests -v`.
  Also run the pinned runtime's full test suite when integration changes warrant it.
