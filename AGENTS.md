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
- Keep two Video OS tools, plus module tools only when enabled: memory_read /
  memory_note for memory and verify_fresh for verification. The user authorized
  these native module tools in place of the separate MCP experiment.
  `observe(start_seconds, end_seconds, goal)` selects
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
- Memory is an append-only result ledger and a separate working-note ledger.
  Keep original observations, duplicate records, conflicts, scope and caveats.
  Planner reads original records and writes notes explicitly; no summaries,
  ranking, automatic merging or fixed early/recent memory selection.
- Verification is one fresh text-only prompt over original observations, without
  working notes, prior planner dialogue, screenshots, rule-based coverage, verdict
  schemas or repair loops. Without memory, do not restore history-evicted evidence.
  Charge diagnosis requests to the shared max_steps model-call allowance and
  preserve one final planner response; never force diagnosis or block an answer.
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
- Use bounded, independent Judge workers for trace diagnoses (default eight).
  Keep probe services/verdict clients serial within the original episode lane;
  restore manifest order before the coordinator freezes votes and checks health.
  Cache each successful diagnosis and stop new submissions on unexpected errors.
- Verify with `PYTHONPATH=src:<runtime-root> python -m unittest discover -s tests -v`.
  Also run the pinned runtime's full test suite when integration changes warrant it.

- Source-relative observer execution lives in `execution.py`: target sampling rate
  0.5/1/2 FPS, source edge ratio and temporal/spatial/balanced priority. Translate
  rates to min(ceil(rate * window duration), 128) explicit target frames before
  budget allocation. Preserve auto (1 FPS) and fixed-frame renderer primitives,
  but do not mix an explicit rate with fixed frames or present frame counts as rates.
  Keep the shared
  feasible-set allocator, pinned renderer/token accounting, and existing OCR/ASR
  routes. Do not revive fixed-resolution presets or silently migrate old profiles.
  Probe one fixed request with one fresh control and at most six distinct
  one-coordinate alternatives; preflight media no-ops without model calls and
  require the actual receipt to verify a change before crediting a rescue.
- With `perception_calibration=true`, structural adaptation keeps H0 visual
  execution and changes only support modules/specialists. Specialist attribution
  retains its fixed-support execution probes; other execution preferences are
  deferred to the final validation stage rather than gated by Judge labels.
  Freeze structure, then evaluate 0.5/1/2 FPS by 0.5/0.75/1.0 source edge ratios,
  balanced priority and shared f_view=128. Use the same complete validation samples
  and repeats for all nine policies, a separate final-stage episode cache, measured
  utility and deterministic ties. Preserve the structural promotion gate; final
  selection is a validation argmax. Never export a partial sweep as calibrated.
  Keep actual window duration, target and realized frames/FPS/resolution and cap
  truncation in receipts. Fixed planner settings do not imply fixed trajectories.
