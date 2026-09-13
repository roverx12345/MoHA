# MoHA working agreement

- This directory is the independent MoHA Git repository. Keep one current
  implementation under `src/moha`, tests under `tests`, and one README.
- Use Git history for previous code. Do not add backup copies, dated source
  directories, alternate-version scripts, or experiment outputs in this tree.
- Flat is an external, read-only runtime dependency, selected explicitly by
  `config.runtime.root` and its exact `commit`. Do not discover versions by name,
  fall back to the parent directory, or duplicate its tools/model adapters here.
- Preserve the paper's shared H0, catalog-constrained single-coordinate proposals,
  calibration/validation separation, fixed-support observer probes, and validation
  gate. Diagnoses are hypotheses, not fixed intervention-routing rules.
- Keep two Flat tools and verify_fresh only when verification is enabled.
  The user authorized replacing callable memory tools with automatic persistent
  evidence injection on 2026-09-12. Do not expose memory_read or memory_note.
  `observe(start_seconds, end_seconds, instruction, evidence_type, reference?)` selects
  exact valid source-time support, independent of search candidate IDs. Planner
  controls temporal scope; harness controls sampling and observer routing. Keep
  the thin window adapter in `tools.py`, reusing the pinned execution path.
  Receipts and probes must retain the selected window. Interface changes require
  a new full calibration from H0; never reuse candidate-only episode caches.
  `search(query, start_seconds, end_seconds, top_k?)` requires an explicit valid
  source-time range. Filter candidates before ranking and never silently widen
  an empty search. Retain each search's bounds in results and navigation history.
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
  Tool-result events store the actual public return in `result` and execution
  provenance separately in `audit`. Do not reinsert legacy runtime envelopes.
  Omni uses Flat's instruction-only prompt and payload. Current observer audits
  retain instruction/evidence_type/reference; readers also accept immutable old
  typed requests. Do not rewrite historical requests to look like current ones.
- Memory is an append-only archive of original perceptual observations. Preserve
  duplicates, conflicts, scope and caveats. Automatically inject only records absent
  from actual bounded history as user-role observation data before recent history.
  Keep exact-copy deduplication in the view only; never merge conflicting claims,
  summarize observations or retain planner notes. memory_basic remains one boolean
  candidate; do not add a memory routing/selection catalog.
  The restored block uses at most 6000 estimated tokens inside history_tokens.
  Recompute history after reservation so newly evicted evidence is included too.
  Restore the complete missing set, or keep ordinary bounded history and explicitly
  report capacity limits. No relevance ranking or partial conflict selection.
  Keep raw archives and per-call visibility/budget audits. H0 uses the unchanged
  bounded-history path. Existing tool-memory runs retain their frozen implementation.
- Verification is an answer-audit capability with one automatic pre-submit or
  two-call budget-floor trigger, at most one audit including manual verify_fresh,
  and exactly one final planner response afterward with all tools disabled.
  Reserve those calls inside max_steps; do not add compute or corrective perception.
  Audit every complete option, the candidate, visible original observations with their
  full source context and caveats, and explicit unverified planner hypotheses in a
  fresh text-only context. Verification must read only observations present in the
  actual projected planner input and never access the memory archive directly; memory
  composes with it only through ordinary context injection. No working-note ledger,
  prior dialogue, screenshots or reference answer. Focus IDs must not hide contrary
  available evidence. Request the support/option-checks/best-option/diagnosis JSON
  audit; validate option coverage and source IDs once, retain invalid raw output
  explicitly, and never retry the audit.
  The final planner may keep, revise or abstain; the audit cannot impose an answer.
- Keep the terminal prompt transport-neutral and do not add output-schema or verbose
  no-tool instructions to suppress provider behavior. The final call advertises no
  executable tools. If a provider returns exactly one unadvertised `submit_answer`,
  `answer` or `final_answer` call with the complete terminal payload, normalize it
  as the model's answer without executing a tool. Reject mixed calls, unknown fields
  and ambiguous values.
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
  same-media/instruction retry for unusable observation JSON or window-relative times,
  capped at 2048 output tokens (or an existing lower cap). Charge and audit both
  attempts; never expose invalid facts as evidence. Exhaustion is explicit tool
  feedback and an inconclusive probe, while infrastructure errors remain fatal.
  Do not post-process valid claims or merge genuine repeated events. Reuse old
  episodes across this change only after verifying the recovery path would not
  activate and recording their original source and artifact hashes in a new run.
- Use one episode batch path for one or multiple endpoint lanes. Keep a separate
  service/client per lane, stable sample-index assignment, and at most one active
  episode per lane. The shared Qwen3-Omni vLLM server must use max_num_seqs=1;
  heterogeneous concurrent multimodal requests can corrupt placeholder alignment.
  Only the coordinator makes calibration decisions. Stop new
  submissions on errors, preserve completed caches, and probe on the sample's lane.
- Use bounded, independent Judge workers for trace diagnoses (default eight).
  Keep probe services/verdict clients serial within the original episode lane;
  restore manifest order before the coordinator freezes votes and checks health.
  Cache each successful diagnosis and stop new submissions on unexpected errors.
- Verify with `PYTHONPATH=src:<runtime-root>/src python -m unittest discover -s tests -v`.
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
