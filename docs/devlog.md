# Devlog

## Stage 1: Build

- Scaffolded the Phase 2 layout (src, data, experiments, results, logs, tests, docs).
- Added `src/agents/geospatial_agent.py` (Gap 1): road mask -> skeleton -> NetworkX MultiGraph,
  with spur pruning, degree-2 contraction and per-edge passability from a blocked-road mask.
- Tests in `tests/test_geospatial_agent.py` use synthetic masks (plus, ring, split roads, spur).

## Stage 2: data loaders and E1

- `data/floodnet.py`: loader that refuses anything but 1,445 / 450 / 448, checks id/label pairing and
  disjointness, and freezes the file lists in `data/splits/floodnet_split.json` (run `--build-manifest`, commit it).
- `data/prepare_yolo.py`: exports train + val only to YOLO polygon format. Test is never written out.
- `experiments/train_baseline.py`: fine-tunes YOLOv8x-Seg per seed. COCO weights have no FloodNet classes,
  so E1 is the fine-tuned model; `run_baseline.py` rejects weights that aren't the 9 FloodNet classes.
- `experiments/run_baseline.py`: E1 on the 448 test images, metrics from a dataset-level confusion matrix,
  mean ± std over seeds, written to `results/e1_baseline_metrics.json`.
- Building "damage" in FloodNet = flooded vs non-flooded building. There are no intact/damaged/collapsed levels.

## Stage 3: shared state, Router (Gap 3), Verifier (Gap 2)

- `src/state.py`: `TileState` follows one tile through the pipeline (routing, vision, road graph summary,
  uncertainty report, verdict, step trajectory). Arrays are kept out of JSON.
- `src/components/router.py`: cascade router. The fast path always runs, then a top-k cell entropy score decides
  whether the VLM runs too, and on which regions. `fit_threshold` picks the threshold for a target escalation rate;
  `max_escalation_rate` turns the latency target into that rate. `always_fast` / `always_vlm` modes serve E5 ablation 3.
- `src/components/evidential.py`: Dirichlet statistics (vacuity, entropy, epistemic), numpy EDL reference loss,
  and the torch head and loss.
- `src/components/calibration.py`: streaming ECE, risk-coverage, AURC, threshold fitting by coverage.
- `src/agents/verifier_agent.py`: one threshold applied to tile score, grid regions and road-graph edges; reasons are logged.
- Not yet verified on this machine's sandbox: the two torch tests (skipped without torch). Run `pytest` after installing torch.

## Stage 4: neck hook, uncertainty heads, training script

- `src/components/yolo_features.py`: forward pre-hook on the YOLO Segment head captures the neck features
  [P3, P4, P5]. `extract()` stops right after the capture; `predict_with_features()` returns instances and
  features from one forward pass. Ultralytics runs `predict()` on its own fused copy of the model and rebuilds it
  when arguments such as `device` change, so a hook on `YOLO.model` silently captures nothing; `for_predictor` +
  rebind handles that.
- `src/components/heads.py`: `SoftmaxSegHead` has the same body as `EvidentialSegHead`. Only the output differs.
- `src/components/calibration.py`: `fit_temperature` (NLL, scalar T, fitted on val).
- `src/components/fallback_uncertainty.py`: 1 - instance confidence map, the router input when Gap 2 is ablated in E5.
- `experiments/train_uncertainty_head.py` (+ `configs/head.yaml`): frozen E1 backbone, per-step KL annealing for the
  evidential head, cross-entropy + temperature scaling for the softmax baseline, best epoch by val mIoU.
  Named separately from `train_baseline.py`, which is the E1 YOLO fine-tune.
- Verified on CPU with real Ultralytics 8.4.203 and yolov8n-seg weights, on a fake FloodNet-layout dataset
  (mechanics only; no real-data results exist yet).

## Stage 5: E2 surrogate baseline, E3 orchestration, shared cell protocol

- **Change since Phase 1 (E2 baseline).** SkyEyeGPT has no public checkpoint (the official repo says the model
  and checkpoint are "coming soon"; only the SkyEye-968k dataset is on Hugging Face). E2 uses zero-shot
  Qwen2-VL-7B-Instruct as a surrogate monolithic VLM. The paper must say this and treat it as a limitation.
- **Shared scoring protocol** (`src/utils/cells.py`). A VLM outputs text, not masks, so pixel mIoU cannot compare
  it with E1/E3. All systems are also reduced to a 4x6 grid; per cell and task (building, road) the state is
  absent / dry / flooded, scored with F1. Cell thresholds (5% building, 3% road) were fixed before any result.
  Paired bootstrap over tiles is used for significance; E2 is deterministic (greedy), so it has one run.
- **Heavy path semantics.** The VLM re-judges flagged regions with the same cell prompt as E2. Where it says
  flooded or dry, building/road pixels in that region take that state; "none" never erases a detection;
  unparsable replies change nothing and are counted.
- **E3 pipeline** (`src/pipeline.py`): one YOLO forward pass yields instances and neck features; evidential head;
  router; VLM on flagged regions; geospatial agent; verifier. E5 ablations are config switches
  (`--ablation no_gap1|no_gap2|no_gap3`), so E3 and E5 share one code path.
- **Fitting protocol** (`experiments/run_e3.py`): thresholds are fitted on val only. t_fast and the extra
  escalation cost are measured on val tiles, converted to a maximum escalation rate for the 2.0 s target, and the
  router threshold is the one that respects that rate. Verifier threshold = target coverage on val.
- **Bugs found while testing:** `max_escalation_rate` returned a negative rate when the measured escalation cost
  was zero or negative (timing noise); fixed. `imgsz` was not an explicit E3 config key, so the Gap 2 ablation
  silently fell back to a default; fixed.
- E1 (`run_baseline.py`) now also emits cell metrics and per-tile confusions so E1, E2, E3 are comparable.

## Stage 6: route-query benchmark (Gap 1) and comparison script

- **Gap 1 benchmark** (`src/eval/route_benchmark.py`, `experiments/run_gap1_routes.py`). Question: is there a
  passable route between points A and B? Ground truth is pixel connectivity of the ground-truth dry road
  (8-connected components after the same cleaning the agent uses), so it does not depend on the road graph.
  Points are sampled on the dry-road centreline in three pair types (passable, blocked by flooding,
  disconnected) with a fixed seed, so every system answers the same query set. Systems: predicted label maps
  answered by connected components (`pixel_cc`) or by the road graph (`graph`), an oracle on ground-truth masks,
  and direct VLM prompting with the two points marked.
- **Bug found by the benchmark.** On clean ground-truth masks the road graph disagreed with pixel connectivity:
  a road that is dry, then flooded, then dry was one edge, so it was blocked as a whole and the dry stretches
  at its ends became unreachable. `split_edges_at_flood_boundaries` now cuts edges where the flood starts and
  ends (short dry holes inside a flood are bridged, flood runs shorter than `min_blocked_run_px` are ignored).
  Oracle agreement is tested across 40 tiles. `oracle/graph` is reported as the error floor of the abstraction.
- **Sensitivity to note.** `min_blocked_run_px` (20 processing px) controls the false-passable rate. All real
  tiles are normalised to 1024 px, so it is consistent across tiles; tune it on val with this benchmark.
- **Comparison script** (`src/eval/compare.py`, `experiments/compare_results.py`). Builds the gap-resolution,
  main, significance (paired bootstrap over tiles), ablation and route tables as Markdown, LaTeX and JSON.
  Missing experiments are listed, never filled in; non-official (smoke/limited) files raise a warning.
  "Addressed?" follows a written rule (Fully = all targets met and a significant improvement; Partially = a
  significant improvement or at least one target met; Not = neither). Targets come from
  `experiments/configs/success_criteria.yaml` (Phase 1 values; the F1 target is flagged for revision).
- Gap 3's base column is E2 (monolithic VLM); the no-router ablation (E5/no_gap3) is shown as extra evidence.
- `run_e3 --save-preds DIR` writes final label maps (`DIR/seed<N>/<id>.png`); E1/E2 now also store per-tile latency.
- Real-data-path checks found two more bugs: NaN in the route bootstrap when a metric is undefined (now null),
  and the comparison crashing on those nulls (now reported as "not evaluable").

## Stage 7: Gap 2 calibration experiment, seed-aware runbook

- **Gap 2 experiment** (`src/eval/gap2.py`, `experiments/run_gap2_calibration.py`). Five systems on the same test pixels,
  all from the same E1 backbone: YOLO instance confidence, softmax, softmax + temperature scaling, evidential, and
  evidential + temperature scaling (an extra control: evidential heads trained with a KL term tend to be
  underconfident). Primary ECE is on "covered" pixels (those a YOLO instance claims), where every system makes a
  claim; heads are also scored on all pixels. NLL and Brier score are reported for the heads. ECE and risk-coverage
  (AURC with entropy and vacuity) are kept as separate claims.
- **Significance.** Per-tile bin statistics are stored, so ECE gets a bootstrap CI over tiles and the comparison
  with the baselines is a paired bootstrap (`paired_bootstrap_ece`). The comparison script uses it for the Gap 2 row.
- **Fairness choices.** Softmax and evidential heads share architecture, features and output grid; the script refuses
  heads trained with different settings. Temperatures are fitted on val only.
- **Windows encoding bug** (found by running the suite on Windows): `Path.write_text` without an encoding failed on
  the Greek delta in the report. Every text read/write now names `utf-8`, and `tests/test_encoding.py` fails if one
  reappears (it also reproduces the failure under an ASCII locale).
- `run_baseline --save-preds` now writes one folder per weights file (`DIR/seed<N>/`), like E3, so the route
  benchmark can use every seed. `min_blocked_run_px` is a config key for E3 and a CLI option for the route benchmark;
  `experiments/summarize_sweep.py` compares settings on val.
- `docs/RUNBOOK.md`: the master execution order with checks after each stage.

## Stage 8: Dataset Protocol Resolution, Documentation Alignment & Kaggle GPU Migration

- **Windows Encoding Fix & Linting:** Resolved `UnicodeEncodeError: 'charmap'` when writing mathematical symbols ($\Delta$) into comparison Markdown tables under Windows default `cp1252` encoding. Enforced `encoding="utf-8"` across all 49 file I/O calls and added `tests/test_encoding.py` to prevent regressions. All 224 pytest unit/integration tests pass cleanly.
- **Proposal Documentation Sync:** Updated the Phase 1 proposal document (`Phase1_G1-11_22I-0854`) Section 7 (Technical Risks) to document cross-platform I/O encoding mitigations, and Section 6.D (Ablations) to include the Gap 2 evidential calibration evaluation methodology.
- **Dataset Ingestion Protocol:** Identified missing validation and test masks in the EarthVision Challenge dataset download. Re-aligned data ingestion with the official post-competition `FloodNet-Supervised_v1.0` release (2,343 images: 1,445 train / 450 val / 448 test) to satisfy `data/floodnet.py` manifest verification.
- **Kaggle GPU Migration:** Configured execution workflow for Kaggle Notebooks (T4 x2 / P100 GPU accelerators) to execute Stage 1 baseline fine-tuning, Stage 3 uncertainty head training, and Stage 6 heavy VLM zero-shot evaluation without local hardware constraints.
- **Distributed Git Strategy & README Overhaul:** Updated `README.md` to reflect the Phase 2 multi-agent directory tree, $4 \times 6$ grid-cell scoring, and `RUNBOOK.md` stage dependencies. Structured a 13-commit git history divided across proposal team roles:
  - **Abdurrehman (Core Developer: Vision & Data):** Pushed foundational data split enforcement, YOLO feature hooks, Qwen2-VL backend, and execution runbook (`feat(data)`, `feat(vision)`, `build(infra)`).
  - **Zuhaak (Core Developer: Geospatial & Benchmarks):** Road graph skeletonization, flood-boundary edge splitting, dynamic router, grid-cell metrics, and route-query benchmark.
  - **Momin (Group Head: Orchestration & Verifier):** Shared state schema, Dirichlet evidential head, Coordinating Verifier Agent, and Gap 2 calibration baseline.