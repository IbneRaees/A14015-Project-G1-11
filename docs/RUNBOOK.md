# Master execution runbook

Everything below runs from the repository root, inside the venv, in PowerShell. Seeds are **42, 123, 456**.
Every stage lists what it needs, the command, and what to check before moving on. Do not skip the checks;
most of the mistakes that cost a rerun show up there.

```powershell
$ROOT = "data/FloodNet-Supervised_v1.0"      # official release: train/ val/ test/ folders
$SEEDS = 42, 123, 456
```

## Rules that keep the results honest

1. **Nothing is fitted or tuned on test.** Temperatures, the router threshold, the verifier threshold and
   `min_blocked_run_px` all come from val. The scripts enforce this for their own thresholds; the rest is on you.
2. **Freeze before the first test run, and write it in the devlog:** YOLO `conf` (0.25), the 4x6 cell grid and its
   thresholds, `target_latency_s`, `target_coverage`, the chosen `min_blocked_run_px`, `success_criteria.yaml`.
3. A run only counts if its JSON says `"official_run": true` (full test split, strict 1,445/450/448 split, no
   `--limit`, no `--smoke-counts`). `compare_results` warns about anything else.
4. If a stage has to be rerun after you have seen test numbers, say so in the devlog and say why.

## What depends on what

```
Stage 0 preflight
   |
   +--> Stage 1  E1 fine-tune (x3 seeds) ---> Stage 2 E1 eval + saved label maps
   |                  |
   |                  +--> Stage 3 heads: evidential + softmax (x3 seeds each)
   |                          |
   |                          +--> Stage 4 Gap 2 calibration
   |                          |
   |      Stage 5 tune min_blocked_run_px on val (needs Stage 2 val maps)
   |                          |
   |                          +--> Stage 7 E3 (+ saved label maps) --> Stage 8 E5 ablations
   |                                              |
   +--> Stage 6  E2 VLM zero-shot (independent)   +--> Stage 9 route benchmark (needs Stage 2 + 7 maps)
                                                                  |
                                    Stage 10 compare  <-----------+  (needs everything above)
```

Stages 6 can run any time after Stage 0, but it needs the GPU for hours: do not run it at the same time as a
training stage.

---

## Stage 0 - Preflight

- [ ] `pip install -r requirements.txt` finished without errors; `python -c "import torch; print(torch.cuda.is_available())"` prints `True`.
- [ ] `pytest` passes. Then run the gated real-Ultralytics test once:
      `$env:YOLO_SMOKE_WEIGHTS = "<path to any yolov8*-seg.pt>"; pytest tests/test_yolo_hook.py`
- [ ] Verify the split and freeze it:
      ```powershell
      python -m data.floodnet --root $ROOT --build-manifest
      python -m data.floodnet --root $ROOT --verify
      ```
      Expect counts 1445 / 450 / 448. **Commit `data/splits/floodnet_split.json`.** If the counts differ, stop: the
      dataset version is not the one the protocol assumes.
- [ ] Open two label PNGs and check the values are class ids 0-9 (the loader raises if they are colour-coded).
- [ ] If DataLoader workers hang on Windows, set `num_workers: 0` in `experiments/configs/head.yaml`.

## Stage 1 - E1 fine-tune (3 seeds)

Needs: Stage 0. Output: `runs/e1/seed<S>/weights/best.pt`.

```powershell
python -m experiments.train_baseline --seed 42 --data-root $ROOT      # time this one first
foreach ($s in 123, 456) { python -m experiments.train_baseline --seed $s --data-root $ROOT }
```

- [ ] After seed 42, note the wall-clock time and multiply by 3 before committing to the rest. If it is too long,
      lower `epochs` in `configs/e1_baseline.yaml` **for all seeds** and log the change.
- [ ] Ultralytics' own validation numbers (in `runs/e1/seed42/results.csv`) look sane (not near zero, not still climbing).

## Stage 2 - E1 evaluation and saved label maps

Needs: Stage 1. Writes `results/e1_baseline_val_metrics.json` (val) and `results/e1_baseline_metrics.json` (test).

```powershell
python -m experiments.run_baseline --split val  --data-root $ROOT --save-preds results/preds/e1_val
python -m experiments.run_baseline              --data-root $ROOT --save-preds results/preds/e1
```

- [ ] Test run reports `n_runs: 3`, `strict_split: true`, `manifest_verified: true`.
- [ ] Label maps exist: `results/preds/e1/seed42/`, `seed123/`, `seed456/` (448 PNGs each), and the same under `e1_val` (450 each).
- [ ] Building and road F1 are not 0.0 (a zero means the model predicts nothing; go back to Stage 1).

## Stage 3 - Uncertainty heads (3 seeds x 2 heads)

Needs: Stage 1. Output: `runs/head/<head>_seed<S>/best.pt`, `history.json`, `train_summary.json`.

```powershell
python -m experiments.train_uncertainty_head --head evidential --seed 42 --data-root $ROOT --val-max-images 100 --epochs 3 --out-dir runs/head_check   # throwaway plumbing check
foreach ($s in $SEEDS) { foreach ($h in "evidential","softmax") {
    python -m experiments.train_uncertainty_head --head $h --seed $s --data-root $ROOT } }
```

(The first command is a throwaway check that writes to `runs/head_check`; delete that folder afterwards.)

- [ ] In every `train_summary.json` the softmax run has a fitted `temperature` that is not 1.0.
- [ ] Val mIoU in `history.json` is rising and the best epoch is not the very first. If the evidential head's val ECE
      is large and mIoU is fine, it is probably underconfident (common with the KL term). That is a result, not a
      bug: Stage 4 reports it, and `evidential_ts` shows what a temperature does for it. Do not tune on test.
- [ ] Each seed's heads were trained on the **same seed's** E1 weights (`yolo_weights` in the checkpoint meta).

## Stage 4 - Gap 2 calibration

Needs: Stages 1 and 3. Writes `results/e4_gap2_calibration.json` and `results/figures/gap2_reliability.png`.

```powershell
python -m experiments.run_gap2_calibration --data-root $ROOT --plot
```

- [ ] `n_runs: 3`, `official_run: true`. The script warns if a softmax checkpoint has temperature 1.0.
- [ ] Open the reliability figure: diagonal is perfect calibration. Compare `softmax_raw` with `softmax_ts` and
      `evidential` with `evidential_ts`.

## Stage 5 - Tune `min_blocked_run_px` on val

Needs: Stage 2 (val label maps). This decides how long a flooded stretch must be before the road graph treats the
edge as blocked, and it moves the false-passable rate directly.

```powershell
New-Item -ItemType Directory -Force results/tune | Out-Null
foreach ($m in 10, 15, 20, 30, 40) {
    python -m experiments.run_gap1_routes --data-root $ROOT --split val `
        --preds e1_s42=results/preds/e1_val/seed42 --min-blocked-run-px $m `
        --output results/tune/routes_val_m$m.json --queries-out results/tune/queries_val.json }
python -m experiments.summarize_sweep results/tune/routes_val_m*.json --system e1_s42/graph
```

- [ ] All files share one query fingerprint (the summary refuses mixed query sets).
- [ ] Pick the value with the lowest connectivity error; if two are close, prefer the one with the lower
      false-passable rate (the dangerous error). Write the choice in the devlog.
- [ ] Put it in `experiments/configs/e3_pipeline.yaml` (`min_blocked_run_px`) and use the same value in Stage 9.
      The default of 20 is untested on real data.
- [ ] Check `oracle/graph` in one of the outputs: its error is the floor of the graph abstraction on clean masks.

## Stage 6 - E2 zero-shot VLM (independent)

Needs: Stage 0 and the Qwen2-VL weights (downloaded on first use). 24 VLM calls per tile, so 10,752 calls for the
448 test tiles. Resumable: progress is appended to `logs/e2/tiles.jsonl`.

```powershell
python -m experiments.run_vlm_zeroshot --data-root $ROOT --limit 5      # timing only; its log goes to logs/e2/partial
python -m experiments.run_vlm_zeroshot --data-root $ROOT                # full run; rerun the same command to resume
```

- [ ] From the 5-tile timing, estimate the total (seconds per tile x 448) before starting the full run.
- [ ] If it runs out of VRAM, add `--load-in-4bit` (bitsandbytes on Windows can be fiddly).
- [ ] The output has `parse_failure_rate` close to 0. A high value means the prompt is not being followed: look at
      a few lines of `logs/e2/answers.jsonl` before trusting the scores.
- [ ] One run only (greedy decoding is deterministic). Significance comes from the bootstrap in Stage 10.

## Stage 7 - E3 full pipeline

Needs: Stages 1, 3, 5 (for `min_blocked_run_px`). Loads YOLOv8x, the head and Qwen2-VL together (about 16 GB for
the VLM in bf16, plus the rest). Per seed it first fits thresholds on val, then evaluates on test.

```powershell
python -m experiments.run_e3 --data-root $ROOT --seeds 42 --split val --fit-tiles 50 --calib-tiles 5   # plumbing check, writes e3_smoke_metrics.json
python -m experiments.run_e3 --data-root $ROOT --save-preds results/preds/e3                           # the real run, 3 seeds
```

- [ ] Look at `runs[].fit` for each seed: `t_fast_s`, `t_vlm_extra_s`, `max_escalation_rate`, `router_threshold`.
      If `latency_target_feasible` is `false`, the fast stage alone is slower than the 2.0 s target; the router
      then never escalates and E3 is just the fast path. Report that as it is.
- [ ] `routing.escalation_rate` is at or below `max_escalation_rate` on test (thresholds were fitted on val, so a
      small difference is normal).
- [ ] Label maps exist: `results/preds/e3/seed42/` etc., 448 PNGs each.
- [ ] `latency.mean_s` is measured on the same machine as E1.

## Stage 8 - E5 ablations

Needs: Stage 7 settings. Each one refits its own thresholds on val.

```powershell
python -m experiments.run_e3 --data-root $ROOT --ablation no_gap1     # no road graph
python -m experiments.run_e3 --data-root $ROOT --ablation no_gap2     # YOLO confidences instead of the evidential head
python -m experiments.run_e3 --data-root $ROOT --ablation no_gap3     # VLM on every cell, no router
```

- [ ] `no_gap3` is the expensive one: 10,752 VLM calls per seed. If time is short, run it with `--seeds 42` only and
      say so in the paper; the other two are cheap.
- [ ] Outputs: `results/e5_no_gap1_metrics.json`, `e5_no_gap2_metrics.json`, `e5_no_gap3_metrics.json`.

## Stage 9 - Gap 1 route benchmark (test)

Needs: Stage 2 and Stage 7 label maps, and the value from Stage 5. Run it once without the VLM, then add `--vlm`
(slow, resumable) for the direct-prompting comparison that RQ1 needs.

```powershell
$preds = foreach ($s in $SEEDS) { "--preds"; "e1_s$s=results/preds/e1/seed$s"; "--preds"; "e3_s$s=results/preds/e3/seed$s" }
python -m experiments.run_gap1_routes --data-root $ROOT --min-blocked-run-px 20 @preds
python -m experiments.run_gap1_routes --data-root $ROOT --min-blocked-run-px 20 @preds --vlm   # replace 20 with your Stage 5 value in both
```

- [ ] `queries_by_type` has all three kinds (passable, blocked, disconnected) in reasonable numbers.
- [ ] `oracle/pixel_cc` has error 0.0 (it must; if not, something is wrong with the benchmark).
- [ ] `oracle/graph` error is small. Whatever it is, report it as the floor.
- [ ] The final file has a `vlm/direct` system; without it the Gap 1 row of the table cannot be evaluated.

## Stage 10 - Compare

Needs: all of the above.

```powershell
python -m experiments.compare_results
```

- [ ] No `WARNING:` lines (they mean a smoke, limited or val run is being read as if it were official).
- [ ] "not yet available" is empty. Any file listed there is an experiment you have not run.
- [ ] Open `results/comparison/comparison.md`. The gap-resolution table is the headline; its last column
      gives the numbers behind each "Addressed?" so you can write the sentence yourself.
- [ ] `results/comparison/tables/*.tex` are ready for the paper (wide tables use `table*`).

## Stage 11 - Freeze

- [ ] Commit `results/*.json`, `results/comparison/`, `results/figures/`, the manifest, the configs as run, and the
      devlog. Do **not** commit `runs/` (weights) or `results/preds/`.
- [ ] `git tag` the commit the numbers came from, and put the hash in the paper's reproducibility note
      (every result JSON also records `git_commit`).
- [ ] Re-run `pytest` on that commit.

---

## Seeds at a glance

| Item | How seeds are handled | Output naming |
|---|---|---|
| E1 fine-tune | one training per seed | `runs/e1/seed<S>/` |
| E1 eval | all seeds in one call, mean ± std | `results/e1_baseline_metrics.json` |
| Heads | one run per seed per head, tied to that seed's E1 weights | `runs/head/<head>_seed<S>/` |
| Gap 2 calibration | all seeds in one call, mean ± std, bins pooled for the bootstrap | `results/e4_gap2_calibration.json` |
| E2 | single deterministic run (no seeds) | `results/e2_vlm_zeroshot_metrics.json` |
| E3 / E5 | all seeds in one call; thresholds refitted per seed on val | `results/e3_proposed_metrics.json`, `e5_*` |
| Route benchmark | one source per seed (`e3_s42`, ...), pooled by family in the comparison | `results/e4_gap1_routes.json` |

The query set of the route benchmark is fixed by `--seed` (default 0) and does not change with the system seeds.

## If something goes wrong

| Symptom | Likely cause |
|---|---|
| `weights not found` | Stage 1 not finished for that seed, or `yolo_weights` template in the config points elsewhere |
| `heads differ in 'yolo_weights'` | evidential and softmax heads were trained on different backbones; retrain the odd one |
| `SplitError: ... expected 1445` | wrong dataset version or a partial copy |
| `route=vlm but no VLM backend` in a trajectory | E3 ran without a VLM; entropy routing needs one |
| CUDA out of memory in E3 | set `vlm_load_in_4bit: true` in `e3_pipeline.yaml`, or free the GPU from other jobs |
| `latency_target_feasible: false` | the fast stage is slower than the target on this machine; report it, do not hide it |
| comparison says "Not evaluable" for Gap 1 | the routes file has no `vlm/direct` (run Stage 9 with `--vlm`) |
