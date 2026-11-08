# Agentic AI for Disaster Scene Analysis from UAV Imagery (G1-11)

**AI4015 Agentic Artificial Intelligence (Fall 2026\)**  
**FAST National University of Computer & Emerging Sciences, Islamabad** **Supervisor:** Muhammad Sajjad

## Team Members (Group G1-11)

* **Muhammad Momin Munir Misbah** (22I-0854) \- *Group Head*  
* **Abdurrehman Ahmed Siddiqui** (22I-0988)  
* **Zuhaak Ahmed** (22I-1352)

---

## Project Overview

Following natural disasters (e.g., hurricanes, floods), emergency responders require rapid, automated analysis of aerial imagery to direct search-and-rescue operations. This project develops an agentic AI framework combining Vision-Language Models (VLMs), semantic segmentation, and geospatial graph reasoning to evaluate high-resolution UAV imagery across two primary decisions:  
The system utilizes a unified \$4 \\times 6\$ grid-cell scoring methodology to align pixel-level segmentations with VLM text outputs.

1. **Building Damage Assessment:** Classifying structural collapse and damage levels across affected grid tiles.  
2. **Road Passability Analysis:** Identifying clear, flooded, or debris-blocked access routes.

---

## Repository Structure

```
A14015-Project-G1-11/
├── docs/
│   ├── RUNBOOK.md                  # Master execution checklist and rules
│   └── devlog.md                   # Iterative development log (Changes since Phase 1)
├── experiments/
│   ├── configs/                    # YAML configs for baseline, E2, E3, and gap2 calibration
│   ├── run_baseline.py             # E1 baseline evaluation (writes cell metrics)
│   ├── train_baseline.py           # E1 YOLOv8x-Seg fine-tuning
│   ├── train_uncertainty_head.py   # Evidential and softmax head training
│   ├── run_vlm_zeroshot.py         # E2 Qwen2-VL monolithic surrogate evaluation
│   ├── run_e3.py                   # E3 full pipeline orchestration and E5 ablations
│   ├── run_gap1_routes.py          # Gap 1 route benchmark runner
│   ├── run_gap2_calibration.py     # Gap 2 evidential calibration baseline experiment
│   ├── summarize_sweep.py          # Sweeps min_blocked_run_px on val
│   └── compare_results.py          # Assembles statistical comparison tables (Markdown/LaTeX)
├── src/
│   ├── agents/                     # Verifier, Baseline Verifier, and Geospatial Agents
│   ├── components/                 # YOLO hooks, Router, Evidential math, and Calibration
│   ├── eval/                       # Gap 1, Gap 2, and compare modules
│   ├── utils/                      # Shared cells and segmentation utils
│   ├── vlm/                        # Qwen2-VL backends and prompts
│   ├── pipeline.py                 # E3 multi-agent orchestration
│   └── state.py                    # Multi-agent shared state schema
├── data/
│   ├── floodnet.py                 # Dataset loader enforcing 1445/450/448 split
│   └── prepare_yolo.py             # YOLO polygon mask conversion
└── tests/                          # 224 unit and integration tests (pytest)
```

---

## Setup & Execution

### 1\. Environment Installation

```sh
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

*Note: Ensure you have `transformers>=4.45.0` and `accelerate>=0.30.0` installed for Qwen2-VL support.*

### 2\. Data Preparation

Download the official `FloodNet-Supervised_v1.0` release and place it in the `data/` folder.

### 3\. Execution Pipeline

This repository enforces a strict, reproducible evaluation protocol across 3 random seeds (42, 123, 456). To run the training, evaluations, and ablation studies in the correct dependency order, you **must** follow the 11-stage master checklist provided in `docs/RUNBOOK.md`. Do not run scripts out of order.  
