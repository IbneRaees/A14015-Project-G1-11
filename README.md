# Agentic AI for Disaster Scene Analysis from UAV Imagery (G1-11)

**AI4015 Agentic Artificial Intelligence (Fall 2026)**  
**FAST National University of Computer & Emerging Sciences, Islamabad**
**Supervisor:** Muhammad Sajjad  

## Team Members (Group G1-11)
* **Muhammad Momin Munir Misbah** (22I-0854) - *Group Head*
* **Abdurrehman Ahmed Siddiqui** (22I-0988)
* **Zuhaak Ahmed** (22I-1352)

---

## Project Overview

Following natural disasters (e.g., hurricanes, floods), emergency responders require rapid, automated analysis of aerial imagery to direct search-and-rescue operations. This project develops an agentic AI framework combining Vision-Language Models (VLMs), semantic segmentation, and geospatial graph reasoning to evaluate high-resolution UAV imagery across two primary decisions:
1. **Building Damage Assessment:** Classifying structural collapse and damage levels across affected grid tiles.
2. **Road Passability Analysis:** Identifying clear, flooded, or debris-blocked access routes.

---

## 📊 Phase 1 Baseline Benchmark Results

The non-agentic single-pass baseline was benchmarked using YOLOv8 instance segmentation (`yolov8n-seg`) across 10 authentic post-disaster UAV tiles from the FloodNet dataset:

| Metric | Baseline Value |
| :--- | :--- |
| **Evaluated Model** | Single-pass YOLOv8 Segmentation (`yolov8n-seg.pt`) |
| **Dataset Evaluated** | 10 Real FloodNet UAV Aerial Image Tiles |
| **Tile Resolution** | $4000 \times 3000$ to $4592 \times 3072$ pixels |
| **Average Frame Latency** | **1.9470 seconds/frame** |
| **Data Integrity Status** | 100% verified (10/10 valid images) |

---

## Repository Structure

```text
A14015-Project-G1-11/
├── data/
│   ├── floodnet_sample/          # 10 real FloodNet UAV sample tiles
│   └── process_real_data.py      # Dataset inspection & verification script
├── baseline/
│   └── run_baseline.py           # Single-pass non-agentic baseline script
├── src/                          # Agentic system framework (Phase 2 & 3)
│   ├── agents/                   # Vision, Geospatial, and Coordinating Agents
│   └── utils/                    # Graph reasoning & confidence evaluation tools
├── .gitignore                    # File exclusion configurations
├── requirements.txt              # System dependencies
└── README.md                     # Project documentation
```

---

## Setup & Execution
### 1. Environment Installation
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```
### 2. Verify Dataset
Place authentic FloodNet UAV image tiles inside data/floodnet_sample/ and run the inspection script:
```bash
python data/process_real_data.py
```
### 3. Run Non-Agentic Baseline Pipeline
Execute the baseline script to calculate single-pass inference latency across sample UAV tiles:
```bash
python baseline/run_baseline.py
```
