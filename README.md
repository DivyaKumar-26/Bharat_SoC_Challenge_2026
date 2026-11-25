# Real-time Industrial Defect Detection & Quality-Control Assistant
**Bharat AI-SoC Challenge 2026–27 · Problem Statement 5 (Manufacturing & Industry 4.0) · Phase 1 submission**

Team: Divya Kumar, Garvit Bansal, Rajat Panchal · Faculty Mentor: Prof. Bishnu Prasad Das

An edge quality-control assistant that inspects manufactured parts, finds and localises surface defects, rates their severity, gives the operator a recommended action, and logs every inspection, with no cloud dependence.

---

## Why this design
- **Defects are rare and varied.** A line produces thousands of good parts and only a few examples of each defect, and new defect types keep appearing. So we model what a *good* part looks like and flag anything that deviates, instead of training a classifier that only knows defects it has already seen.
- **The LLM must never slow the line.** A small LLM on an Arm CPU needs seconds per answer, while inspection must keep pace with the conveyor. The advisor therefore runs *alongside* the vision loop, not inside it.

## Architecture (three separate pipelines)
| Pipeline | What it does | Runs on |
|---|---|---|
| **Vision** | Anomaly detection, heatmap localisation, bounding box, severity | CPU (Phase 2: Hailo NPU via HailoRT) |
| **Advisor** | Turns a defect event + recent history + SOP into ≤3 operator actions | Arm CPU via ExecuTorch (Llama 3.2 1B Instruct) |
| **Logging** | Stores every inspection | SQLite (offline, serverless) |

```
camera / images ──► ResNet18 patch features ──► kNN vs memory bank of GOOD patches
                                                   │
                         anomaly heatmap ◄─────────┘
                               │
             threshold ──► OK ─────────────────────────────► SQLite log
                    └──► DEFECT ─► bbox + severity ─┬──────► SQLite log
                                                    └─► queue ─► Llama (ExecuTorch) ─► advice ─► log
                                                              (rule-based SOP fallback if busy/unavailable)
```

### Vision
ImageNet-pretrained ResNet18 used as a frozen feature extractor (no network training). Layer-2 and upsampled layer-3 features are concatenated into a 384-dim descriptor per cell of a 28×28 grid and averaged over 3×3 neighbourhoods (PatchCore-style [2]). Descriptors from defect-free images form a memory bank; at inference each patch's score is its distance to the nearest normal patch. The image score is the maximum patch score; the smoothed score map is the heatmap.
**Threshold:** 99th percentile of scores on 20% of good training images **held out** from the memory bank.
**Severity:** minor / major / critical from affected-area fraction and score-to-threshold ratio.

### Advisor
Each defect event (type, severity, score ratio, location, last-hour statistics, SOP text) is formatted as a Llama 3 chat prompt for **Llama 3.2 1B Instruct** run through ExecuTorch's `llama_main`. The system prompt restricts answers to the SOP and given data, max three actions, and escalation when information is insufficient. Runs in a background thread with a bounded queue; a rule-based SOP fallback keeps the operator informed if the LLM is busy or unavailable.
**Phase 1 status:** demo uses the rule-based fallback; on-device ExecuTorch Llama is the first Phase 2 task.

### Logging
SQLite table `inspections`: timestamp, verdict, score, defect type, severity, bounding box, latency, advice. The log also feeds the advisor so it can spot repeating defects (often an upstream machine fault).

---

## Results (Phase 1)
Dataset: **MVTec AD — bottle** [1], official test split (good, broken_large, broken_small, contamination).

| Metric | Value |
|---|---|
| Test images | 83 |
| Image-level AUROC | **1.0000** |
| Recall at threshold | 0.873 |
| False-positive rate at threshold | 0.000 |
| Median latency per image | 273.7 ms (≈3.7 images/s) |
| Memory bank | 20,000 × 384 |
| Threshold | 2.713 (from 41 held-out good images) |

**Reading the numbers.** AUROC 1.0 means every defective bottle scored higher than every good one. Bottle is one of the easier MVTec categories, and there is no overlap between fitting and test data (calibration images held out from the bank; test split disjoint from training). The threshold is conservative: zero false alarms but ~1 in 8 defects missed. It will be tuned on a validation set with defects in Phase 2. No labelled defect examples were supplied in this run, so defect type is reported as `unknown`.

**Hardware:** Intel Core i5-12450H laptop, 16 GB RAM, Windows, CPU-only PyTorch, batch size 1. *These are development-machine numbers, not target-hardware numbers.* Raw outputs: `fit.txt`, `results.txt`.

---

## Quick start
```bash
pip install -r requirements.txt
python download_data.py                                                  # MVTec AD bottle → mvtec/bottle
python qc_assistant.py fit  --normal mvtec/bottle/train/good --out bank.pt
python qc_assistant.py eval --bank bank.pt --test mvtec/bottle/test      # AUROC, recall, FPR, latency
python qc_assistant.py run  --bank bank.pt --source mvtec/bottle/test    # live demo (or --source 0 for webcam)
```
Windows one-click: `run_all.bat`.

Optional — with an ExecuTorch Llama build:
```bash
python qc_assistant.py run --bank bank.pt --source 0 \
  --llama_main <path>/llama_main --pte llama3_2_1b.pte --tokenizer tokenizer.model
```

## Repository contents
| File | Purpose |
|---|---|
| `qc_assistant.py` | Full pipeline: `fit`, `eval`, `run` (vision, advisor, logging) |
| `download_data.py` | Downloads only MVTec AD bottle from the HF mirror |
| `run_all.bat` | Windows end-to-end script |
| `requirements.txt` | Python dependencies |
| `fit.txt`, `results.txt` | Measured Phase 1 outputs |
| `PS5_Phase1_Report.docx` | Technical report |

## Optimizations
**Done:** memory bank capped at 20k descriptors (fast kNN, small footprint); single forward pass for multi-scale features; no network training (new product = new good images only); non-blocking advisor with bounded queue + fallback; serverless SQLite logging.
**Phase 2:** greedy coreset subsampling; feature extractor on Hailo NPU; 4-bit Llama 3.2 1B via ExecuTorch kept resident in memory; EfficientAD comparison [3]; on-device latency, tokens/s, memory and power.

## Limitations
One MVTec category so far · laptop measurements only · LLM advisor integrated but not yet run on-device · SOP texts and severity thresholds are placeholders for process engineers to set · defect typing needs labelled examples.

## Phase 2 roadmap
1. Deploy on Raspberry Pi 5 (± AI HAT+) / Arm Android with camera
2. Export and run Llama 3.2 1B through ExecuTorch on the Arm CPU
3. Hailo NPU for the vision backbone (separate pipeline)
4. More MVTec / NEU / KolektorSDD categories; threshold tuning with defect validation data
5. Full on-device benchmark report

## Data & references
MVTec AD is CC BY-NC-SA 4.0 (non-commercial); images are **not** included in this repo.
1. P. Bergmann, M. Fauser, D. Sattlegger, C. Steger. *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.* CVPR 2019.
2. K. Roth et al. *Towards Total Recall in Industrial Anomaly Detection.* CVPR 2022.
3. K. Batzner, L. Heckler, R. König. *EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies.* WACV 2024.
