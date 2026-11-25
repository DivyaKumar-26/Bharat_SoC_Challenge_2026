# PS5 — Real-time Industrial Defect Detection & QC Assistant
Bharat AI-SoC Challenge 2026–27, Phase 1 submission.

## Architecture (separate pipelines)
1. **Vision** — ResNet18 (ImageNet) layer2+layer3 patch features → k-nearest-neighbour distance to a memory bank of defect-free patches (PatchCore-style, Roth et al., CVPR 2022) → anomaly heatmap → bounding box → defect type (nearest class prototype) → severity rules.
2. **Advisor** — defect event + last-hour log statistics + SOP text → Llama 3.2 1B Instruct via ExecuTorch `llama_main` on the Arm CPU, asynchronous (never blocks inspection). Rule-based fallback when the runner is not available.
3. **Logging** — every inspection → SQLite (`inspections.db`).

## Run
```
pip install -r requirements.txt
python download_data.py                       # MVTec AD bottle from HF mirror foersben/mvtec-ad
python qc_assistant.py fit  --normal mvtec/bottle/train/good --out bank.pt
python qc_assistant.py eval --bank bank.pt --test mvtec/bottle/test
python qc_assistant.py run  --bank bank.pt --source mvtec/bottle/test   # or --source 0 for webcam
```
Windows: `run_all.bat`. Linux/macOS: `./run_all.sh`.

## Results (Phase 1)
See `results.txt` (measured on the development machine, not yet on the target Arm board).

## Phase 2 plan
Raspberry Pi 5 / Arm Android deployment; backbone on Hailo NPU via HailoRT; Llama 3.2 1B 4-bit ExecuTorch export with a persistent runner; EfficientAD comparison; on-device latency, tokens/s and power benchmarks.

## Data & citation
MVTec AD (CC BY-NC-SA 4.0, non-commercial; images not included in this repo).
Bergmann et al., "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection," CVPR 2019.
