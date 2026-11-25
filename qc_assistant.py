#!/usr/bin/env python3
"""
qc_assistant.py - PS5 baseline: edge visual inspection + on-device Llama advisor.

Pipeline (two separate pipelines, as the problem statement asks you to document):
  VISION  : camera -> ResNet18 patch features -> kNN to memory bank of NORMAL patches
            (PatchCore-style, Roth et al. CVPR 2022) -> anomaly map -> bbox -> defect type
            (nearest class prototype) -> severity rules.   [later: backbone -> Hailo NPU]
  ADVISOR : defect event (JSON) + recent log history + SOP text -> Llama 3.2 1B Instruct
            via ExecuTorch `llama_main` on the Arm CPU, ASYNC (never blocks the vision loop).
  LOG     : every inspection -> SQLite; advisor text attached to the event row when ready.

Usage
  pip install torch torchvision opencv-python scikit-learn
  # 1) fit on defect-free images (e.g. MVTec AD <category>/train/good)
  python qc_assistant.py fit  --normal mvtec/bottle/train/good --defects my_labelled_defects/ --out bank.pt
  # 2) evaluate image-level AUROC + latency (folder with good/ and one subfolder per defect type)
  python qc_assistant.py eval --bank bank.pt --test mvtec/bottle/test
  # 3) run live (camera index) or on a folder; advisor falls back to rules if llama_main absent
  python qc_assistant.py run  --bank bank.pt --source 0 \
      --llama_main ~/executorch/cmake-out/examples/models/llama/llama_main \
      --pte llama3_2_1b_8da4w.pte --tokenizer tokenizer.model
"""
import argparse, json, queue, sqlite3, subprocess, threading, time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

IMG = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

# Placeholder SOPs - REPLACE with real ones from your domain (keeps the LLM grounded).
SOP = {
    "scratches": "Check guide rollers and conveyor contact points; inspect last 20 parts.",
    "inclusion": "Quarantine lot; notify melt-shop QA; sample adjacent coils.",
    "crack": "Stop line if repeated; check press force and die wear.",
    "unknown": "Hold part for manual inspection; capture 3 more images for labelling.",
}
CRITICAL = {"crack", "inclusion"}  # placeholder; agree with process engineers


def images(d):
    return sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in EXTS)


def to_tensor(bgr):
    rgb = cv2.cvtColor(cv2.resize(bgr, (IMG, IMG)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    return torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1)).unsqueeze(0)


class Backbone(torch.nn.Module):
    """ResNet18 layer2+layer3 features, 3x3 neighbourhood-averaged (PatchCore-style)."""
    def __init__(self):
        super().__init__()
        m = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
        self.stem = torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1)
        self.l2, self.l3 = m.layer2, m.layer3

    @torch.no_grad()
    def forward(self, x):
        f2 = self.l2(self.stem(x))
        f3 = F.interpolate(self.l3(f2), size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f = F.avg_pool2d(torch.cat([f2, f3], 1), 3, 1, 1)  # (1, 384, 28, 28)
        return f, F.normalize(f.mean((2, 3)), dim=1)[0]     # patch feats, global embedding


def patches(f):
    return f[0].permute(1, 2, 0).reshape(-1, f.shape[1])


def score(bb, bank, bgr):
    f, _ = bb(to_tensor(bgr))
    h, w = f.shape[-2:]
    d = torch.cdist(patches(f), bank).min(1).values.reshape(h, w).numpy()
    amap = cv2.GaussianBlur(cv2.resize(d, (bgr.shape[1], bgr.shape[0])), (0, 0), 4)
    return amap, float(d.max())


def classify(bb, protos, names, crop):
    if not names or crop.size == 0:
        return "unknown", 0.0
    _, g = bb(to_tensor(crop))
    sims = protos @ g
    i = int(sims.argmax())
    return names[i], float(sims[i])


def severity(cls, area_frac, peak_ratio):
    # Placeholder rules. Real thresholds come from the customer's acceptance spec.
    if cls in CRITICAL or area_frac > 0.05 or peak_ratio > 2.0:
        return "critical"
    if area_frac > 0.01 or peak_ratio > 1.5:
        return "major"
    return "minor"


# ---------------------------------------------------------------- fit / eval
def fit(a):
    torch.manual_seed(0)
    bb = Backbone()
    paths = images(a.normal)
    perm = torch.randperm(len(paths)).tolist()
    n_cal = max(1, int(0.2 * len(paths)))           # held-out normals for threshold
    cal, train = [paths[i] for i in perm[:n_cal]], [paths[i] for i in perm[n_cal:]]
    bank = torch.cat([patches(bb(to_tensor(cv2.imread(str(p))))[0]) for p in train])
    if len(bank) > a.bank_size:                       # random subsample (PatchCore uses greedy coreset)
        bank = bank[torch.randperm(len(bank))[: a.bank_size]]
    s = np.array([score(bb, bank, cv2.imread(str(p)))[1] for p in cal])
    thr = float(np.quantile(s, a.q))                  # few cal images -> noisy; report it
    names, protos = [], []
    if a.defects:
        for d in sorted(p for p in Path(a.defects).iterdir() if p.is_dir()):
            embs = [bb(to_tensor(cv2.imread(str(p))))[1] for p in images(d)]
            if embs:
                names.append(d.name)
                protos.append(F.normalize(torch.stack(embs).mean(0), dim=0))
    torch.save({"bank": bank, "thr": thr, "names": names,
                "protos": torch.stack(protos) if protos else torch.empty(0)}, a.out)
    print(f"bank {tuple(bank.shape)}  thr={thr:.3f} (q={a.q}, n_cal={len(cal)})  classes={names}")


def evaluate(a):
    from sklearn.metrics import roc_auc_score
    ck, bb = torch.load(a.bank), Backbone()
    y, s, lat = [], [], []
    for p in images(a.test):
        t0 = time.perf_counter()
        s.append(score(bb, ck["bank"], cv2.imread(str(p)))[1])
        lat.append(time.perf_counter() - t0)
        y.append(0 if p.parent.name == "good" else 1)
    pred = np.array(s) > ck["thr"]
    y = np.array(y)
    print(f"image AUROC={roc_auc_score(y, s):.4f}  n={len(y)}  "
          f"recall@thr={pred[y == 1].mean():.3f}  FPR@thr={pred[y == 0].mean():.3f}  "
          f"latency median={1e3 * np.median(lat):.1f} ms (this machine, CPU, batch 1)")


# ---------------------------------------------------------------- logging + advisor
class Log:
    def __init__(self, path):
        self.db, self.lock = sqlite3.connect(path, check_same_thread=False), threading.Lock()
        self.db.execute("""CREATE TABLE IF NOT EXISTS inspections(id INTEGER PRIMARY KEY,
            ts REAL, verdict TEXT, score REAL, cls TEXT, severity TEXT, bbox TEXT,
            latency_ms REAL, advice TEXT)""")

    def add(self, **r):
        with self.lock:
            c = self.db.execute("INSERT INTO inspections(ts,verdict,score,cls,severity,bbox,latency_ms)"
                                " VALUES(?,?,?,?,?,?,?)", (r["ts"], r["verdict"], r["score"], r["cls"],
                                r["severity"], json.dumps(r["bbox"]), r["latency_ms"]))
            self.db.commit()
            return c.lastrowid

    def advise(self, rid, text):
        with self.lock:
            self.db.execute("UPDATE inspections SET advice=? WHERE id=?", (text, rid))
            self.db.commit()

    def history(self, window_s=3600):
        with self.lock:
            rows = self.db.execute("SELECT cls, severity, COUNT(*) FROM inspections WHERE ts>? AND "
                                   "verdict='DEFECT' GROUP BY cls, severity", (time.time() - window_s,))
            n = self.db.execute("SELECT COUNT(*) FROM inspections WHERE ts>?", (time.time() - window_s,))
            return {"defects_last_hour": [list(r) for r in rows], "parts_last_hour": n.fetchone()[0]}


def llama3_prompt(system, user):
    # Llama 3 chat format. Check llama_main does not prepend a second BOS for your tokenizer.
    return (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n")


class Advisor(threading.Thread):
    SYSTEM = ("You are a quality-control assistant on a factory floor. Use ONLY the SOP text and "
              "the data given. Reply in at most 3 short numbered actions. If data is insufficient, "
              "say 'escalate to QC engineer'.")

    def __init__(self, a, log):
        super().__init__(daemon=True)
        self.a, self.log, self.q = a, log, queue.Queue(maxsize=8)
        self.ok = bool(a.llama_main) and Path(a.llama_main).exists()

    def submit(self, rid, ev):
        try:
            self.q.put_nowait((rid, ev))
        except queue.Full:                      # never stall the line for the LLM
            self.log.advise(rid, "advisor busy - " + SOP.get(ev["cls"], SOP["unknown"]))

    def ask(self, ev):
        sop = SOP.get(ev["cls"], SOP["unknown"])
        if not self.ok:
            return f"[rule-based] {ev['severity'].upper()} {ev['cls']}: {sop}"
        user = json.dumps({"event": ev, "recent": self.log.history(), "sop": sop})
        prompt = llama3_prompt(self.SYSTEM, user)
        # NOTE: one process per call reloads the .pte each time (seconds). For the final demo
        # keep the model resident (ExecuTorch LLM runner bindings or a small C++ server).
        out = subprocess.run([self.a.llama_main, f"--model_path={self.a.pte}",
                              f"--tokenizer_path={self.a.tokenizer}", f"--prompt={prompt}",
                              f"--seq_len={self.a.seq_len}", f"--cpu_threads={self.a.cpu_threads}"],
                             capture_output=True, text=True, timeout=180)
        text = out.stdout.split("assistant<|end_header_id|>")[-1]
        return text.replace("<|eot_id|>", "").strip() or f"[LLM empty] {sop}"

    def run(self):
        while True:
            rid, ev = self.q.get()
            t0 = time.perf_counter()
            try:
                text = self.ask(ev)
            except Exception as e:
                text = f"[advisor error: {e}] {SOP.get(ev['cls'], SOP['unknown'])}"
            self.log.advise(rid, text)
            print(f"\n[ADVICE #{rid} {time.perf_counter() - t0:.1f}s] {text}\n")


# ---------------------------------------------------------------- live loop
def frames(src):
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
        while True:
            ok, f = cap.read()
            if not ok:
                break
            yield f
    else:
        for p in images(src):
            yield cv2.imread(str(p))


def run(a):
    torch.set_num_threads(a.cpu_threads)
    ck, bb, log = torch.load(a.bank), Backbone(), Log(a.db)
    adv = Advisor(a, log)
    adv.start()
    for frame in frames(a.source):
        t0 = time.perf_counter()
        amap, s = score(bb, ck["bank"], frame)
        verdict, cls, sev, bbox, conf = "OK", None, None, None, 0.0
        if s > ck["thr"]:
            verdict = "DEFECT"
            mask = (amap > ck["thr"]).astype(np.uint8)
            x, y, w, h = cv2.boundingRect(mask)
            bbox = [x, y, w, h]
            cls, conf = classify(bb, ck["protos"], ck["names"], frame[y:y + h, x:x + w])
            sev = severity(cls, mask.mean(), s / ck["thr"])
        ms = 1e3 * (time.perf_counter() - t0)
        rid = log.add(ts=time.time(), verdict=verdict, score=s, cls=cls, severity=sev, bbox=bbox, latency_ms=ms)
        if verdict == "DEFECT":
            adv.submit(rid, {"id": rid, "cls": cls, "class_sim": round(conf, 3), "severity": sev,
                             "score_ratio": round(s / ck["thr"], 2), "bbox": bbox})
        print(f"#{rid} {verdict:6s} score={s:.3f} {cls or ''} {sev or ''} {ms:.0f} ms", end="\r")
        if not a.headless:
            heat = cv2.applyColorMap(np.clip(255 * amap / (2 * ck["thr"]), 0, 255).astype(np.uint8),
                                     cv2.COLORMAP_JET)
            vis = cv2.addWeighted(frame, 0.6, heat, 0.4, 0)
            if bbox:
                cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 2)
            cv2.putText(vis, f"{verdict} {cls or ''} {sev or ''} {ms:.0f}ms", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("QC", vis)
            if cv2.waitKey(1 if a.source.isdigit() else 300) & 0xFF == ord("q"):
                break
    time.sleep(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fit")
    f.add_argument("--normal", required=True); f.add_argument("--defects")
    f.add_argument("--out", default="bank.pt"); f.add_argument("--bank_size", type=int, default=20000)
    f.add_argument("--q", type=float, default=0.99)
    e = sp.add_parser("eval")
    e.add_argument("--bank", required=True); e.add_argument("--test", required=True)
    r = sp.add_parser("run")
    r.add_argument("--bank", required=True); r.add_argument("--source", default="0")
    r.add_argument("--db", default="inspections.db"); r.add_argument("--headless", action="store_true")
    r.add_argument("--llama_main"); r.add_argument("--pte"); r.add_argument("--tokenizer")
    r.add_argument("--seq_len", type=int, default=768); r.add_argument("--cpu_threads", type=int, default=4)
    a = ap.parse_args()
    {"fit": fit, "eval": evaluate, "run": run}[a.cmd](a)
