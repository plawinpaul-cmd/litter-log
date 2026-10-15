"""Litter Log camera-model benchmark, one tier per run (a fresh process, so peak RAM is that tier's alone).
   python bench/bench.py --clip some-clip.mp4 [--det yolo26s.pt --imgsz 1280 --emb google/siglip2-base-patch16-224]
                         [--device auto|cpu] [--fps 2]
Prints one JSON line: load, first and warm run times, peak RAM, GPU memory, frames with an animal, tracks."""
import argparse, json, os, platform, resource, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.environ.setdefault("CATVISION_HOME", HERE)
os.environ.setdefault("YOLO_OFFLINE", "1")
if not os.path.exists(os.path.join(os.environ["CATVISION_HOME"], "key")):   # vision.py wants a key file; unused here
    open(os.path.join(os.environ["CATVISION_HOME"], "key"), "w").write("bench\n")
sys.path.insert(0, os.path.join(ROOT, "vision"))
ap = argparse.ArgumentParser()
ap.add_argument("--det", default="yolo26x.pt"); ap.add_argument("--imgsz", type=int, default=1920)
ap.add_argument("--emb", default="google/siglip2-so400m-patch16-384"); ap.add_argument("--device", default="auto")
ap.add_argument("--fps", type=float, default=2); ap.add_argument("--clip", required=True)
a = ap.parse_args()
import torch
import vision as V                         # import first: transformers checks torch features at import time
if a.device == "cpu":                      # then hide the GPU from the worker's device pick (it runs at load)
    torch.cuda.is_available = lambda: False
    torch.backends.mps.is_available = lambda: False
V.DET_FILE, V.DET_IMGSZ, V.EMB_ID = a.det, a.imgsz, a.emb
cfg = os.path.join(ROOT, "cats.json")
looks = {k: v["look"] for k, v in json.load(open(cfg)).items() if v.get("look")} if os.path.exists(cfg) else {"cat": "a photo of a cat"}
names, prompts = list(looks), list(looks.values())
mp4 = open(a.clip, "rb").read()
t0 = time.time(); imgs = V.decode(mp4, a.fps); t1 = time.time()
V.M.load(); t2 = time.time()
r1 = V.run(imgs, a.fps, prompts); t3 = time.time()
r2 = V.run(imgs, a.fps, prompts); t4 = time.time()
ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak = ru / 1e9 if platform.system() == "Darwin" else ru / 1e6
gpu = torch.cuda.max_memory_allocated() / 1e9 if V.M.dev == "cuda" else (torch.mps.driver_allocated_memory() / 1e9 if V.M.dev == "mps" else 0)
picks = [names[max(range(len(names)), key=lambda i: c["text"][i])] for c in r2["cats"] if c.get("text")]
print(json.dumps({"host": platform.node(), "device": V.M.dev, "emb_device": V.M.emb_dev, "det": a.det, "imgsz": a.imgsz, "emb": a.emb.split("/")[-1], "fps": a.fps,
                  "frames": len(imgs), "load_s": round(t2 - t1, 1), "first_s": round(t3 - t2, 1), "warm_s": round(t4 - t3, 1),
                  "per_frame_ms": round((t4 - t3) / max(1, len(imgs)) * 1000), "peak_ram_gb": round(peak, 2), "gpu_gb": round(gpu, 2),
                  "cat_frames": r2["cat_frames"], "cats": len(r2["cats"]), "picks": picks}))
