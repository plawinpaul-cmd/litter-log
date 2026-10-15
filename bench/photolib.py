"""Find your cats in your own photo library and turn them into one-tap "who is this?" groups for the app
(Teach the camera). Three steps, each where it runs best:

  1. manifest (on the Mac with Photos; needs `pip install osxphotos`):
       python photolib.py manifest ~/Pictures/"Photos Library.photoslibrary" out/
     lists the best local copy of every photo (Photos keeps a preview of each, even when the original is only in
     iCloud), skipping screenshots. Then copy the files to the GPU box, e.g.
       tar -cf - -T out/files.txt | ssh gpu 'mkdir -p lib && tar -xf - -C lib'
  2. scan (on the GPU box, in the camera model's venv, plus pillow-heif for HEIC originals):
       python photolib.py scan out/manifest.json lib/ out/
     YOLO26-M finds cats (and dogs) and saves a crop of each cat.
  3. embed (anywhere that can reach the camera model, CATBOX_VISION + its key):
       python photolib.py embed out/ http://<gpu>:8904 <key file>
     runs each crop through the live camera model's /embed in colour and in grey (a stand-in for night vision),
     drops dogs, groups near-identical crops from the same day, and writes out/groups.json + out/c/*.jpg.
Copy out/groups.json and out/c/ to the app's data/photos/. Your photos never leave your machines."""
import base64, io, json, os, sys, urllib.request


def manifest(library, out):
    import osxphotos
    os.makedirs(out, exist_ok=True)
    rows = []
    for p in osxphotos.PhotosDB(dbfile=library).photos(images=True, movies=False):
        if p.screenshot:
            continue
        ders = [x for x in (p.path_derivatives or []) if os.path.exists(x)]
        src = p.path if (p.path and os.path.exists(p.path)) else (max(ders, key=os.path.getsize) if ders else None)
        if src:
            rows.append({"uuid": p.uuid, "date": p.date.isoformat(), "src": src})
    json.dump(rows, open(f"{out}/manifest.json", "w"))
    open(f"{out}/files.txt", "w").write("\n".join(r["src"] for r in rows) + "\n")
    print(len(rows), "photos")


def scan(manifest_file, lib, out):
    import glob
    from PIL import Image, ImageOps
    from ultralytics import YOLO
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass
    weights = (glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Ultralytics--YOLO26/snapshots/*/yolo26m.pt"))
               or ["yolo26m.pt"])[0]
    rows = json.load(open(manifest_file))
    m, crops = YOLO(weights), []
    os.makedirs(f"{out}/crops", exist_ok=True)
    for i in range(0, len(rows), 16):
        batch = [r for r in rows[i:i + 16] if os.path.exists(lib + r["src"])]
        for r, x in zip(batch, m.predict([lib + r["src"] for r in batch], imgsz=1024, conf=0.5, classes=[15],
                                         verbose=False)):
            if not len(x.boxes):
                continue
            im = ImageOps.exif_transpose(Image.open(lib + r["src"])).convert("RGB")
            for b in x.boxes.xyxy.tolist():
                x0, y0, x1, y1 = b
                pw, ph = (x1 - x0) * 0.1, (y1 - y0) * 0.1
                name = f"{len(crops)}.jpg"
                im.crop((max(0, x0 - pw), max(0, y0 - ph), min(im.width, x1 + pw), min(im.height, y1 + ph))).save(
                    f"{out}/crops/{name}", quality=90)
                crops.append({"uuid": r["uuid"], "date": r["date"][:10], "crop": name})
    json.dump(crops, open(f"{out}/crops.json", "w"))
    print(len(crops), "cat crops")


def embed(out, vision, key_file):
    import numpy as np
    from PIL import Image
    key = open(key_file).read().strip()

    def fingerprint(img):
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=92)
        req = urllib.request.Request(f"{vision}/embed", data=buf.getvalue(), method="POST",
                                     headers={"x-catbox-key": key, "x-params": json.dumps({"prompts": []})})
        cats = [c for c in json.load(urllib.request.urlopen(req, timeout=120)).get("cats", []) if c.get("emb")]
        best = max(cats, key=lambda c: c.get("score", 0), default=None)
        return (best["emb"], best.get("species", "cat")) if best else (None, None)

    crops = json.load(open(f"{out}/crops.json"))
    for c in crops:
        im = Image.open(f"{out}/crops/{c['crop']}").convert("RGB")
        c["emb"], c["species"] = fingerprint(im)
        c["emb_gray"], _ = fingerprint(im.convert("L").convert("RGB"))
    crops = [c for c in crops if c["emb"] and c["species"] != "dog"]
    vec = lambda b: (lambda v: v / np.linalg.norm(v))(np.frombuffer(base64.b64decode(b), dtype=np.float16).astype("float32"))
    groups = []
    os.makedirs(f"{out}/c", exist_ok=True)
    for day in sorted({c["date"] for c in crops}):
        cs = [c for c in crops if c["date"] == day]
        v = np.stack([vec(c["emb"]) for c in cs])
        lab = list(range(len(cs)))
        for a in range(len(cs)):          # single link: same day and close fingerprints = the same cat
            for b in range(a + 1, len(cs)):
                if float(v[a] @ v[b]) >= 0.86 and lab[a] != lab[b]:
                    old = lab[b]
                    lab = [lab[a] if x == old else x for x in lab]
        for g in sorted(set(lab)):
            members = [cs[k] for k in range(len(cs)) if lab[k] == g]
            for c in members:
                os.replace(f"{out}/crops/{c['crop']}", f"{out}/c/{c['crop']}")
                c["file"] = f"c/{c.pop('crop')}"
            groups.append({"id": f"{day}-{len(groups)}", "date": day, "crops": members})
    json.dump(groups, open(f"{out}/groups.json", "w"))
    print(len(crops), "cat crops in", len(groups), "groups")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    {"manifest": manifest, "scan": scan, "embed": embed}[cmd](*args)
