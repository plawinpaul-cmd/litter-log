"""Which model tells our cats apart on camera? Scores embedders on the crops from bench/nightexport.py (night or
day) by leaving each clip out in turn:
  nn     the crop is named by its closest crop from any OTHER clip (what the app does today)
  probe  a small classifier is trained on the other clips' crops and names the held-out clip's crops
         (what training on our own, scale-labelled clips would give)
The app turns camera naming on at 90% (server.CHECK_MIN_ACC).
Usage: python nightcompare.py <export dir> <model> [<model> ...]
A model is a Hugging Face id or a local folder. Prefixes: timm:<hub id> (e.g. timm:BVRA/MegaDescriptor-L-384,
non-commercial licence) and remote:<hub id> for MiewID (rebuilt here from its weights; e.g. remote:conservationxlabs/miewid-msv3). SigLIP models use their image tower; DINO uses class token + mean patch."""
import json, os, sys
import numpy as np, torch
from PIL import Image

root, models = sys.argv[1], sys.argv[2:]
rows = json.load(open(f"{root}/manifest.json"))
# CPU by default: the live camera model has most of a 6 GB GPU, and a few hundred crops take minutes on a CPU
dev = "cuda" if os.environ.get("NIGHTCMP_GPU") == "1" and torch.cuda.is_available() else "cpu"
imgs = lambda: (Image.open(f"{root}/{r['file']}").convert("RGB") for r in rows)


def miewid(hub_id):
    """MiewID (Conservation X Labs, wildlife re-identification) rebuilt from its weights: its own model code doesn't
    load on transformers 5. Same network: timm efficientnetv2_rw_m, GeM pooling, then BatchNorm1d (2152 features)."""
    import timm
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    class GeM(torch.nn.Module):
        def __init__(self, p=3, eps=1e-6):
            super().__init__()
            self.p, self.eps = torch.nn.Parameter(torch.ones(1) * p), eps

        def forward(self, x):
            return torch.nn.functional.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model("efficientnetv2_rw_m", pretrained=False, num_classes=0)
            self.backbone.global_pool = GeM()
            self.bn = torch.nn.BatchNorm1d(2152)

        def forward(self, x):
            return self.bn(self.backbone(x).view(x.shape[0], -1))

    net = Net()
    state = load_file(hf_hub_download(hub_id, "model.safetensors"))
    missing, _ = net.load_state_dict({k: v for k, v in state.items() if not k.startswith("final.")}, strict=False)
    assert not missing, missing
    return net.eval()


def embed(name):
    out = []
    with torch.inference_mode():
        if name.startswith(("timm:", "remote:")):
            import torchvision.transforms as T
            if name.startswith("timm:"):
                import timm
                model = timm.create_model("hf-hub:" + name[5:], pretrained=True, num_classes=0).eval()
                tf = T.Compose([T.Resize((384, 384)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
            else:
                model = miewid(name[7:])
                tf = T.Compose([T.Resize((440, 440)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            model = model.to(dev)
            for im in imgs():
                e = model(tf(im).unsqueeze(0).to(dev))
                e = e if torch.is_tensor(e) else e[0]          # both return a (1, features) tensor
                out.append(torch.nn.functional.normalize(e.reshape(-1).float(), dim=0).cpu().numpy())
        else:
            from transformers import AutoImageProcessor, AutoModel
            proc = AutoImageProcessor.from_pretrained(name)
            model = AutoModel.from_pretrained(name).to(dev).eval()
            for im in imgs():
                x = proc(images=im, return_tensors="pt").to(dev)
                if hasattr(model, "get_image_features"):      # SigLIP: the same vector the app uses
                    f = model.get_image_features(**x)
                    e = (f if torch.is_tensor(f) else f.pooler_output)[0]   # transformers 5 returns an output object
                else:                                         # DINO: class token + mean of patch tokens
                    h = model(**x).last_hidden_state[0]
                    e = torch.cat([h[0], h[1:].mean(0)])
                out.append(torch.nn.functional.normalize(e.float(), dim=0).cpu().numpy())
    del model
    return np.stack(out)


def probe_predict(Xtr, ytr, Xte, classes, lam=1.0):
    """Ridge classifier (one-vs-rest, closed form): enough to learn which directions separate our cats."""
    Y = np.stack([(ytr == c).astype(float) * 2 - 1 for c in classes], 1)
    W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ Y)
    return [classes[i] for i in (Xte @ W).argmax(1)]


cats = np.array([r["cat"] for r in rows])
clips = np.array([r["clip"] for r in rows])
classes = sorted(set(cats))
for name in models:
    m = embed(name)
    sims = m @ m.T
    nn_right, pr_right, per = 0, 0, {}
    for c in sorted(set(clips)):
        te = clips == c
        tr = ~te
        if len(set(cats[tr])) < len(classes):
            pred = [None] * te.sum()                      # a cat with no other clip can't be learned: counted wrong
        else:
            pred = probe_predict(m[tr], cats[tr], m[te], classes)
        for k, i in enumerate(np.where(te)[0]):
            row = sims[i].copy()
            row[te] = -9
            ok_nn = cats[int(row.argmax())] == cats[i]
            ok_pr = pred[k] == cats[i]
            nn_right += ok_nn
            pr_right += ok_pr
            p = per.setdefault(cats[i], [0, 0, 0])
            p[0] += ok_nn; p[1] += ok_pr; p[2] += 1
    print(json.dumps({"model": name.rstrip("/").split("/")[-1], "crops": len(rows), "clips": len(set(clips)),
                      "nn": round(nn_right / len(rows), 3), "probe": round(pr_right / len(rows), 3),
                      "per_cat_nn_probe": {c: f"{a}/{b} of {n}" for c, (a, b, n) in sorted(per.items())}}), flush=True)
