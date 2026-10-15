"""Export the camera crops of known cats (cat, clip, image) as a tar.gz on stdout, for bench/nightcompare.py on the
GPU box. Run from the app folder: .venv/bin/python bench/nightexport.py [night|day] > crops.tgz"""
import io, json, sqlite3, sys, tarfile, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import server  # noqa: E402

db = sqlite3.connect(server.DATA / "catbox.db")
db.row_factory = sqlite3.Row
mode = (sys.argv[1:] or ["night"])[0]
refs = server.camera_reference_crops(db, int(time.time()), ir=mode == "night")
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    rows = []
    for i, r in enumerate(refs):
        src = server.MEDIA / r["url"].removeprefix("/media/")
        if src.exists():
            tar.add(src, arcname=f"crops/{i}.jpg")
            rows.append({**r, "file": f"crops/{i}.jpg"})
    data = json.dumps(rows).encode()
    info = tarfile.TarInfo("manifest.json"); info.size = len(data)
    tar.addfile(info, io.BytesIO(data))
sys.stdout.buffer.write(buf.getvalue())
print(f"{len(rows)} {mode} crops", {c: len({r['clip'] for r in rows if r['cat'] == c}) for c in {r['cat'] for r in rows}},
      "clips per cat", file=sys.stderr)
