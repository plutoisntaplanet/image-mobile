#!/usr/bin/env python3
"""Imagine's measuring tools: deterministic scores for a picture, beside the reviewer's judgment.

A small local HTTP helper (127.0.0.1:8095), run from its own virtual environment
(~/.local/share/imagine-scorers/venv) so torch/insightface stay out of the app's Python.

POST /match {"image": data-URL, "refs": [data-URL...]} -> {"faces": faces found, "matches": [best cosine of any
  face in the picture to the main face of each reference, or null when either has no face]}  (Library auto-label)
POST /score {"image": data-URL, "goal": text, "refs": [data-URL...]} ->
  {"prompt_fit": CLIP cosine of picture and goal (about 0.15-0.35),
   "aesthetic": LAION aesthetic predictor (about 1-10),
   "faces": faces found, "face_match": best cosine of the main face to any reference face (or null),
   "sharpness": high-frequency detail, "seconds": time taken}

Everything stays in memory: nothing is written, logged or cached, and face embeddings are dropped
after each request (they are biometric). Models load lazily on the first request.
"""
import base64, io, json, os, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = int(os.environ.get("IMAGINE_SCORERS_PORT", "8095"))
HOME = Path.home()
WEIGHTS = HOME / ".local/share/imagine-scorers/weights"
AESTHETIC = WEIGHTS / "sac+logos+ava1-l14-linearMSE.pth"
lock = threading.Lock()   # one scoring at a time (CPU); requests queue
models = {}


def load():
    if models:
        return models
    import torch, open_clip
    torch.set_num_threads(max(2, (os.cpu_count() or 4) // 2))
    clip, _, prep = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    clip.eval()
    tok = open_clip.get_tokenizer("ViT-L-14")
    mlp = torch.nn.Sequential(  # LAION improved aesthetic predictor (on normalized CLIP L/14 image embeddings)
        torch.nn.Linear(768, 1024), torch.nn.Dropout(0.2), torch.nn.Linear(1024, 128), torch.nn.Dropout(0.2),
        torch.nn.Linear(128, 64), torch.nn.Dropout(0.1), torch.nn.Linear(64, 16), torch.nn.Linear(16, 1))
    sd = torch.load(AESTHETIC, map_location="cpu")
    mlp.load_state_dict({k.replace("layers.", ""): v for k, v in sd.items()})
    mlp.eval()
    from insightface.app import FaceAnalysis
    faces = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "recognition"])
    faces.prepare(ctx_id=-1, det_size=(640, 640))
    models.update(torch=torch, clip=clip, prep=prep, tok=tok, mlp=mlp, faces=faces)
    return models


def decode(url):
    from PIL import Image
    return Image.open(io.BytesIO(base64.b64decode(str(url).split(",", 1)[-1]))).convert("RGB")


def face_embeddings(m, img):
    import numpy as np
    found = m["faces"].get(np.asarray(img)[:, :, ::-1])  # RGB -> BGR
    found.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    return [f.normed_embedding for f in found]


def sharpness(img):
    from PIL import ImageChops, ImageFilter, ImageStat
    g = img.convert("L")
    g.thumbnail((512, 512))
    return round(ImageStat.Stat(ImageChops.difference(g, g.filter(ImageFilter.GaussianBlur(2)))).rms[0] / 255, 5)


def score(d):
    t0 = time.time()
    m = load()
    torch = m["torch"]
    img = decode(d["image"])
    out = {"sharpness": sharpness(img)}
    with torch.no_grad():
        e = m["clip"].encode_image(m["prep"](img).unsqueeze(0))
        e = e / e.norm(dim=-1, keepdim=True)
        out["aesthetic"] = round(float(m["mlp"](e.float())[0, 0]), 3)
        goal = str(d.get("goal", "")).strip()[:1000]
        if goal:
            t = m["clip"].encode_text(m["tok"]([goal]))
            t = t / t.norm(dim=-1, keepdim=True)
            out["prompt_fit"] = round(float((e @ t.T)[0, 0]), 4)
    faces = face_embeddings(m, img)
    out["faces"] = len(faces)
    refs = [x for r in (d.get("refs") or [])[:4] for x in face_embeddings(m, decode(r))[:1]]
    out["face_match"] = round(float(max(float(faces[0] @ r) for r in refs)), 4) if faces and refs else None
    del faces, refs, img
    out["seconds"] = round(time.time() - t0, 2)
    return out


def face_matches(d):
    """For each reference: the best cosine between its main face and any face in the picture (a group photo counts)."""
    m = load()
    faces = face_embeddings(m, decode(d["image"]))
    out = []
    for r in (d.get("refs") or [])[:16]:
        ref = face_embeddings(m, decode(r))[:1]
        out.append(round(float(max(float(f @ ref[0]) for f in faces)), 4) if faces and ref else None)
    n = len(faces)
    del faces
    return {"faces": n, "matches": out}


def face_boxes(d):
    """Face boxes (x0, y0, x1, y1 in pixels, largest first) so the app can crop full-resolution close-ups."""
    m = load()
    import numpy as np
    img = decode(d["image"])
    found = m["faces"].get(np.asarray(img)[:, :, ::-1])
    boxes = sorted(([round(float(v)) for v in f.bbox] for f in found), key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return {"boxes": boxes[:8], "size": list(img.size)}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # no logs: requests carry private pictures
        pass

    def send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            return self.send(200, {"ok": True, "loaded": bool(models)})
        self.send(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/score", "/faces", "/match"):
            return self.send(404, {"error": "not found"})
        try:
            d = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            with lock:
                return self.send(200, score(d) if self.path == "/score" else face_matches(d) if self.path == "/match" else face_boxes(d))
        except Exception as e:
            return self.send(500, {"error": type(e).__name__})  # no message: it could echo input


if __name__ == "__main__":
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.umask(0o077)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
