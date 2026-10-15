#!/usr/bin/env python3
"""Imagine's picture engine: builds a ComfyUI workflow and runs it over plain HTTP + websocket.

  engine.py "prompt" [--aspect portrait] [--steps 25] [--seed N] [--turbo] [--quality sketch|quick|preview|fast]
  engine.py "edit instruction" --image in.png [--image ref2.png ...]      (up to 4 input images)
  engine.py status | models | model NAME

Everything model-specific is configuration, nothing is bundled: see README.md ("Configure your image model").
Settings come from the environment, or from KEY=VALUE lines in $IMAGINE_CONFIG (default ~/.config/imagine/engine.env).

  IMAGINE_COMFY_URL       ComfyUI address (default http://127.0.0.1:8188)
  IMAGINE_MODEL           image model file name as ComfyUI lists it
  IMAGINE_TEXT_ENCODER    text encoder file name      IMAGINE_VAE        VAE file name
  IMAGINE_TEXT_ENCODER_TYPE   the "type" value ComfyUI's CLIP loader needs for your model family
  IMAGINE_LORA            optional speed LoRA file    IMAGINE_UPSCALER   optional upscaler model file
  IMAGINE_NODE_LOADER / IMAGINE_NODE_ENCODER / IMAGINE_NODE_LORA / IMAGINE_NODE_SIGMAS
                          ComfyUI node class names for the model loader, prompt encoder, speed LoRA and its step schedule
  IMAGINE_LORA_SIGMAS     the speed schedule, e.g. "1.0, 0.9375, 0.875, 0.75, 0.5, 0.25"
  IMAGINE_MODEL_FILTER    only list model files containing this text

The last stdout line of a command-line run is `MEDIA:<png path>`.
"""
import hashlib
import argparse, json, os, random, sys, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path


def _load_config():
    f = Path(os.environ.get("IMAGINE_CONFIG") or Path.home() / ".config/imagine/engine.env")
    try:
        for line in f.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
    except OSError:
        pass


_load_config()
BASE = os.environ.get("IMAGINE_COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
DATA = Path(os.environ.get("IMAGINE_DATA") or Path.home() / ".local/share/imagine")
OUT_DIR = DATA / "gallery"
NODE_LOADER = os.environ.get("IMAGINE_NODE_LOADER", "")
NODE_ENCODER = os.environ.get("IMAGINE_NODE_ENCODER", "")
NODE_LORA = os.environ.get("IMAGINE_NODE_LORA", "")
NODE_SIGMAS = os.environ.get("IMAGINE_NODE_SIGMAS", "")
TEXT_ENCODER = os.environ.get("IMAGINE_TEXT_ENCODER", "")
VAE = os.environ.get("IMAGINE_VAE", "")
ENCODER_TYPE = os.environ.get("IMAGINE_TEXT_ENCODER_TYPE", "")  # the CLIP loader's "type" setting for your model family
TURBO_LORA = os.environ.get("IMAGINE_LORA", "")  # optional: the speed LoRA
TURBO_NODES = os.environ.get("IMAGINE_LORA_SIGMAS", "1.0, 0.9375, 0.875, 0.75, 0.5, 0.25")  # its step schedule
UPSCALER = os.environ.get("IMAGINE_UPSCALER", "")  # optional: the upscaler model
MODEL_FILTER = os.environ.get("IMAGINE_MODEL_FILTER", "").lower()
HIRES_STEPS, HIRES_DENOISE = 6, 0.35
# Draft sizes (square-equivalent side in the chosen shape, steps).
QUALITY = {"sketch": (384, 8), "quick": (512, 10), "preview": (512, 20), "fast": (768, 20)}
FINISH_MP = 1.25e6        # finish: the preview is enlarged to about this many pixels
FINISH_TEXT = ("Recreate <image1> exactly at a higher resolution with sharper, finer detail (faces, eyes, hair, "
               "skin, fabric, textures). Keep the composition, people, poses, colors and lighting unchanged. ")
ASPECTS = {"square": (1024, 1024), "portrait": (832, 1216), "landscape": (1216, 832),
           "wide": (1344, 768), "tall": (768, 1344)}

DEFAULT_MODEL = os.environ.get("IMAGINE_MODEL", "")
MODEL_FILE = DATA / "image-model.txt"
MODEL_NAMES = {}  # friendly names are not shipped; the file name is shown


def configured():
    """None when the engine is configured, else a message saying what to set."""
    need = {"IMAGINE_MODEL": DEFAULT_MODEL or _saved_model(), "IMAGINE_TEXT_ENCODER": TEXT_ENCODER, "IMAGINE_VAE": VAE,
            "IMAGINE_TEXT_ENCODER_TYPE": ENCODER_TYPE, "IMAGINE_NODE_LOADER": NODE_LOADER, "IMAGINE_NODE_ENCODER": NODE_ENCODER}
    miss = [k for k, v in need.items() if not v]
    return ("Configure your image model first: set " + ", ".join(miss) + " (see README.md, 'Configure your image model').") if miss else None


def _saved_model():
    try:
        return MODEL_FILE.read_text().strip()
    except OSError:
        return ""


def current_model():
    return os.environ.get("IMAGINE_MODEL") or _saved_model() or ""


def available_models():
    """The model files ComfyUI can load right now (through the configured loader node)."""
    if not NODE_LOADER:
        return []
    info = json.loads(api("/object_info/" + NODE_LOADER))[NODE_LOADER]["input"]["required"]
    first = next(iter(info.values()))
    opts = first[1]["options"] if len(first) > 1 and isinstance(first[1], dict) and "options" in first[1] else first[0]
    return [o for o in opts if isinstance(o, str) and MODEL_FILTER in o.lower()]


def model_key(name):
    """Short name: the file name without its folder and extension."""
    return Path(name.replace("\\", "/")).stem.lower()


def set_model(choice):
    """Pick the model by file name or short name; returns the file name."""
    opts = available_models()
    match = [o for o in opts if choice in (o, model_key(o))]
    if not match:
        raise ValueError(f"unknown model '{choice}'. Available: {', '.join(model_key(o) for o in opts)}")
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODEL_FILE.write_text(match[0])
    return match[0]


def api(path, data=None, timeout=15):
    req = urllib.request.Request(BASE + path, data=json.dumps(data).encode() if data else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def up():
    try:
        api("/system_stats", timeout=4)
        return True
    except Exception:
        return False


def start():
    """The engine is not started by Imagine: just check it answers."""
    if not up():
        sys.exit(f"The image engine (ComfyUI) isn't reachable at {BASE}. Start it, or set IMAGINE_COMFY_URL.")


def upload(data, suffix=".png"):
    """Upload image bytes to ComfyUI's input folder; returns its name there."""
    boundary = "----img" + os.urandom(8).hex()
    # named by content: the same picture keeps the same name, so ComfyUI reuses its cached reading of it
    name = f"img_{hashlib.sha256(data).hexdigest()[:20]}{suffix}"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n").encode() + data + \
           f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(BASE + "/upload/image", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["name"]


def workflow(a, w, h, inputs=()):
    wf = {
        "1": {"class_type": NODE_LOADER, "inputs": {"unet_name": getattr(a, "unet", None) or current_model()}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER,
                                                     "type": ENCODER_TYPE, "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "4": {"class_type": NODE_ENCODER, "inputs": {"clip": ["2", 0], "prompt": a.prompt,
                                                                "negative_prompt": a.negative, "resolution": 1024}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["4", 1],
                                                   "latent_image": ["5", 0], "seed": a.seed, "steps": a.steps,
                                                   "cfg": 1, "sampler_name": "euler", "scheduler": "simple",
                                                   "denoise": 1}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "imagine"}},
    }
    if inputs:
        # Edit mode: reference images go into the text encoder (which also needs
        # the VAE), and the encoder's latent (sized from image 1) replaces the
        # empty latent.
        enc = wf["4"]["inputs"]
        enc["vae"] = ["3", 0]
        # edits take image 1's shape at this size: a draft stays draft-sized
        enc["resolution"] = QUALITY[_quality(a)][0] if _quality(a) else 1024
        for i, name in enumerate(inputs, 1):
            wf[f"10{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
            enc[f"images.image_{i}"] = [f"10{i}", 0]
        if not getattr(a, "cast", False):
            # Editing: the output takes image 1's size.
            del wf["5"]
            wf["6"]["inputs"]["latent_image"] = ["4", 2]
        # Cast references (a.cast): the images only guide who/what appears;
        # the output keeps the chosen aspect from the empty latent.
    if getattr(a, "turbo", False):
        if not (TURBO_LORA and NODE_LORA and NODE_SIGMAS):
            raise RuntimeError("The speed LoRA isn't configured: set IMAGINE_LORA, IMAGINE_NODE_LORA and IMAGINE_NODE_SIGMAS, or turn Turbo off.")
        # Speed LoRA: a distilled LoRA applied at runtime (unmerged) + its short step schedule,
        # euler, no CFG (BasicGuider). ~6 passes instead of 20-25. Detail passes below stay on the base model.
        latent = wf["6"]["inputs"]["latent_image"]
        wf["30"] = {"class_type": NODE_LORA, "inputs": {"model": ["1", 0], "lora_name": TURBO_LORA, "strength": 1.0}}
        wf["31"] = {"class_type": NODE_SIGMAS, "inputs": {"latent": latent, "nodes": TURBO_NODES}}
        wf["32"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": a.seed}}
        wf["33"] = {"class_type": "BasicGuider", "inputs": {"model": ["30", 0], "conditioning": ["4", 0]}}
        wf["34"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
        wf["6"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["32", 0], "guider": ["33", 0], "sampler": ["34", 0],
                                                                     "sigmas": ["31", 0], "latent_image": latent}}
    up = getattr(a, "upscale", "") or ""
    if up:
        if not UPSCALER:
            raise RuntimeError("The upscaler isn't configured: set IMAGINE_UPSCALER, or turn Upscale off.")
        # Draw small (fast), then enlarge 2x with the upscaler model (seconds, sharp but adds
        # no new content). "hires" then re-draws the enlarged picture at 1.5x the base
        # size with a short, low-strength pass that adds real fine detail; tiled VAE
        # keeps it inside a small GPU.
        wf["20"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALER}}
        # The VAE may decode 4 channels; the upscaler takes RGB only.
        wf["19"] = {"class_type": "SplitImageWithAlpha", "inputs": {"image": ["7", 0]}}
        wf["21"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["20", 0], "image": ["19", 0]}}
        out = ["21", 0]
        if up == "hires":
            tiles = {"tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8}
            wf["22"] = {"class_type": "ImageScaleBy", "inputs": {"image": ["21", 0], "upscale_method": "lanczos", "scale_by": 0.75}}
            wf["23"] = {"class_type": "VAEEncodeTiled", "inputs": {"pixels": ["22", 0], "vae": ["3", 0], **tiles}}
            wf["24"] = {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["4", 1],
                                                        "latent_image": ["23", 0], "seed": a.seed, "steps": HIRES_STEPS,
                                                        "cfg": 1, "sampler_name": "euler", "scheduler": "simple",
                                                        "denoise": HIRES_DENOISE}}
            wf["25"] = {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["24", 0], "vae": ["3", 0], **tiles}}
            out = ["25", 0]
        wf["8"]["inputs"]["images"] = out
    return wf


def _quality(a):
    """The draft size asked for: a.quality, or the older --preview/--fast flags."""
    q = getattr(a, "quality", None)
    if q in QUALITY:
        return q
    return "preview" if getattr(a, "preview", False) else "fast" if getattr(a, "fast", False) else None


def png_size(b):
    """(width, height) of PNG bytes, or None."""
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        import struct
        return struct.unpack(">II", b[16:24])
    try:  # JPEG and the rest (a Finish from a photo must keep its shape, not fall back to a square)
        import io
        from PIL import Image
        return Image.open(io.BytesIO(b)).size
    except Exception:
        return None


def finish_workflow(a, name, w, h, faces=()):
    """Preview -> full size: 2x upscaler, resize to w x h, then a short low-strength
    redraw guided by the same prompt and by the preview itself as reference, so the
    picture stays the same but gains real detail."""
    if not UPSCALER:
        raise RuntimeError("The upscaler isn't configured: set IMAGINE_UPSCALER (needed by Finish).")
    steps = int(getattr(a, "finish_steps", 0) or HIRES_STEPS)
    denoise = float(getattr(a, "finish_denoise", 0) or HIRES_DENOISE)
    tiles = {"tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8}
    wf = {
        "1": {"class_type": NODE_LOADER, "inputs": {"unet_name": getattr(a, "unet", None) or current_model()}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER,
                                                     "type": ENCODER_TYPE, "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "101": {"class_type": "LoadImage", "inputs": {"image": name}},
        "19": {"class_type": "SplitImageWithAlpha", "inputs": {"image": ["101", 0]}},
        "20": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALER}},
        "21": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["20", 0], "image": ["19", 0]}},
        "22": {"class_type": "ImageScale", "inputs": {"image": ["21", 0], "upscale_method": "lanczos",
                                                      "width": w, "height": h, "crop": "disabled"}},
        "4": {"class_type": NODE_ENCODER, "inputs": {"clip": ["2", 0], "vae": ["3", 0], "resolution": 1024,
                                                                "prompt": FINISH_TEXT + finish_face_text(len(faces)) + a.prompt,
                                                                "negative_prompt": a.negative, "images.image_1": ["19", 0]}},
        "23": {"class_type": "VAEEncodeTiled", "inputs": {"pixels": ["22", 0], "vae": ["3", 0], **tiles}},
        "6": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["4", 1],
                                                   "latent_image": ["23", 0], "seed": a.seed, "steps": steps,
                                                   "cfg": 1, "sampler_name": "euler", "scheduler": "simple",
                                                   "denoise": denoise}},
        "7": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["6", 0], "vae": ["3", 0], **tiles}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "imagine"}},
    }
    for i, face in enumerate(list(faces)[:2], 2):  # 🙂 the original face photos: likeness kept while detail is added
        wf[f"11{i}"] = {"class_type": "LoadImage", "inputs": {"image": face}}
        wf["4"]["inputs"][f"images.image_{i}"] = [f"11{i}", 0]
    if getattr(a, "finish_sharpen", False):
        # one more plain 2x (upscaler) after the redraw: sharper and 2x bigger, no new content
        wf["26"] = {"class_type": "SplitImageWithAlpha", "inputs": {"image": ["7", 0]}}
        wf["27"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["20", 0], "image": ["26", 0]}}
        wf["8"]["inputs"]["images"] = ["27", 0]
    return wf


def finish_face_text(n):
    if not n:
        return ""
    refs = " and ".join(f"<image{i}>" for i in range(2, 2 + min(n, 2)))
    return (f"The face of each person must stay identical to the reference photo{'s' if n > 1 else ''} {refs}: the same identity, "
            "features and proportions; use them only for the face, never their pose, clothes or background. ")


def plan(a, images=()):
    """The picture's size (and a.steps for the draft sizes): the same numbers render() uses, for showing ahead of time."""
    if getattr(a, "finish", False):
        pw, ph = (png_size(images[0]) if images else None) or (768, 768)
        target = float(getattr(a, "finish_mp", 0) or FINISH_MP)
        k = min(4.0, (target / (pw * ph)) ** 0.5)  # to the target size (a sketch goes ~3x: 2x model + resize)
        w, h = round(pw * k), round(ph * k)
    elif _quality(a):
        side, steps = QUALITY[_quality(a)]
        bw, bh = ASPECTS.get(getattr(a, "aspect", None) or "square", (1024, 1024))
        k = (side * side / (bw * bh)) ** 0.5  # same shape, side² pixels
        w, h, a.steps = round(bw * k), round(bh * k), min(a.steps, steps)
    elif getattr(a, "size", None):
        w, h = (int(x) for x in a.size.lower().split("x"))
    else:
        w, h = ASPECTS[a.aspect]
    return w // 16 * 16, h // 16 * 16


def render(a, images=(), log=print):
    """Run one job; returns PNG bytes. Raises RuntimeError on failure.

    With a.temp the result comes back over ComfyUI's websocket
    (SaveImageWebsocket), so the output never touches the engine's disk; uploaded
    inputs and the history entry are deleted afterwards.
    """
    w, h = plan(a, images)
    if callable(getattr(a, "on_plan", None)):  # what is about to be drawn, for the page's task row and the history
        a.on_plan({"w": w, "h": h, "steps": 6 if getattr(a, "turbo", False) and not getattr(a, "finish", False) else a.steps})
    a.seed = a.seed if a.seed is not None else random.randint(0, 2**31)
    miss = configured()
    if miss:
        raise RuntimeError(miss)
    start()
    if not getattr(a, "unet", None):
        # the chosen model may have been removed since: fall back to one that's there
        try:
            have = available_models()
            if have and current_model() not in have:
                a.unet = have[0]
        except Exception:
            pass
    inputs = [upload(b) for b in list(images)[:4]]
    wf = finish_workflow(a, inputs[0], w, h, inputs[1:]) if getattr(a, "finish", False) else workflow(a, w, h, inputs)
    try:
        if a.temp:
            return _render_ws(a, wf, w, h, log)
        return _render_saved(a, wf, w, h, log)
    finally:
        pass


def _submit(wf, client_id=None):
    body = {"prompt": wf}
    if client_id:
        body["client_id"] = client_id
    try:
        return json.loads(api("/prompt", body))["prompt_id"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"rejected: {e.read().decode()[:800]}")


def cancel(pid):
    """Stop one job: drop it if still waiting, interrupt it if it's the one running
    (targeted: never interrupts anyone else's job)."""
    for path, body in (("/queue", {"delete": [pid]}), ("/interrupt", {"prompt_id": pid})):
        try:
            api(path, body)
        except Exception:
            pass


def _submitted(a, pid):
    cb = getattr(a, "on_submit", None)  # lets a caller (the web app) cancel it later
    if cb:
        cb(pid)


def _cancelled(a):
    return bool(getattr(a, "cancelled", lambda: False)())


# What each workflow node is doing, for progress reports.
STAGES = {"1": "loading the model", "2": "loading the model", "3": "loading the model", "20": "loading the upscaler",
          "4": "reading the prompt", "6": "drawing", "7": "decoding", "19": "upscaling", "21": "upscaling",
          "22": "upscaling", "23": "preparing the detail pass", "24": "detail pass", "25": "decoding", "8": "finishing"}


def _progress(a, **info):
    cb = getattr(a, "on_progress", None)  # the web app shows this instead of a guessed timer
    if cb:
        try:
            cb(info)
        except Exception:
            pass


def _follow(a, wf, want_png, log, w, h):
    """Submit wf and follow it over ComfyUI's websocket until it ends, reporting real
    progress (queue wait, stage, sampler step). Returns (prompt id, PNG bytes or None)."""
    import websocket  # websocket-client
    cid = os.urandom(8).hex()
    ws = websocket.create_connection(f"ws://{BASE.split("://",1)[1]}/ws?clientId={cid}", timeout=3600)
    pid, png, node, started = None, None, None, False
    try:
        pid = a._pid = _submit(wf, cid)  # a._pid: lets temp cleanup find it even if this fails
        _submitted(a, pid)
        log(f"queued {pid} {w}x{h} steps={a.steps} seed={a.seed}")
        _progress(a, stage="waiting in the engine's queue")
        ws.settimeout(10)
        while True:
            try:
                msg = ws.recv()
            except websocket.WebSocketTimeoutException:
                if _cancelled(a):  # dropped from ComfyUI's queue before it started
                    raise RuntimeError("stopped")
                continue
            if isinstance(msg, bytes):
                if want_png and node == "8" and len(msg) > 8:
                    png = msg[8:]  # 8-byte event/format header, then the PNG
                continue
            m = json.loads(msg)
            d = m.get("data", {})
            if m["type"] == "status" and not started:
                q = d.get("status", {}).get("exec_info", {}).get("queue_remaining")
                if q:
                    _progress(a, stage="waiting in the engine's queue", ahead=max(0, int(q) - 1))
                continue
            if d.get("prompt_id") not in (None, pid):
                continue
            if m["type"] == "execution_start":
                started = True
                _progress(a, stage="starting")
            elif m["type"] == "executing":
                node = d.get("node")
                if node is None and d.get("prompt_id") == pid:
                    break
                if node:
                    started = True
                    _progress(a, stage=STAGES.get(str(node), "working"))
            elif m["type"] == "progress":
                _progress(a, stage=STAGES.get(str(d.get("node")), "working"), step=d.get("value"), of=d.get("max"))
            elif m["type"] == "execution_error":
                where = f" (in {d.get('node_type')})" if d.get("node_type") else ""
                tb = "".join(d.get("traceback") or [])[-600:]
                if tb:
                    print(f"ComfyUI error{where}:\n{tb}", file=sys.stderr, flush=True)  # the cause, for the logs
                raise RuntimeError(f"generation failed{where}: {d.get('exception_message', '')[:400]}")
            elif m["type"] == "execution_interrupted":
                raise RuntimeError("stopped")
            elif m["type"] == "execution_success":
                break
        return pid, png
    finally:
        ws.close()


def _render_saved(a, wf, w, h, log):
    pid, _ = _follow(a, wf, False, log, w, h)
    hist = json.loads(api(f"/history/{pid}")).get(pid) or {}
    status = hist.get("status", {})
    if status.get("status_str") != "success":
        msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
        raise RuntimeError(f"generation failed: {json.dumps(msgs)[:800]}")
    img = hist["outputs"]["8"]["images"][0]
    q = urllib.parse.urlencode({"filename": img["filename"], "subfolder": img["subfolder"], "type": img["type"]})
    return api(f"/view?{q}", timeout=60)


def _render_ws(a, wf, w, h, log):
    wf["8"] = {"class_type": "SaveImageWebsocket", "inputs": {"images": wf["8"]["inputs"]["images"]}}
    a._pid = None
    try:
        _, png = _follow(a, wf, True, log, w, h)
        if not png:
            raise RuntimeError("no image received")
        return png
    finally:
        pid = a._pid
        if pid:
            try:
                api("/history", {"delete": [pid]})
            except Exception:
                pass


def generate(a):
    t0 = time.time()
    images = [Path(f).read_bytes() for f in a.image]
    try:
        data = render(a, images, log=lambda s: print(s, flush=True))
    except RuntimeError as e:
        sys.exit(str(e))
    if a.out:
        out = Path(a.out)
    else:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"image_{time.strftime('%Y%m%d_%H%M%S')}_{a.seed}.png"
    out.write_bytes(data)
    print(f"done in {time.time() - t0:.0f}s")
    print(f"MEDIA:{out}")


def main():
    if len(sys.argv) in (2, 3) and sys.argv[1] in ("models", "model"):
        if sys.argv[1] == "model" and len(sys.argv) == 3:
            try:
                name = set_model(sys.argv[2])
            except ValueError as e:
                sys.exit(str(e))
            print(f"now using {model_key(name)}")
            return
        cur = current_model()
        for o in available_models():
            print(f"{'*' if o == cur else ' '} {model_key(o):24} {o}")
        return
    if len(sys.argv) == 2 and sys.argv[1] == "status":
        if up():
            q = json.loads(api("/queue"))
            print(f"up; running={len(q['queue_running'])} pending={len(q['queue_pending'])}")
        else:
            print("down")
        return
    p = argparse.ArgumentParser(description="Imagine picture engine (ComfyUI)")
    p.add_argument("prompt")
    p.add_argument("--negative", default="")
    p.add_argument("--size", help="WxH, e.g. 1024x1024")
    p.add_argument("--aspect", choices=ASPECTS, default="square")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--seed", type=int)
    p.add_argument("--fast", action="store_true", help="~0.6 MP in the chosen shape, <=20 steps")
    p.add_argument("--turbo", action="store_true", help="use the speed LoRA (few steps)")
    p.add_argument("--quality", choices=list(QUALITY), help="draft size")
    p.add_argument("--preview", action="store_true", help="small quick draft, ~0.26 MP; finish it with --finish")
    p.add_argument("--finish", action="store_true", help="--image PREVIEW.png -> full size: 2x + detail pass keeping the picture")
    p.add_argument("--upscale", choices=["up", "hires"], help="up: 2x upscaler after drawing; hires: 2x + a short detail pass at 1.5x")
    p.add_argument("--temp", action="store_true", help="leave nothing on the engine: result via websocket, history deleted")
    p.add_argument("--out", help="write the PNG here instead of the gallery folder")
    p.add_argument("--image", action="append", default=[], help="input image to edit (repeatable, max 4)")
    generate(p.parse_args())


if __name__ == "__main__":
    main()
