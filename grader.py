#!/usr/bin/env python3
"""🌙 Background grading for one incognito session, with the page closed.

Started by server.py (/api/grader/start) with the session's password-derived wrap key on stdin (the password itself
never leaves the browser). It holds the session key only in its own memory: locked against swap (mlockall), not
dumpable (no core, no ptrace, no /proc/<pid>/mem reads by other processes), and gone when the process ends.

It reads the session's pictures, grades each ungraded one through the server (which waits until the text model is
free), and writes the grade sealed, exactly as the page does. Then it looks for new pictures every few minutes.
It ends on Stop, when the session is deleted, when the server goes away (stdin closes) or after MAX_HOURS.

Protocol (JSON lines): stdin <- {"wrap": b64} then one {"g": grade} / {"error": text} per request;
stdout -> {"grade": {"image": data-URL, "request": text}} and {"status": {...}} (counts only).
"""
import base64, ctypes, hashlib, io, json, os, resource, sys, time
from pathlib import Path

MAX_HOURS = 24
RESCAN = 300  # seconds between looks for new pictures
NOT_PICTURES = {"story", "cast", "asset", "request", "lesson", "meditation", "resume", "medmark", "character",
                "note", "grade", "qgrade"}
HKDF_INFO = b"imagine-v1"


def harden():
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.prctl(4, 0, 0, 0, 0)  # PR_SET_DUMPABLE = 0: no core, no ptrace, no /proc/<pid>/mem for other processes
    libc.prctl(1, 9, 0, 0, 0)  # PR_SET_PDEATHSIG = SIGKILL: ends with the server, key and all
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if libc.mlockall(1 | 2) != 0:  # MCL_CURRENT | MCL_FUTURE: never swapped to disk
        raise SystemExit("could not lock memory (mlockall)")


b64d = lambda s: base64.b64decode(s)
b64e = lambda b: base64.b64encode(b).decode()


def say(obj):
    sys.stdout.write(json.dumps(obj) + "\n"); sys.stdout.flush()


def main():
    harden()
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    vault_dir = Path(sys.argv[1])
    items = vault_dir / "items"
    wrap = b64d(json.loads(sys.stdin.readline())["wrap"])
    auth = json.loads((vault_dir / "auth.json").read_text())
    priv = serialization.load_der_private_key(AESGCM(wrap).decrypt(b64d(auth["wrap_iv"]), b64d(auth["wrapped"]), None), None)
    del wrap
    pub = priv.public_key()

    def hkdf(shared):
        return HKDF(hashes.SHA256(), 32, salt=b"\0" * 32, info=HKDF_INFO).derive(shared)

    def unseal(box):
        epk = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), b64d(box["epk"]))
        return json.loads(AESGCM(hkdf(priv.exchange(ec.ECDH(), epk))).decrypt(b64d(box["iv"]), b64d(box["ct"]), None))

    def seal(obj):
        eph = ec.generate_private_key(ec.SECP256R1())
        iv = os.urandom(12)
        return {"epk": b64e(eph.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)),
                "iv": b64e(iv), "ct": b64e(AESGCM(hkdf(eph.exchange(ec.ECDH(), pub))).encrypt(iv, json.dumps(obj).encode(), None))}

    def write(jid, name, obj):
        f = items / f"{jid}.{name}.json"
        if not items.is_dir():
            return
        tmp = f.with_suffix(".tmp"); tmp.write_text(json.dumps(obj)); tmp.replace(f)

    def read(jid, name):
        try:
            return unseal(json.loads((items / f"{jid}.{name}.json").read_text()))
        except Exception:
            return None

    metas, provs = {}, {}  # decrypted once: id -> (mtime, record)

    def cached(store, jid, name):
        f = items / f"{jid}.{name}.json"
        try:
            m = f.stat().st_mtime
        except OSError:
            return None
        if jid not in store or store[jid][0] != m:
            r = read(jid, name)  # keep only what grading needs: never a Library photo or a picture
            store[jid] = (m, r and {"kind": r.get("kind"), "sha256": r.get("sha256"), "picture": r.get("picture"), "prompt": r.get("prompt"),
                                    "meditation": {"goal": (r.get("meditation") or {}).get("goal")},
                                    "inputs": [{"sha256": x.get("sha256")} for x in r.get("inputs") or [] if isinstance(x, dict)]})
            del r
        return store[jid][1]

    def root_prompt(jid, by_sha):
        cur, seen = jid, set()
        while cur and cur not in seen:
            seen.add(cur)
            r = cached(provs, cur, "prov")
            if not r:
                break
            parent = next((by_sha[x.get("sha256")] for x in r.get("inputs") or [] if by_sha.get(x.get("sha256"))), None)
            if not parent:
                return (r.get("meditation") or {}).get("goal") or r.get("prompt") or (cached(metas, cur, "meta") or {}).get("prompt") or ""
            cur = parent
        return (cached(metas, jid, "meta") or {}).get("prompt") or ""

    def small_jpeg(png):
        from PIL import Image
        im = Image.open(io.BytesIO(png)).convert("RGB"); im.thumbnail((640, 640))
        out = io.BytesIO(); im.save(out, "JPEG", quality=85)
        return "data:image/jpeg;base64," + b64e(out.getvalue())

    end = time.time() + MAX_HOURS * 3600
    done = failed = 0
    while time.time() < end and items.is_dir():
        ids = [f.name.split(".")[0] for f in items.glob("*.state.json")]
        graded, graded_ids, pics, by_sha = set(), set(), [], {}
        for jid in ids:
            m = cached(metas, jid, "meta") or {}
            if m.get("kind") == "qgrade":
                graded.add(m.get("sha256")); graded_ids.add(m.get("picture"))
            elif m.get("kind") not in NOT_PICTURES and (items / f"{jid}.result.json").exists():
                p = cached(provs, jid, "prov")  # older pictures have no history: known by their id, fingerprinted when graded
                h = p.get("sha256") if p else None
                pics.append((jid, h))
                if h:
                    by_sha[h] = jid
        todo = [(jid, h) for jid, h in pics if jid not in graded_ids and (not h or h not in graded)]
        say({"status": {"state": "grading" if todo else "waiting for new pictures", "done": done, "failed": failed,
                        "left": len(todo), "pictures": len(pics), "graded": len(pics) - len(todo)}})
        for n, (jid, h) in enumerate(todo):
            res = read(jid, "result")
            if not res or not res.get("png"):
                continue
            png = b64d(res["png"]); del res
            h = h or hashlib.sha256(png).hexdigest()
            img = small_jpeg(png); del png
            say({"grade": {"image": img, "request": root_prompt(jid, by_sha)}}); del img
            line = sys.stdin.readline()
            if not line:
                return  # the server went away: the key goes with this process
            reply = json.loads(line)
            if reply.get("stop"):
                return
            g = reply.get("g")
            if not g:
                failed += 1
            else:
                gid = os.urandom(6).hex()
                write(gid, "meta", seal({"kind": "qgrade", "sha256": h, "picture": jid, **g, "created": time.time()}))
                now = time.time()
                write(gid, "state", {"id": gid, "state": "done", "updated": now, "created": now})
                done += 1
                if reply.get("delete") and not (items / f"{jid}.protect").exists():  # 🧹 auto-delete (never 🛡 protected) → 🗑 trash
                    trash = vault_dir / "trash"
                    trash.mkdir(exist_ok=True)
                    for f in items.glob(f"{jid}.*"):
                        f.replace(trash / f.name)
                    (trash / f"{jid}.at").write_text(str(time.time()))
            say({"status": {"state": "grading", "done": done, "failed": failed, "left": len(todo) - n - 1,
                            "pictures": len(pics), "graded": len(pics) - len(todo) + n + 1}})
            if time.time() > end or not items.is_dir():
                break
        if not todo:
            say({"status": {"state": "waiting for new pictures", "done": done, "failed": failed, "left": 0,
                            "pictures": len(pics), "graded": len(pics)}})
            for _ in range(RESCAN):
                time.sleep(1)
                if not items.is_dir():
                    return


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        say({"status": {"state": "stopped: " + type(e).__name__}})  # no message: it could echo private data
