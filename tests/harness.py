"""Test harness: copies the app into a temporary directory, points HOME and IMAGINE_DATA there, replaces the
picture engine with a fake one and serves the app on a random local port. Nothing real is touched."""
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'tests/assets'
APP_FILES = ('server.py', 'engine.py', 'index.html', 'meditation_strategy.py', 'meditation_metrics.py', 'grader.py',
             'scorers.py', 'imagine_requests.py')


def check(ok, evidence):
    if not ok:
        raise AssertionError(evidence)
    return evidence


class Lab:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='imagine-test-')
        home = Path(self.tmp.name) / 'home'
        app = Path(self.tmp.name) / 'app'
        home.mkdir(); app.mkdir()
        for name in APP_FILES:
            if (ROOT / name).exists():
                shutil.copy2(ROOT / name, app / name)
        os.environ['HOME'] = str(home)
        os.environ['IMAGINE_DATA'] = str(home / 'data')
        for key in [k for k in os.environ if k.startswith('IMAGINE_') and k != 'IMAGINE_DATA']:
            del os.environ[key]
        sys.path.insert(0, str(app))
        import server as sv
        self.sv = sv
        self.calls = []
        for d in (sv.IMG_DIR, sv.STORIES, sv.ASSETS, sv.VAULTS, sv.REQUESTS):
            d.mkdir(parents=True, exist_ok=True)
        sv.qi.render = self.fake_render
        sv.qi.cancel = lambda pid: None
        sv.qi.available_models = lambda: ['test-model']
        sv.qi.current_model = lambda: 'test-model'
        self.http = sv.ThreadingHTTPServer(('127.0.0.1', 0), sv.H)
        self.url = 'http://127.0.0.1:' + str(self.http.server_port)
        for target in (self.http.serve_forever, sv.worker):
            threading.Thread(target=target, daemon=True).start()

    def fake_render(self, a, images=(), log=None):
        self.calls.append({k: v for k, v in vars(a).items() if not callable(v)})
        if callable(getattr(a, 'on_submit', None)):
            a.on_submit('test-' + str(len(self.calls)))
        from PIL import Image
        src = Image.open(ASSETS / 'lincoln_1863_head.jpg').convert('RGB')
        src.thumbnail((192, 192))
        out = io.BytesIO(); src.save(out, 'PNG')
        return out.getvalue()

    def api(self, path, body=None):
        req = urllib.request.Request(self.url + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={'X-Imagine': '1', 'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())

    def wait_idle(self, timeout=60):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if not self.sv.series and not any(j['status'] in ('running', 'queued') for j in self.sv.jobs.values()):
                return
            time.sleep(.1)
        raise TimeoutError('jobs did not finish')
