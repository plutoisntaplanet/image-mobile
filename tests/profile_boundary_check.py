#!/usr/bin/env python3
"""Profile photos never reach Normal storage: every image-taking POST refuses `profile_inputs` outside Profile mode."""
import base64
import sys
from pathlib import Path
from urllib.error import HTTPError
from harness import Lab, check

PNG = 'data:image/png;base64,' + base64.b64encode(
    base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==')).decode()
# (path, body): every one carries photos and takes `mode`; none may be accepted flagged as profile in a non-profile mode
CASES = [
    ('/api/generate', {'prompt': 'x', 'images': [PNG]}),
    ('/api/generate', {'prompt': 'x', 'images': [PNG], 'finish': True}),
    ('/api/meditate', {'image': PNG, 'goal': 'x'}),
    ('/api/abtest', {'prompt': 'x', 'images': [PNG], 'version': {'id': 'v'}}),
    ('/api/abtest/meditate', {'image': PNG, 'goal': 'x'}),
    ('/api/chat', {'pictures': [{'image': PNG}], 'messages': [{'role': 'user', 'text': 'hi'}]}),
    ('/api/lineup/review', {'story': 'abc', 'shots': [{'images': [PNG]}]}),
    ('/api/story/write', {'idea': 'x', 'cast': [{'name': 'a', 'image': PNG}]}),
    ('/api/story/save', {'title': 't', 'text': 'x', 'cast': [{'name': 'a', 'image': PNG}]}),
    ('/api/story/direct', {'idea': 'x', 'cast': [{'name': 'a', 'image': PNG}]}),
    ('/api/story/scene', {'story': 'abc', 'cast': [{'name': 'a', 'image': PNG}]}),
    ('/api/story/photo', {'story': 'abc', 'cast': [{'name': 'a', 'image': PNG}]}),
    ('/api/story/cast', {'story': 'abc', 'cast': [{'name': 'a', 'image': PNG}]}),
    ('/api/asset/save', {'image': PNG, 'name': 'p'}),
    ('/api/character/capture', {'name': 'a', 'photos': [PNG]}),
    ('/api/swap/find', {'images': [PNG]}),
    ('/api/request/edit', {'id': 'aaaaaaaaaaaa', 'prompt': 'x', 'images': [PNG]}),
]


def tree(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file())


def main():
    pass
    t = Lab()
    sv = t.sv
    before = tree(sv.DATA) + tree(sv.IMG_DIR)
    for path, body in CASES:
        for mode in ('normal', 'temporary', None):
            b = {**body, 'profile_inputs': True, **({'mode': mode} if mode else {})}
            try:
                t.api(path, b)
                check(False, f'{path} accepted profile photos in {mode or "no"} mode')
            except HTTPError as e:
                check(e.code == 409, f'{path} ({mode}): expected 409, got {e.code}')
    check(tree(sv.DATA) + tree(sv.IMG_DIR) == before, 'a refused request wrote to Normal storage')
    print('profile boundary: PASS', len(CASES) * 3, 'refusals')


if __name__ == '__main__':
    main()
