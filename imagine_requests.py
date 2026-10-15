"""Additive Request v2 records. No global paths, startup migration or Profile I/O.

Flat v1 fields remain the compatibility projection for existing pages. A render's
id identifies an occurrence; its SHA identifies pixels and may repeat.
"""
import base64
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import time

ATTRIBUTES = ('aspect', 'quality', 'turbo', 'upscale', 'models', 'count', 'forever',
              'raw', 'blend_mix', 'negative', 'seed', 'fast', 'preview')
_LOCKS = {}
_LOCKS_GATE = threading.Lock()


def valid_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{12}', value):
        raise ValueError('Invalid request id')
    return value


def snapshot(record):
    """Freeze the prompt, settings and inputs that a revision actually used."""
    images, roles = record.get('images') or [], record.get('roles') or []
    inputs = []
    for i, data in enumerate(images):
        role = roles[i] if i < len(roles) and isinstance(roles[i], dict) else {}
        try:
            raw = base64.b64decode(data.split(',', 1)[1], validate=True)
        except (ValueError, IndexError, TypeError):
            raise ValueError('Invalid request image') from None
        inputs.append({'i': i, 'sha256': hashlib.sha256(raw).hexdigest(),
                       'role': role.get('role') or ('edit' if i == 0 else 'face'),
                       'who': role.get('who') or '', 'source': {'upload': True}, 'data': data})
    return {'prompt': record.get('prompt') or '',
            'attributes': {k: deepcopy(record[k]) for k in ATTRIBUTES if k in record},
            'inputs': inputs}


def add_revision(doc, record, *, author=None, status='sent', at=None):
    at = time.time() if at is None else at
    frozen = snapshot(record)
    author = author or record.get('prompt_by') or 'unknown'
    revision = {'rev': len(doc['revisions']) + 1, 'created': at, 'author': author,
                'status': status, 'by': {}, **frozen}
    doc['revisions'].append(revision)
    doc.update(attributes=deepcopy(frozen['attributes']), inputs=deepcopy(frozen['inputs']), updated=at)
    return revision['rev']


def to_v2(record, rid=None, *, at=None):
    """Pure, repeatable conversion of one explicit record; never scans storage."""
    doc = deepcopy(record)
    rid = valid_id(rid or doc.get('id'))
    if doc.get('v') == 2:
        if doc.get('id') != rid or doc.get('mode') != 'normal':
            raise ValueError('Request identity or mode mismatch')
        return doc
    if doc.get('v', 1) != 1:
        raise ValueError('Unsupported request version')
    if doc.get('mode', 'normal') != 'normal':
        raise ValueError('Profile and Temporary records are not normal requests')
    at = time.time() if at is None else at
    created = doc.get('created', at)
    doc.update(id=rid, v=2, kind='request', mode='normal',
               request_kind='edit' if any(r.get('role') == 'edit' for r in doc.get('roles', []) if isinstance(r, dict)) else 'create',
               title=str(doc.get('prompt') or '')[:80], created=created, updated=at,
               legacy=deepcopy(record), legacy_ids=[], revisions=[], plans=[],
               operations=[], renders=[], evaluations=[], verdicts={})
    if doc.get('previous_request'):
        doc['parent'] = {'request': doc['previous_request'], 'how': 'revision'}
    add_revision(doc, record, at=created)
    doc['updated'] = at
    return doc


def revise(doc, record, *, at=None):
    """Append an immutable revision, preserving all prior operations and outputs."""
    add_revision(doc, record, at=at)
    for k in ('prompt', 'images', 'roles', 'prompt_by', 'edited', *ATTRIBUTES):
        if k in record:
            doc[k] = deepcopy(record[k])
    return doc


def add_operation(doc, operation):
    op = deepcopy(operation)
    valid_id(op['id'])
    existing = next((x for x in doc['operations'] if x['id'] == op['id']), None)
    if existing:
        existing.update(op)
    else:
        doc['operations'].append(op)
    derive_status(doc)


def derive_status(doc):
    ops = doc.get('operations') or []
    if not ops:
        return
    active = [x for x in ops if x['status'] in ('queued', 'running', 'paused')]
    selected = max(active or ops, key=lambda x: x.get('finished') or x.get('created') or 0)
    status = selected['status']
    doc['last_run'] = {'status': {'error': 'failed', 'stopped': 'cancelled',
                                 'queued': 'running', 'paused': 'running'}.get(status, status),
                       'at': selected.get('finished') or selected.get('created') or doc['updated']}


def add_render(doc, render):
    """Retries are idempotent; identical pixels from separate jobs remain distinct."""
    entry = deepcopy(render)
    valid_id(entry['id'])
    if not re.fullmatch(r'[0-9a-f]{64}', entry.get('sha256', '')):
        raise ValueError('Invalid render SHA')
    existing = next((x for x in doc['renders'] if x['id'] == entry['id']), None)
    if existing and existing != entry:
        raise ValueError('Render occurrence already recorded with different content')
    if not existing:
        doc['renders'].append(entry)


class RequestStore:
    """Atomic normal-mode writes, serialized across threads and app processes.

    Constructing a store never touches disk. Missing records are never recreated
    by late worker updates after deletion. Legacy conversion happens only for an
    explicit create/revise/render action, not for listing or startup recovery.
    """
    def __init__(self, directory):
        self.directory = Path(directory)
        with _LOCKS_GATE:
            self.lock = _LOCKS.setdefault(str(self.directory.resolve()), threading.RLock())

    def path(self, rid):
        return self.directory / (valid_id(rid) + '.json')

    def read(self, rid):
        try:
            doc = json.loads(self.path(rid).read_text())
        except FileNotFoundError:
            return None
        if doc.get('id') != rid or doc.get('mode', 'normal') != 'normal':
            raise ValueError('Request identity or mode mismatch')
        return doc

    def _write(self, rid, doc):
        dest = self.path(rid)
        fd, tmp = tempfile.mkstemp(prefix='.' + rid + '-', dir=self.directory)
        try:
            with os.fdopen(fd, 'w') as stream:
                json.dump(doc, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, dest)
            directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            Path(tmp).unlink(missing_ok=True)

    def transaction(self, rid, change, *, create=None, upgrade=False):
        valid_id(rid)
        with self.lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            with (self.directory / '.request.lock').open('a') as gate:
                fcntl.flock(gate, fcntl.LOCK_EX)
                doc = self.read(rid)
                if create is not None:
                    if doc is not None:
                        raise ValueError('Request already exists')
                    doc = to_v2(create, rid)
                elif doc is None:
                    return None
                elif upgrade:
                    doc = to_v2(doc, rid)
                change(doc)
                doc['updated'] = time.time()
                self._write(rid, doc)
                return deepcopy(doc)

    def create(self, rid, record):
        return self.transaction(rid, lambda doc: None, create=record)

    def interrupt(self, rid):
        """Recover interrupted work before restoring any runs; do not upgrade v1."""
        def change(doc):
            if (doc.get('last_run') or {}).get('status') != 'running':
                return
            at = time.time()
            for op in doc.get('operations', []):
                if op['status'] in ('queued', 'running', 'paused'):
                    op.update(status='stopped', finished=at)
            doc['last_run'] = {'status': 'cancelled', 'at': at}
        return self.transaction(rid, change)

    def delete(self, rid):
        # Preserve existing delete semantics until the archive UI ships.
        with self.lock:
            if not self.directory.exists():
                valid_id(rid)
                return
            with (self.directory / '.request.lock').open('a') as gate:
                fcntl.flock(gate, fcntl.LOCK_EX)
                self.path(rid).unlink(missing_ok=True)
