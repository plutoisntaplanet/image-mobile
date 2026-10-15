"""In-memory candidate history for a future separate Strategist meditation mode.

Pictures, prompts, issues and reasons are private. This module performs no I/O; callers
must seal any report they persist and clear the history when the run ends.
"""
from dataclasses import dataclass, field
import math


SIZES = ('sketch', 'quick', 'preview', 'fast', 'full')
ROLES = frozenset(('face', 'likeness', 'identity', 'expression', 'pose',
                   'outfit', 'object', 'place', 'style'))


@dataclass(frozen=True)
class Choice:
    base: int
    action: str
    prompt: str = field(repr=False)
    size: str | None = None
    turbo: bool | None = None
    refs: tuple[tuple[str, str], ...] = ()
    why: str = field(default='', repr=False)


@dataclass
class Candidate:
    id: int
    parent: int | None
    image: bytes = field(repr=False)
    prompt: str = field(repr=False)
    action: str | None = None
    requested_size: str | None = None
    requested_turbo: bool | None = None
    size: str | None = None
    turbo: bool | None = None
    refs: tuple[tuple[str, str], ...] = ()
    reason: str = field(default='', repr=False)
    render_seconds: float | None = None
    score: int | None = None
    issues: tuple[str, ...] = field(default=(), repr=False)
    change_analysis: str = field(default='', repr=False)
    review_seconds: float | None = None


class CandidateHistory:
    """Own every image in one run, including the starting image as candidate zero."""

    def __init__(self, image: bytes, prompt: str, *, max_candidates=31, max_bytes=160_000_000):
        if not isinstance(image, bytes) or not image:
            raise ValueError('starting image is required')
        if max_candidates < 1 or max_bytes < len(image):
            raise ValueError('history budget is too small')
        self.candidates = [Candidate(0, None, image, prompt)]
        self.best_id = None  # selection is explicit; scores alone do not choose a winner
        self.max_candidates = max_candidates
        self.max_bytes = max_bytes
        self.image_bytes = len(image)

    def get(self, candidate_id: int) -> Candidate:
        if type(candidate_id) is not int or candidate_id < 0 or candidate_id >= len(self.candidates):
            raise ValueError('base is not an earlier candidate')
        return self.candidates[candidate_id]

    def validate_choice(self, raw, available_refs=(), *, can_redraw=False) -> Choice:
        """Reject malformed model output instead of silently changing its choice."""
        if not isinstance(raw, dict):
            raise ValueError('choice must be an object')
        base = raw.get('base')
        self.get(base)
        action = raw.get('action')
        if action not in ('edit', 'redraw', 'stop'):
            raise ValueError('unknown action')
        why = raw.get('why')
        if not isinstance(why, str) or not why.strip() or len(why) > 300:
            raise ValueError('a short reason is required')
        if action == 'stop':
            return Choice(base, action, '', why=why.strip())
        if action == 'redraw' and not can_redraw:
            raise ValueError('redraw is unavailable')
        prompt = raw.get('prompt')
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 2000:
            raise ValueError('a render instruction is required')
        size = raw.get('size')
        turbo = raw.get('turbo')
        if size not in SIZES or type(turbo) is not bool:
            raise ValueError('size and turbo must be chosen explicitly')
        refs = raw.get('refs', [])
        if not isinstance(refs, list) or len(refs) > 3:
            raise ValueError('up to three references are allowed')
        selected = []
        for ref in refs:
            if not isinstance(ref, dict) or not isinstance(ref.get('id'), str) or \
                    ref['id'] not in available_refs or ref.get('role') not in ROLES:
                raise ValueError('reference or role is unavailable')
            selected.append((ref['id'], ref['role']))
        if len({ref_id for ref_id, _ in selected}) != len(selected):
            raise ValueError('duplicate reference')
        return Choice(base, action, prompt.strip(), size, turbo, tuple(selected), why.strip())

    def add_render(self, choice: Choice, image: bytes, *, actual_size: str, actual_turbo: bool,
                   render_seconds: float) -> Candidate:
        """Record the actual result; parent is the selected earlier candidate."""
        base = self.get(choice.base)
        if base.score is None:
            raise ValueError('review the base before drawing from it')
        if choice.action not in ('edit', 'redraw') or choice.size not in SIZES:
            raise ValueError('stop does not create a picture')
        if actual_size not in SIZES or type(actual_turbo) is not bool or \
                type(render_seconds) not in (int, float) or not math.isfinite(render_seconds) or render_seconds < 0:
            raise ValueError('actual render settings and time are required')
        if not isinstance(image, bytes) or not image:
            raise ValueError('rendered image is required')
        if len(self.candidates) >= self.max_candidates or self.image_bytes + len(image) > self.max_bytes:
            raise MemoryError('candidate history limit reached')
        candidate = Candidate(id=len(self.candidates), parent=choice.base, image=image,
                              prompt=choice.prompt, action=choice.action,
                              requested_size=choice.size, requested_turbo=choice.turbo,
                              size=actual_size, turbo=actual_turbo, refs=choice.refs,
                              reason=choice.why, render_seconds=render_seconds)
        self.candidates.append(candidate)
        self.image_bytes += len(image)
        return candidate

    def add_review(self, candidate_id: int, score: int, issues=(), *, review_seconds: float, change_analysis=''):
        candidate = self.get(candidate_id)
        if candidate.score is not None:
            raise ValueError('candidate was already reviewed')
        if type(score) is not int or score < 1 or score > 10 or \
                type(review_seconds) not in (int, float) or not math.isfinite(review_seconds) or review_seconds < 0:
            raise ValueError('score and review time are invalid')
        if not isinstance(issues, (list, tuple)) or any(not isinstance(x, str) for x in issues):
            raise ValueError('issues must be text')
        candidate.score = score
        candidate.issues = tuple(issues[:4])
        candidate.change_analysis = str(change_analysis)[:500]
        candidate.review_seconds = review_seconds

    def select_best(self, candidate_id: int):
        candidate = self.get(candidate_id)
        if candidate.score is None:
            raise ValueError('review the candidate before selecting it')
        self.best_id = candidate_id

    def brief(self):
        """Private text for the planner; callers keep it in memory."""
        return [{'id': c.id, 'parent': c.parent, 'prompt': c.prompt, 'action': c.action,
                 'requested_size': c.requested_size, 'requested_turbo': c.requested_turbo,
                 'size': c.size, 'turbo': c.turbo, 'refs': list(c.refs),
                 'render_seconds': c.render_seconds,
                 'score': c.score, 'issues': list(c.issues), 'change_analysis': c.change_analysis, 'reason': c.reason}
                for c in self.candidates]

    def path(self):
        """Private report for an unlocked or sealed view; image bytes are excluded."""
        return [{'id': c.id, 'parent': c.parent, 'action': c.action, 'size': c.size,
                 'turbo': c.turbo, 'requested_size': c.requested_size,
                 'requested_turbo': c.requested_turbo,
                 'settings_honored': c.size == c.requested_size and c.turbo == c.requested_turbo
                 if c.parent is not None else None,
                 'render_seconds': c.render_seconds,
                 'review_seconds': c.review_seconds, 'score': c.score,
                 'best': c.id == self.best_id}
                for c in self.candidates]

    def clear(self):
        """Release pictures and text at the end of the run."""
        self.candidates.clear()
        self.best_id = None
        self.image_bytes = 0
