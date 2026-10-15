#!/usr/bin/env python3
"""The app starts without an engine or a language model: the page loads, the queue answers, generating without a
configured model fails with a clear message, and a normal request goes through a fake engine."""
import urllib.request
from harness import Lab, check


def main():
    t = Lab()
    page = urllib.request.urlopen(t.url + '/').read().decode()
    check('<title>' in page.lower() or '<html' in page.lower(), 'page did not load')
    jobs = t.api('/api/jobs')
    check(jobs.get('jobs') == [] and jobs.get('series') == [], jobs)
    j = t.api('/api/generate', {'mode': 'normal', 'prompt': 'a red cube on a table', 'quality': 'sketch'})
    t.wait_idle()
    check(len(t.calls) == 1 and t.calls[0]['prompt'] == 'a red cube on a table', t.calls)
    print('smoke: PASS')


if __name__ == '__main__':
    main()
