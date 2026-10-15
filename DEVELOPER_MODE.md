# Developer mode

Enable Developer mode in Settings. Notebook and rules load directly from the same
sources as Meditation. Each source appears as it arrives; a failed or slow source
does not hide the rest. Already opened Profile prompts are reused from browser
memory. Load Profile prompts scans additional encrypted records separately, with
progress, limited concurrency and a timeout; it never blocks copying notebook or
rules. Refresh sources retries shared context loading. Lists start collapsed. Select all includes every loaded notebook lesson, rule and
user prompt, even when filtered out, plus the selected system prompt. Each list also
has its own Select all button. Newly loaded prompts join an active Select all
selection; manually changing a checkbox ends that automatic selection.
Select notebook lessons or rules and use their
Copy buttons. Each part explains what it is, includes your task, and asks the agent
for improved plain text. Copy whole prompt combines the selected parts. User prompts
can be included as examples; the system prompt is optional. Normal exports are text
files ready to paste into an agent. Images are excluded.

Paste the improved lessons or rules, with a blank line between items. Replace selected
items changes only the selected entries and saves the previous text under Versions.
Restore previous text reverses that change if those items have not since been edited.
Manual updates survive automatic digesting. Replacement rules are active owner rules;
they do not inherit old testing scores or pretend to be proven. Save version only
keeps a version without replacing running lessons or rules. System prompt versions
can be saved and copied but do not change the app's built-in templates.

Profile copies and exports remain encrypted using a separate export password that is
never saved. Use View encrypted context in an unlocked Profile to open an exported file with its
export password. Decryption happens only in this browser; the opened text is never
uploaded or saved. Clear opened context removes it. Copy reports errors beside its
button and offers selectable encrypted text when clipboard access is blocked.
A tool must decrypt exports before the plain prompt can be pasted into an agent. Profile versions stay browser-sealed. Shared rules and lessons can be replaced
only from Normal, while logged in. Profile text is never sent to the shared replacement
endpoint. Locking or switching profiles clears the private text and export password.

## Encrypted export format

The envelope remains `imagine-agent-context-encrypted`, version 1, with `cipher`,
`kdf`, `iv`, `aad`, and `ct`. Standard base64 encodes the salt, IV and ciphertext.
Derive a 32-byte key from the UTF-8 export password using PBKDF2-HMAC-SHA256 and the
provided salt and iteration count (600000). Decrypt using AES-256-GCM, the decoded
12-byte IV and UTF-8 AAD `imagine-agent-context:v1`. The ciphertext ends with its
16-byte authentication tag. The decrypted UTF-8 JSON package contains a `text` field:
this is the plain prompt to give the agent. `context` retains the selected source text
and metadata for compatibility. It no longer requires a structured agent response.
Do not log or save decrypted Profile context in plaintext.

## Isolated checks

`python3 tests/smoke_test.py` and `python3 tests/profile_boundary_check.py` run the app in a throwaway
directory with a fake engine and synthetic data; they never touch your real data.
