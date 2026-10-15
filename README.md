# Imagine

A private, single-user web UI for drawing, editing and improving pictures with your own
[ComfyUI](https://github.com/comfyanonymous/ComfyUI) image engine and (optionally) your own
OpenAI-compatible language model. It adds a job queue, a gallery, a reusable photo library, picture
stories, batches, prompt improvement, automatic review-and-improve loops ("Meditate"), grading, and an
encrypted **Profile** mode whose contents the server can never read.

No models are bundled and none are named here: you choose them. Third-party model, LoRA and upscaler
licences are your responsibility.

## Features

- Create: new image, edit image (up to 4 input images with roles), batch (locked + cycled inputs), draft sizes and a Finish pass to full size, optional speed LoRA and upscaler.
- Gallery, History (every request can be reloaded or rerun), Library of tagged reference photos, Stories and lineups.
- Language-model features (need `IMAGINE_LLM_URL`): Improve prompt, chat with picture suggestions, Meditate (a reviewer scores each picture and writes the next prompt or edit), grading, auto-labelling.
- Profile mode: prompts, pictures, library, history and chats are sealed in the browser; the server stores only ciphertext.
- Update button: saves unfinished work, restarts (if you set `IMAGINE_RESTART_CMD`) and resumes.

## Quick start

```sh
pip install cryptography pillow websocket-client
python3 server.py            # http://127.0.0.1:8791
```

Open the page. Until the image engine is configured, Create shows "configure your model".

## Configure your image model

Imagine builds a ComfyUI workflow from names **you** provide. Put them in the environment or in
`~/.config/imagine/engine.env` (`KEY=value` lines; override the path with `IMAGINE_CONFIG`).
Use the file names exactly as ComfyUI lists them.

| Variable | Meaning | Required |
|---|---|---|
| `IMAGINE_COMFY_URL` | ComfyUI address, default `http://127.0.0.1:8188` | no |
| `IMAGINE_MODEL` | image model file | yes |
| `IMAGINE_TEXT_ENCODER` | text encoder file | yes |
| `IMAGINE_TEXT_ENCODER_TYPE` | the `type` value ComfyUI's CLIP loader needs for your model family | yes |
| `IMAGINE_VAE` | VAE file | yes |
| `IMAGINE_NODE_LOADER` | ComfyUI node class that loads the image model | yes |
| `IMAGINE_NODE_ENCODER` | ComfyUI node class that encodes prompt + input images | yes |
| `IMAGINE_LORA`, `IMAGINE_NODE_LORA`, `IMAGINE_NODE_SIGMAS`, `IMAGINE_LORA_SIGMAS` | optional speed LoRA: file, loader node, step-schedule node, schedule values | no (Turbo is off without them) |
| `IMAGINE_UPSCALER` | optional upscaler model file (needed for Upscale and Finish) | no |
| `IMAGINE_MODEL_FILTER` | only list model files containing this text in the model picker | no |

The workflow wiring (node inputs such as `clip`, `vae`, `prompt`, `images.image_N`) follows ComfyUI's
standard loader and sampler nodes; your encoder node must accept those inputs.

## Language model (optional)

| Variable | Meaning |
|---|---|
| `IMAGINE_LLM_URL` | an OpenAI-compatible chat endpoint (`http://host:port/v1` or the full `/chat/completions` URL); needs vision for picture review |
| `IMAGINE_LLM_KEY` | optional bearer key |
| `IMAGINE_LLM_MODEL` | optional model name sent with each request |

Without it Improve, Meditate, chat and grading answer with a clear message and everything else works.

## Server settings

| Variable | Meaning | Default |
|---|---|---|
| `IMAGINE_HOST` / `IMAGINE_PORT` | listen address and port | `127.0.0.1` / `8791` |
| `IMAGINE_DATA` | data directory (gallery, library, sealed vaults) | `~/.local/share/imagine` |
| `IMAGINE_RESTART_CMD` | command the Update button runs to restart the app | unset |

A systemd example is in `tools/systemd/imagine.service`.

## Security notes

- Imagine has **no user accounts and no TLS**. Normal and Temporary modes are open to anyone who can reach the port. Keep it on `127.0.0.1`, or put your own authentication and HTTPS in front (a reverse proxy, a VPN). It is not meant for untrusted networks.
- Profile mode: the browser derives keys from the profile password (PBKDF2) and seals prompts and pictures to the profile's public key; the server stores sealed data and cannot decrypt it. Plaintext exists in server memory only while a job for that profile is queued or running. The password never reaches the server; lose it and the profile is gone.
- Prompts are never written to logs or command lines; the process is made non-dumpable.
- Language-model calls send your prompts and pictures to whatever endpoint you configure.

## Tests

```sh
python3 tests/smoke_test.py
python3 tests/profile_boundary_check.py
```

They run the app in a temporary directory with a fake engine; test pictures are public-domain (see `tests/assets/SOURCES.md`).

## License

MIT, see `LICENSE`.
