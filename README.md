# AutoLabel-Tool

Auto-annotation models **inside CVAT**: pick a model from *Actions → Automatic annotation* or *AI Tools → Interactors*, and the labels appear in your task. No extra web page to open.

| Model in CVAT | What it returns | Engine |
|---|---|---|
| AutoLabel: SAM3 boxes (COCO-80 / Cabin / your labels) | bounding boxes for any label name | SAM3 text prompts |
| AutoLabel: SAM3 masks | masks; tick *Convert masks to polygons* in CVAT for polygons | SAM3 |
| AutoLabel: SAM3 click-to-segment | click points or draw a box → mask (interactive) | SAM3 |
| AutoLabel: Human pose (COCO-17) | `pose` skeleton, 17 points per person | YOLOX + RTMW |
| AutoLabel: Face landmarks (68) | `face68` skeleton (iBUG / 300W order) | RTMW |
| AutoLabel: Face13 (Cabin) | `face13` skeleton (eyes, nose, mouth; Cabin DMS schema), auto-rotates sideways cameras | RTMW |
| AutoLabel: Hand keypoints (21) | `hand` skeleton with a `side` attribute (person's left/right) | RTMW |

## How it works

```
CVAT (Docker) ──► Nuclio proxy functions (tiny, stdlib only) ──► host.docker.internal
                                                                     │
                     your machine:  SAM3 server :8766 (Apple GPU / CUDA / CPU)
                                    keypoint server :8767 (YOLOX + RTMW, CPU)
```

The models run natively on the host, not in Docker, so SAM3 can use the Apple Silicon GPU (MPS). Both servers listen on `127.0.0.1` only. CVAT's Nuclio functions are thin proxies, and `./autolabel.sh deploy` creates them for you.

## Requirements

- A local CVAT with the serverless (Nuclio) components running. Tested with CVAT 2.82. CVAT must be started with `components/serverless/docker-compose.serverless.yml`.
- macOS (Apple Silicon) or Linux. Python 3.10–3.12, git, Docker.
- Access to the SAM3 checkpoint on Hugging Face (gated): request it at <https://huggingface.co/facebook/sam3>.

## Install

```bash
git clone https://github.com/Cily49/AutoLabel-Tool.git && cd AutoLabel-Tool
./setup.sh          # app env + SAM3 env (reuses ~/sam3 if it works); prints the hf login command if needed
./autolabel.sh start    # start the model servers (or: ./autolabel.sh install-service to start at login)
./autolabel.sh deploy   # add the models to CVAT (builds the Nuclio proxies, a few minutes the first time)
./autolabel.sh status
```

If SAM3 is not installed yet, `setup.sh` clones it at a tested commit and applies `patches/sam3_mac_mps.patch`. The patch picks cuda, then mps, then cpu, and avoids the CUDA-only Triton import, so SAM3 also runs on Apple Silicon. On Linux with an NVIDIA GPU, set `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128` before `./setup.sh`.

## Menu bar app (macOS)

```bash
./autolabel.sh app      # creates ~/Applications/AutoLabel.app and opens it at login
```
A tag icon appears in the menu bar near the clock. It shows `!` when something is not ready, and `…` while deploying. From its menu you can:
- see whether the SAM3 / keypoint servers are ready and how many CVAT models are deployed (refreshed every 10 s; red when CVAT / Docker is not reachable)
- start or stop the model servers, and toggle *Start servers at login* and *Open this menu at login*
- open CVAT or its Models page, deploy the models, edit `functions.yaml`, open the logs

The menu runs as the launchd job `com.autolabel.menubar`; opening `AutoLabel.app` starts it. If the icon does not appear, check *System Settings → Menu Bar → Allow in the Menu Bar* and enable **python3.x**. You can also free some room: hold ⌘ and drag other icons off the menu bar (Macs with a notch hide items that do not fit).

## Use in CVAT

**Automatic annotation (whole task):** open the task → *Actions* → *Automatic annotation* → choose an *AutoLabel:* model → map its labels to your task's labels (same names match automatically) → set *Threshold* → *Annotate*.

**Interactive segmentation:** in a job, open *AI Tools* → *Interactors* → *AutoLabel: SAM3 click-to-segment*. Click inside the object (positive), outside it (negative), or draw a box.

**Skeleton labels:** your task needs skeleton labels with the same point names. Print a ready-made definition and paste it into *Project/Task → Labels → Raw*:
```bash
./autolabel.sh labels pose      # also: face68, face13, hand
```

## Choose your own labels

Edit `functions.yaml`, then run `./autolabel.sh deploy` again:
```yaml
  - id: autolabel-sam3-site
    name: "AutoLabel: SAM3 boxes (site safety)"
    kind: sam3-detect
    labels: [person, helmet, safety vest, forklift]
    output: box            # box | mask | polygon
```
The label names are the SAM3 text prompts. Better prompts for some names live in `autolabel/vocab.py`. SAM3 runs one prompt per label, so a short list is fast (about 1–2 s per frame for 5 labels on an M-series Mac), while COCO-80 takes about 10–15 s per frame.

Keypoint options per model: `skeleton` (body17 | face68 | face13 | hand21), `auto_rotate`, `mode` (performance | lightweight), `out_thr` (points scored below this are sent as *outside*), `face_thr` (skip faces that are not clearly visible).

## Commands

| Command | Does |
|---|---|
| `./autolabel.sh start / stop / status` | run, stop, or check the model servers; `status` also shows the CVAT models |
| `./autolabel.sh install-service` | start the servers at login (macOS launchd / Linux systemd user unit) |
| `./autolabel.sh deploy [--only id ...]` | add or update the models in CVAT |
| `./autolabel.sh undeploy` | remove the AutoLabel models from CVAT |
| `./autolabel.sh labels pose` | print a CVAT Raw skeleton label definition |
| `./autolabel.sh app` | create the macOS menu bar app (`~/Applications/AutoLabel.app`) |

Logs are in `~/.autolabel/logs/`.

## Limitations

- CVAT's Nuclio integration keeps only the *outside* flag of skeleton points (*occluded* is always reset to false). Points the model cannot see are sent as *outside*. Mark occluded points by hand.
- The labels are drafts: review every frame in CVAT.
- *Tracking* is not included yet.

## Licenses

The tool's code is yours to share. The models have their own licenses: SAM3 uses the SAM License (Meta; accept it on Hugging Face), and the patch in `patches/` modifies SAM3 code under that license. RTMW, YOLOX and rtmlib are Apache-2.0. Check the license of the data you label.
