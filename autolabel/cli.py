"""AutoLabel-Tool manager: add auto-annotation models to a local CVAT and run their model servers.

    ./autolabel.sh start                start the model servers in the background (SAM3 + keypoints)
    ./autolabel.sh deploy               add/update the models from functions.yaml in CVAT (Nuclio)
    ./autolabel.sh status | stop        check / stop the model servers
    ./autolabel.sh install-service      start the model servers automatically at login
    ./autolabel.sh labels pose          print a CVAT "Raw" label definition (pose|face68|face13|hand)
    ./autolabel.sh undeploy             remove the AutoLabel models from CVAT

Then in CVAT: open a task > Actions > Automatic annotation > pick an "AutoLabel: ..." model,
or in a job: AI Tools > Interactors > "AutoLabel: SAM3 click-to-segment".
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
HOME = Path(os.environ.get("AUTOLABEL_HOME", Path.home() / ".autolabel"))
LOGS = HOME / "logs"
PIDFILE = HOME / "servers.pid"
PY = sys.executable


# --------------------------------------------------------------------------- config
def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def sam3_python() -> str:
    env = ROOT / "config.env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("SAM3_PYTHON="):
                return line.split("=", 1)[1].strip().strip("'\"")
    return os.environ.get("SAM3_PYTHON", str(Path.home() / "sam3/.venv/bin/python"))


def ports(cfg: dict) -> tuple[int, int]:
    s = cfg.get("servers", {})
    return s.get("sam3", {}).get("port", 8766), s.get("keypoints", {}).get("port", 8767)


def needed_servers(cfg: dict) -> set[str]:
    kinds = {f["kind"] for f in cfg["functions"]}
    return ({"sam3"} if kinds & {"sam3-detect", "sam3-interact"} else set()) | ({"keypoints"} if "keypoints" in kinds else set())


# --------------------------------------------------------------------------- model servers
def supervise(a):
    """Foreground: run the model servers and restart them if they exit."""
    cfg = load_config(a.config)
    p_sam3, p_kp = ports(cfg)
    want = needed_servers(cfg)
    jobs = {}
    if "sam3" in want:
        jobs["sam3"] = [sam3_python(), "-m", "autolabel.sam3_server", "--port", str(p_sam3)]
    if "keypoints" in want:
        jobs["keypoints"] = [PY, "-m", "autolabel.kp_server", "--port", str(p_kp)]
    LOGS.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1")
    procs: dict[str, subprocess.Popen] = {}

    def launch(name):
        log = open(LOGS / f"{name}.log", "a")
        log.write(f"\n=== start {time.ctime()} ===\n")
        log.flush()
        procs[name] = subprocess.Popen(jobs[name], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)

    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True
        for p in procs.values():
            p.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for name in jobs:
        launch(name)
    print(f"Running {', '.join(jobs)}; logs in {LOGS}", flush=True)
    while not stopping:
        time.sleep(5)
        for name, p in list(procs.items()):
            if p.poll() is not None and not stopping:
                print(f"{name} exited ({p.returncode}); restarting in 10 s", flush=True)
                time.sleep(10)
                launch(name)
    for p in procs.values():
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            p.kill()


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def running_pid() -> int | None:
    if PIDFILE.exists():
        pid = int(PIDFILE.read_text())
        if alive(pid):
            return pid
    return None


def cmd_start(a):
    if running_pid():
        sys.exit(f"Already running (pid {running_pid()}). See ./autolabel.sh status")
    if service_installed():
        sys.exit("The login service already runs the servers. Use ./autolabel.sh status")
    LOGS.mkdir(parents=True, exist_ok=True)
    out = open(LOGS / "supervisor.log", "a")
    p = subprocess.Popen([PY, "-m", "autolabel.cli", "--config", str(a.config), "run"], cwd=ROOT,
                         env=dict(os.environ, PYTHONPATH=str(ROOT)), stdout=out, stderr=subprocess.STDOUT,
                         start_new_session=True)
    PIDFILE.write_text(str(p.pid))
    print(f"Started (pid {p.pid}). SAM3 takes ~10-30 s to load (first run downloads 3.4 GB).")
    print("Check with: ./autolabel.sh status")


def cmd_stop(a):
    pid = running_pid()
    if not pid:
        print("Not running" + (" (the login service manages the servers)" if service_installed() else "") + ".")
        PIDFILE.unlink(missing_ok=True)
        return
    os.kill(pid, signal.SIGTERM)
    PIDFILE.unlink(missing_ok=True)
    print("Stopped.")


def health(port: int):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            return json.loads(r.read())
    except OSError:
        return None


def cmd_status(a):
    cfg = load_config(a.config)
    p_sam3, p_kp = ports(cfg)
    pid = running_pid()
    print(f"Servers     : {'running (pid %d)' % pid if pid else 'login service' if service_installed() else 'stopped'}")
    for name, port in (("sam3", p_sam3), ("keypoints", p_kp)):
        if name in needed_servers(cfg):
            h = health(port)
            print(f"  {name:<10}: {'ready ' + json.dumps(h) if h and h.get('ok') else 'loading / not answering'} (:{port})")
    try:
        deployed = nuclio_list(a.cvat_container)
        mine = {f["id"] for f in cfg["functions"]}
        print("CVAT models :")
        for fid in sorted(mine):
            st = deployed.get(fid)
            print(f"  {fid:<24} {st or 'not deployed'}")
    except Exception as e:  # noqa: BLE001
        print(f"CVAT models : cannot query Nuclio ({e})")
    print(f"Logs        : {LOGS}")


# --------------------------------------------------------------------------- CVAT / Nuclio
NUCLIO_SCRIPT = r'''
import json, sys, time, requests
B = "http://nuclio:8070/api"
H = {"x-nuclio-project-name": "cvat", "Content-Type": "application/json"}
op = sys.argv[1]
if op == "list":
    r = requests.get(B + "/functions", headers=H).json()
    print(json.dumps({k: v.get("status", {}).get("state") for k, v in r.items()}))
    sys.exit(0)
requests.post(B + "/projects", json={"metadata": {"name": "cvat", "namespace": "nuclio"}, "spec": {}}, headers=H)
items = json.load(sys.stdin)
for f in items:
    n = f["metadata"]["name"]
    requests.delete(B + "/functions", json={"metadata": {"name": n, "namespace": "nuclio"}}, headers=H)
if op == "delete":
    print("removed", len(items)); sys.exit(0)
time.sleep(3)
for f in items:
    r = requests.post(B + "/functions", json=f, headers=H)
    print(f["metadata"]["name"], "deploy request:", r.status_code, r.text[:200], flush=True)
deadline = time.time() + 900
pending = {f["metadata"]["name"] for f in items}
while pending and time.time() < deadline:
    time.sleep(5)
    for n in list(pending):
        st = requests.get(B + "/functions/" + n, headers=H).json().get("status", {})
        if st.get("state") in ("ready", "error", "unhealthy"):
            print(n, "->", st.get("state"), st.get("message", "")[:300], flush=True)
            pending.discard(n)
for n in pending:
    print(n, "-> still building (check later with ./autolabel.sh status)")
'''


def nuclio(op: str, container: str, payload=None, capture=False):
    return subprocess.run(["docker", "exec", "-i", container, "python3", "-c", NUCLIO_SCRIPT, op],
                          input=json.dumps(payload or []), text=True, capture_output=capture)


def nuclio_list(container: str) -> dict:
    r = nuclio("list", container, capture=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "docker exec failed")
    return json.loads(r.stdout)


def ensure_base_image(tag: str) -> None:
    """Nuclio builds the proxies on this image. Create it once from python:3.11-slim (amd64, like Nuclio)."""
    if subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode == 0:
        return
    if subprocess.run(["docker", "image", "inspect", "cvat.sam3.base:amd64"], capture_output=True).returncode == 0:
        src = "cvat.sam3.base:amd64"
    else:
        print("Pulling python:3.11-slim (linux/amd64) for the CVAT model proxies ...")
        subprocess.run(["docker", "pull", "--platform", "linux/amd64", "python:3.11-slim"], check=True)
        src = "python:3.11-slim"
    subprocess.run(["docker", "tag", src, tag], check=True)


def function_def(f: dict, cfg: dict) -> dict:
    from autolabel import skeletons, vocab

    p_sam3, p_kp = ports(cfg)
    kind = f["kind"]
    ann = {"name": f["name"], "framework": "autolabel"}
    if kind == "sam3-detect":
        labels = f.get("labels") or vocab.resolve("", f.get("preset", "coco"))
        out = f.get("output", "box")
        shape = {"box": "rectangle", "mask": "mask", "polygon": "polygon"}[out]
        ann.update(type="detector", spec=json.dumps([{"id": i, "name": l, "type": shape} for i, l in enumerate(labels)]))
        url, path, params = f"http://host.docker.internal:{p_sam3}", "/detect", {"labels": labels, "output": out}
    elif kind == "sam3-interact":
        ann.update(type="interactor", version="2", min_pos_points="0", min_neg_points="0",
                   startswith_box_optional="true",
                   help_message="Click positive / negative points or draw a box; SAM3 returns a mask")
        url, path, params = f"http://host.docker.internal:{p_sam3}", "/interact", {}
    elif kind == "keypoints":
        sk = f.get("skeleton", "body17")
        raw = skeletons.cvat_raw_label(sk)
        label = {"id": 0, "name": raw["name"], "type": "skeleton", "svg": raw["svg"],
                 "sublabels": [{"id": i + 1, "name": s["name"], "type": "points"} for i, s in enumerate(raw["sublabels"])],
                 "attributes": [{"name": x["name"], "input_type": x["input_type"], "values": x["values"]}
                                for x in raw["attributes"]]}
        ann.update(type="detector", spec=json.dumps([label]))
        url, path = f"http://host.docker.internal:{p_kp}", "/detect"
        params = {k: f[k] for k in ("auto_rotate", "mode", "kp_thr", "out_thr", "face_thr") if k in f}
        params["kind"] = sk
    else:
        raise ValueError(f"unknown kind {kind!r} in {f['id']}")
    src = base64.b64encode((ROOT / "nuclio/main.py").read_bytes()).decode()
    return {
        "metadata": {"name": f["id"], "namespace": "nuclio", "labels": {"nuclio.io/project-name": "cvat"},
                     "annotations": ann},
        "spec": {
            "description": f"AutoLabel-Tool {kind} (proxy to the host model server)",
            "runtime": "python:3.11", "handler": "main:handler", "eventTimeout": "600s",
            "env": [{"name": "AL_URL", "value": url}, {"name": "AL_PATH", "value": path},
                    {"name": "AL_PARAMS", "value": json.dumps(params)}],
            "build": {"functionSourceCode": src, "baseImage": cfg.get("base_image", "autolabel.base:amd64"),
                      "image": f"autolabel.{f['id']}:latest"},
            "triggers": {"http": {"kind": "http", "numWorkers": 1, "workerAvailabilityTimeoutMilliseconds": 600000,
                                  "attributes": {"maxRequestBodySize": 134217728}}},
            "platform": {"attributes": {"network": a_network(), "restartPolicy": {"name": "always", "maximumRetryCount": 3}}},
            "resources": {}, "minReplicas": 1, "maxReplicas": 1,
        },
    }


def a_network() -> str:
    return os.environ.get("AUTOLABEL_CVAT_NETWORK", "cvat_cvat")


def cmd_deploy(a):
    cfg = load_config(a.config)
    fns = [f for f in cfg["functions"] if not a.only or f["id"] in a.only]
    if not fns:
        sys.exit("Nothing to deploy (check --only / functions.yaml).")
    ensure_base_image(cfg.get("base_image", "autolabel.base:amd64"))
    defs = [function_def(f, cfg) for f in fns]
    print(f"Deploying {len(defs)} model(s) to CVAT's Nuclio (first build takes a few minutes) ...")
    r = nuclio("deploy", a.cvat_container, defs)
    if r.returncode == 0:
        print("Done. In CVAT: Models page, or Actions > Automatic annotation. Keep the servers running: ./autolabel.sh start")
    sys.exit(r.returncode)


def cmd_undeploy(a):
    cfg = load_config(a.config)
    defs = [{"metadata": {"name": f["id"]}} for f in cfg["functions"] if not a.only or f["id"] in a.only]
    sys.exit(nuclio("delete", a.cvat_container, defs).returncode)


def cmd_labels(a):
    from autolabel import skeletons

    kind = {"pose": "body17", "face68": "face68", "face13": "face13", "hand": "hand21"}[a.kind]
    print(json.dumps([skeletons.cvat_raw_label(kind)], indent=2))


# --------------------------------------------------------------------------- login service
PLIST = Path.home() / "Library/LaunchAgents/com.autolabel.servers.plist"
UNIT = Path.home() / ".config/systemd/user/autolabel.service"


def service_installed() -> bool:
    return PLIST.exists() if sys.platform == "darwin" else UNIT.exists()


def cmd_install_service(a):
    if running_pid():
        cmd_stop(a)
    LOGS.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":
        PLIST.parent.mkdir(parents=True, exist_ok=True)
        PLIST.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.autolabel.servers</string>
  <key>ProgramArguments</key><array>
    <string>{PY}</string><string>-m</string><string>autolabel.cli</string>
    <string>--config</string><string>{a.config}</string><string>run</string></array>
  <key>WorkingDirectory</key><string>{ROOT}</string>
  <key>EnvironmentVariables</key><dict><key>PYTHONPATH</key><string>{ROOT}</string></dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>{LOGS}/supervisor.log</string>
  <key>StandardErrorPath</key><string>{LOGS}/supervisor.log</string>
</dict></plist>
""")
        subprocess.run(["launchctl", "unload", str(PLIST)], capture_output=True)
        subprocess.run(["launchctl", "load", str(PLIST)], check=True)
        print(f"Installed {PLIST}: the model servers now start at login. Remove: ./autolabel.sh uninstall-service")
    else:
        UNIT.parent.mkdir(parents=True, exist_ok=True)
        UNIT.write_text(f"""[Unit]
Description=AutoLabel-Tool model servers for CVAT
[Service]
WorkingDirectory={ROOT}
Environment=PYTHONPATH={ROOT}
ExecStart={PY} -m autolabel.cli --config {a.config} run
Restart=always
[Install]
WantedBy=default.target
""")
        subprocess.run(["systemctl", "--user", "daemon-reload"])
        subprocess.run(["systemctl", "--user", "enable", "--now", "autolabel.service"])
        print(f"Installed {UNIT}. Remove: ./autolabel.sh uninstall-service")


def cmd_uninstall_service(a):
    if sys.platform == "darwin":
        subprocess.run(["launchctl", "unload", str(PLIST)], capture_output=True)
        PLIST.unlink(missing_ok=True)
    else:
        subprocess.run(["systemctl", "--user", "disable", "--now", "autolabel.service"])
        UNIT.unlink(missing_ok=True)
    print("Service removed.")


APP = Path.home() / "Applications/AutoLabel.app"
MENUBAR_LABEL = "com.autolabel.menubar"
MENUBAR_PLIST = Path.home() / f"Library/LaunchAgents/{MENUBAR_LABEL}.plist"


def write_menubar_agent(run_at_load: bool) -> None:
    """launchd job for the menu bar app. run_at_load = open it at login. No KeepAlive, so Quit works."""
    MENUBAR_PLIST.parent.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    MENUBAR_PLIST.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>{MENUBAR_LABEL}</string>
  <key>ProgramArguments</key><array>
    <string>{PY}</string><string>-m</string><string>autolabel.menubar</string></array>
  <key>WorkingDirectory</key><string>{ROOT}</string>
  <key>EnvironmentVariables</key><dict><key>PYTHONPATH</key><string>{ROOT}</string></dict>
  <key>RunAtLoad</key>{'<true/>' if run_at_load else '<false/>'}
  <key>ProcessType</key><string>Interactive</string>
  <key>StandardOutPath</key><string>{LOGS}/menubar.log</string>
  <key>StandardErrorPath</key><string>{LOGS}/menubar.log</string>
</dict></plist>
""")


def menubar_opens_at_login() -> bool:
    return MENUBAR_PLIST.exists() and "<key>RunAtLoad</key><true/>" in MENUBAR_PLIST.read_text()


def _app_icon(dst: Path) -> None:
    """Build AppIcon.icns: rounded square with a white label tag."""
    import tempfile

    from PIL import Image, ImageDraw

    tmp = Path(tempfile.mkdtemp()) / "AppIcon.iconset"
    tmp.mkdir()

    def render(size: int) -> "Image.Image":
        im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle((size * 0.06, size * 0.06, size * 0.94, size * 0.94), radius=size * 0.22,
                            fill=(37, 99, 235, 255))
        s = size / 44
        d.polygon([(8 * s, 11 * s), (24 * s, 11 * s), (37 * s, 24 * s), (24 * s, 37 * s), (8 * s, 24 * s)],
                  fill=(255, 255, 255, 255))
        d.ellipse((13 * s, 16 * s, 18 * s, 21 * s), fill=(37, 99, 235, 255))
        return im

    for base in (16, 32, 128, 256, 512):  # the file names iconutil expects
        render(base).save(tmp / f"icon_{base}x{base}.png")
        render(base * 2).save(tmp / f"icon_{base}x{base}@2x.png")
    subprocess.run(["iconutil", "-c", "icns", str(tmp), "-o", str(dst)], check=True)


def cmd_app(a):
    """Create ~/Applications/AutoLabel.app (menu bar app, macOS)."""
    if sys.platform != "darwin":
        sys.exit("The menu bar app is macOS only. On Linux use ./autolabel.sh install-service")
    contents = APP / "Contents"
    (contents / "MacOS").mkdir(parents=True, exist_ok=True)
    (contents / "Resources").mkdir(parents=True, exist_ok=True)
    (contents / "Info.plist").write_text("""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>AutoLabel</string>
  <key>CFBundleDisplayName</key><string>AutoLabel</string>
  <key>CFBundleIdentifier</key><string>com.autolabel.menubar</string>
  <key>CFBundleVersion</key><string>0.1.0</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>AutoLabel</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>LSUIElement</key><true/>
</dict></plist>
""")
    exe = contents / "MacOS/AutoLabel"
    # The menu bar runs as a launchd job, not inside this unsigned bundle: macOS (Tahoe) does not show
    # status items of a script-launched bundle, and kills its children when the launcher exits.
    write_menubar_agent(run_at_load=True)
    exe.write_text(f"""#!/bin/bash
# Opens the AutoLabel menu bar app (launchd job {MENUBAR_LABEL}).
PLIST="{MENUBAR_PLIST}"
launchctl print "gui/$(id -u)/{MENUBAR_LABEL}" >/dev/null 2>&1 || launchctl bootstrap "gui/$(id -u)" "$PLIST"
launchctl kickstart "gui/$(id -u)/{MENUBAR_LABEL}"
""")
    exe.chmod(0o755)
    _app_icon(contents / "Resources/AppIcon.icns")
    subprocess.run(["touch", str(APP)])
    print(f"Created {APP}. Open it from Finder / Spotlight; it lives in the menu bar (no Dock icon).")
    if a.open:
        subprocess.run(["open", str(APP)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "functions.yaml")
    ap.add_argument("--cvat-container", default=os.environ.get("AUTOLABEL_CVAT_CONTAINER", "cvat_server"),
                    help="CVAT server container used to reach Nuclio (default cvat_server)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("deploy", "undeploy"):
        p = sub.add_parser(name)
        p.add_argument("--only", nargs="+", help="function ids from functions.yaml")
    for name in ("start", "stop", "status", "run", "install-service", "uninstall-service"):
        sub.add_parser(name)
    p = sub.add_parser("labels")
    p.add_argument("kind", choices=["pose", "face68", "face13", "hand"])
    p = sub.add_parser("app", help="create ~/Applications/AutoLabel.app (macOS menu bar app)")
    p.add_argument("--open", action="store_true", help="open the app after creating it")
    a = ap.parse_args()
    a.config = a.config.resolve()
    {"deploy": cmd_deploy, "undeploy": cmd_undeploy, "start": cmd_start, "stop": cmd_stop, "status": cmd_status,
     "run": supervise, "labels": cmd_labels, "install-service": cmd_install_service,
     "uninstall-service": cmd_uninstall_service, "app": cmd_app}[a.cmd](a)


if __name__ == "__main__":
    main()
