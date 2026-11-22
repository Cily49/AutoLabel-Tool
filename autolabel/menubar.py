"""AutoLabel menu bar app (macOS): status of the model servers and CVAT models, start/stop,
start at login, deploy, open CVAT. Built into AutoLabel.app by `./autolabel.sh app`.
"""

from __future__ import annotations

import argparse
import subprocess
import threading
from pathlib import Path

import AppKit
import rumps
from PIL import Image, ImageDraw

from autolabel import cli

CONFIG = cli.ROOT / "functions.yaml"


def args():
    return argparse.Namespace(config=CONFIG, cvat_container="cvat_server", only=None)


def make_icon() -> str:
    """Template icon (black tag shape) for the menu bar; macOS recolours it for light/dark mode."""
    path = cli.HOME / "menubar_icon.png"
    if not path.exists():
        cli.HOME.mkdir(parents=True, exist_ok=True)
        s = 44
        im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.polygon([(6, 8), (24, 8), (40, 24), (24, 40), (6, 22)], fill=(0, 0, 0, 255))
        d.ellipse((12, 13, 18, 19), fill=(0, 0, 0, 0))
        im.save(path)
    return str(path)


class AutoLabelApp(rumps.App):
    def __init__(self):
        super().__init__("AutoLabel", title=None, icon=make_icon(), template=True, quit_button=None)
        self.cfg = cli.load_config(CONFIG)
        self.st_sam3 = rumps.MenuItem("SAM3 server: …")
        self.st_kp = rumps.MenuItem("Keypoint server: …")
        self.st_cvat = rumps.MenuItem("CVAT models: …")
        self.toggle = rumps.MenuItem("Start model servers", callback=self.on_toggle)
        self.login = rumps.MenuItem("Start servers at login", callback=self.on_login)
        self.app_login = rumps.MenuItem("Open this menu at login", callback=self.on_app_login)
        self.app_login.state = 1 if cli.menubar_opens_at_login() else 0
        self.menu = [
            self.st_sam3, self.st_kp, self.st_cvat, None,
            self.toggle, self.login, self.app_login, None,
            rumps.MenuItem("Open CVAT", callback=lambda _: self.open_url("")),
            rumps.MenuItem("Open CVAT Models page", callback=lambda _: self.open_url("/models")),
            rumps.MenuItem("Deploy models to CVAT", callback=self.on_deploy),
            rumps.MenuItem("Edit models (functions.yaml)", callback=lambda _: subprocess.run(["open", "-t", str(CONFIG)])),
            rumps.MenuItem("Open logs", callback=self.on_logs),
            None,
            rumps.MenuItem("Quit AutoLabel menu", callback=lambda _: rumps.quit_application()),
        ]
        self.busy = False
        self._status = {}
        rumps.Timer(self.refresh_ui, 3).start()
        self.poll()

    # ---------------------------------------------------------------- status (polled off the UI thread)
    def poll(self):
        def work():
            p_sam3, p_kp = cli.ports(self.cfg)
            st = {"sam3": cli.health(p_sam3), "kp": cli.health(p_kp), "pid": cli.running_pid(),
                  "service": cli.service_installed()}
            try:
                deployed = cli.nuclio_list("cvat_server")
                ids = [f["id"] for f in self.cfg["functions"]]
                st["models"] = (sum(deployed.get(i) == "ready" for i in ids), len(ids))
            except Exception:  # noqa: BLE001 - CVAT / Docker not running
                st["models"] = None
            self._status = st
            threading.Timer(10, self.poll).start()

        threading.Thread(target=work, daemon=True).start()

    def refresh_ui(self, _=None):
        st = self._status
        if not st:
            return
        sam3, kp = st.get("sam3"), st.get("kp")
        self.st_sam3.title = f"{'🟢' if sam3 and sam3.get('ok') else '⚪️'} SAM3 server: " + (
            f"ready ({sam3.get('device')})" if sam3 and sam3.get("ok") else "stopped / loading")
        self.st_kp.title = f"{'🟢' if kp and kp.get('ok') else '⚪️'} Keypoint server: " + (
            "ready" if kp and kp.get("ok") else "stopped / loading")
        m = st.get("models")
        self.st_cvat.title = ("🔴 CVAT: not reachable (is Docker running?)" if m is None
                              else f"{'🟢' if m[0] == m[1] else '🟡'} CVAT models: {m[0]}/{m[1]} ready")
        running = bool(st.get("pid")) or st.get("service")
        self.login.state = 1 if st.get("service") else 0
        if st.get("service"):
            self.toggle.title = "Servers run at login (turn off below to stop)"
        else:
            self.toggle.title = "Stop model servers" if running else "Start model servers"
        ready = bool(sam3 and sam3.get("ok")) and bool(kp and kp.get("ok")) and m is not None and m[0] == m[1]
        # keep the item narrow (menu bars with a notch have little room): text only when it needs attention
        self.title = "…" if self.busy else (None if ready else "!")

    # ---------------------------------------------------------------- actions
    def on_toggle(self, _):
        if cli.service_installed():
            return
        try:
            if cli.running_pid():
                cli.cmd_stop(args())
            else:
                cli.cmd_start(args())
                rumps.notification("AutoLabel", "Starting model servers", "SAM3 needs about 10-30 s to load.")
        except SystemExit as e:
            rumps.alert("AutoLabel", str(e))
        self.poll()

    def on_login(self, sender):
        try:
            if sender.state:
                cli.cmd_uninstall_service(args())
            else:
                cli.cmd_install_service(args())
        except Exception as e:  # noqa: BLE001
            rumps.alert("AutoLabel", f"Could not change the login service: {e}")
        self.poll()

    def on_app_login(self, sender):
        cli.write_menubar_agent(run_at_load=not sender.state)
        sender.state = 0 if sender.state else 1

    def on_deploy(self, _):
        if self.busy:
            return
        self.busy = True
        rumps.notification("AutoLabel", "Deploying models to CVAT", "The first build takes a few minutes.")

        def work():
            log = cli.LOGS / "deploy.log"
            cli.LOGS.mkdir(parents=True, exist_ok=True)
            with open(log, "w") as fh:
                r = subprocess.run([str(cli.ROOT / "autolabel.sh"), "deploy"], stdout=fh, stderr=subprocess.STDOUT)
            self.busy = False
            ok = r.returncode == 0 and "error" not in log.read_text().lower()
            rumps.notification("AutoLabel", "Deploy finished" if ok else "Deploy had problems",
                               "Models are ready in CVAT." if ok else f"See {log}")
            self.poll()

        threading.Thread(target=work, daemon=True).start()

    def on_logs(self, _):
        cli.LOGS.mkdir(parents=True, exist_ok=True)
        subprocess.run(["open", str(cli.LOGS)])

    def open_url(self, path: str):
        subprocess.run(["open", self.cfg.get("cvat_url", "http://localhost:8080").rstrip("/") + path])


AUTOSAVE = "AutoLabel"
# Distance from the right edge of the screen (points). New status items normally go to the far left,
# which is hidden behind the notch when the menu bar is busy; ask for a spot near the clock instead.
PREFERRED_POSITION = 260


def place_status_item(app: AutoLabelApp) -> None:
    defaults = AppKit.NSUserDefaults.standardUserDefaults()
    key = f"NSStatusItem Preferred Position {AUTOSAVE}"
    if defaults.objectForKey_(key) is None:  # keep the place the user dragged it to (⌘-drag)
        defaults.setDouble_forKey_(PREFERRED_POSITION, key)
    defaults.setBool_forKey_(True, f"NSStatusItem Visible {AUTOSAVE}")
    app._nsapp.nsstatusitem.setAutosaveName_(AUTOSAVE)


def main():
    # menu bar only: no Dock icon, no app switcher entry
    AppKit.NSApplication.sharedApplication().setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
    app = AutoLabelApp()
    rumps.Timer(lambda t: (t.stop(), place_status_item(app)), 0.5).start()
    app.run()


if __name__ == "__main__":
    main()
