# AutoLabel-Tool proxy function for CVAT (Nuclio). It forwards CVAT's request to a model server
# running natively on the host (SAM3 on Apple Silicon / GPU, RTMW on CPU), adding this
# function's parameters. Standard library only, so any small Python base image works.
import json
import os
import urllib.error
import urllib.request

URL = os.environ.get("AL_URL", "http://host.docker.internal:8766")
PATH = os.environ.get("AL_PATH", "/detect")
PARAMS = json.loads(os.environ.get("AL_PARAMS", "{}"))


def init_context(context):
    context.logger.info(f"AutoLabel proxy -> {URL}{PATH} {PARAMS}")


def handler(context, event):
    data = event.body
    if isinstance(data, (bytes, str)):
        data = json.loads(data)
    data.update(PARAMS)
    req = urllib.request.Request(URL + PATH, json.dumps(data).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            body, status = r.read(), 200
    except urllib.error.HTTPError as e:
        body, status = e.read(), e.code
    except Exception as e:  # server not running
        body = json.dumps({"error": f"AutoLabel model server not reachable at {URL}: {e}. "
                                    "Start it on the host with: ./autolabel.sh start"}).encode()
        status = 503
    return context.Response(body=body, headers={}, content_type="application/json", status_code=status)
