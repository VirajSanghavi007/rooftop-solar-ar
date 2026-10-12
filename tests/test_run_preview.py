"""Smoke test: run_preview.bat serves the preview page and both models."""
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows batch file")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_run_preview_bat_serves_page_and_models():
    port = _free_port()
    env = {**os.environ, "PORT": str(port), "NO_BROWSER": "1"}
    proc = subprocess.Popen(["cmd", "/c", str(ROOT / "run_preview.bat")], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        base = f"http://127.0.0.1:{port}/"
        for _ in range(100):
            try:
                html = urllib.request.urlopen(base, timeout=1).read().decode()
                break
            except OSError:
                time.sleep(0.1)
        else:
            pytest.fail(f"server never answered: {proc.stdout.read1(4000) if proc.poll() is not None else ''}")
        assert "<title>Terrace Solar Preview</title>" in html
        for name in ("terrace_tilted.glb", "terrace_flat.glb"):
            assert name in html
            assert urllib.request.urlopen(base + name, timeout=2).read()[:4] == b"glTF"
    finally:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
