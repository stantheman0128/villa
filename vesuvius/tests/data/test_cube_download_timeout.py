"""Cube.load_data must time out and retry, not park on a stalled download.

The zarr read path retries transient failures. The NRRD download in
Cube.load_data called requests.get with no timeout, so a connection that
accepted the request and then stopped sending left the process blocked.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
import requests

import vesuvius.data.volume as volume_module
from vesuvius.data.volume import Cube


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(volume_module.time, "sleep", lambda _s: None)


def _cube(url: str, *, cache: bool = False, cache_dir=None) -> Cube:
    cube = Cube.__new__(Cube)
    cube.aws = False
    cube.cache = cache
    cube.volume_url = url
    cube.mask_url = url
    cube.z = 1
    cube.y = 2
    cube.x = 3
    if cache:
        cube.cache_dir = cache_dir
    return cube


def _serve(handler) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _url(server: ThreadingHTTPServer, name: str) -> str:
    host, port = server.server_address
    return f"http://{host}:{port}/{name}"


def _nrrd_bytes(tmp_path) -> bytes:
    array = np.zeros((2, 2, 2), dtype=np.uint8)
    array[0, 0, 0] = 7
    path = tmp_path / "sample.nrrd"
    import nrrd

    nrrd.write(str(path), array)
    return path.read_bytes()


def test_both_download_paths_pass_a_timeout(monkeypatch, tmp_path):
    timeouts = []

    def spy(url, **kwargs):
        timeouts.append(kwargs.get("timeout"))
        raise requests.Timeout("timed out")

    monkeypatch.setattr(volume_module.requests, "get", spy)
    with pytest.raises(requests.Timeout):
        _cube("http://example.invalid/volume.nrrd").load_data()
    with pytest.raises(requests.Timeout):
        _cube(
            "http://example.invalid/volume.nrrd",
            cache=True,
            cache_dir=tmp_path,
        ).load_data()
    assert timeouts
    assert all(timeout is not None and timeout > 0 for timeout in timeouts)


def test_stalled_download_raises(monkeypatch):
    monkeypatch.setattr(volume_module, "_CUBE_DOWNLOAD_TIMEOUT", 0.25, raising=False)
    monkeypatch.setattr(volume_module, "_CUBE_DOWNLOAD_RETRIES", 1, raising=False)
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "100000")
            self.end_headers()
            self.wfile.write(b"partial")
            self.wfile.flush()
            release.wait(30)

        def log_message(self, format, *args):
            return

    server = _serve(Handler)
    errors = []

    def run():
        try:
            _cube(_url(server, "stall.nrrd")).load_data()
        except BaseException as exc:  # noqa: BLE001 - the assertion checks the type
            errors.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    try:
        worker.start()
        worker.join(1.5)
        still_blocked = worker.is_alive()
    finally:
        release.set()
        worker.join(2)
        server.shutdown()
        server.server_close()
    assert not still_blocked
    assert errors
    # urllib3 reports the stall as ConnectionError(ReadTimeoutError("Read timed out")).
    assert isinstance(errors[0], requests.RequestException)
    assert "timed out" in str(errors[0]).lower()


def test_service_unavailable_is_retried(monkeypatch, tmp_path):
    monkeypatch.setattr(volume_module, "_CUBE_DOWNLOAD_RETRIES", 2, raising=False)
    blob = _nrrd_bytes(tmp_path)
    hits = {"n": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits["n"] += 1
            # volume, then mask: each gets one 503 and then the file.
            if hits["n"] in (1, 3):
                self.send_response(503)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"unavailable")
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

        def log_message(self, format, *args):
            return

    server = _serve(Handler)
    try:
        volume, mask = _cube(_url(server, "cube.nrrd")).load_data()
    finally:
        server.shutdown()
        server.server_close()
    assert hits["n"] == 4
    assert int(volume[0, 0, 0]) == 7
    assert int(mask[0, 0, 0]) == 7


def test_cache_download_reads_the_written_file(monkeypatch, tmp_path):
    monkeypatch.setattr(volume_module, "_CUBE_DOWNLOAD_RETRIES", 1, raising=False)
    blob = _nrrd_bytes(tmp_path)
    cache_dir = tmp_path / "cache"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

        def log_message(self, format, *args):
            return

    server = _serve(Handler)
    try:
        volume, mask = _cube(
            _url(server, "cube.nrrd"),
            cache=True,
            cache_dir=cache_dir,
        ).load_data()
    finally:
        server.shutdown()
        server.server_close()
    assert int(volume[0, 0, 0]) == 7
    assert int(mask[0, 0, 0]) == 7
    assert (cache_dir / "00001_00002_00003" / "cube.nrrd").is_file()


def test_not_found_is_not_retried(monkeypatch):
    monkeypatch.setattr(volume_module, "_CUBE_DOWNLOAD_RETRIES", 4, raising=False)
    hits = {"n": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits["n"] += 1
            self.send_response(404)
            self.end_headers()

        def log_message(self, format, *args):
            return

    server = _serve(Handler)
    try:
        with pytest.raises(requests.HTTPError):
            _cube(_url(server, "missing.nrrd")).load_data()
    finally:
        server.shutdown()
        server.server_close()
    assert hits["n"] == 1


def test_retries_below_one_still_attempts_once(monkeypatch):
    calls = {"n": 0}

    def spy(url, **kwargs):
        calls["n"] += 1
        raise requests.Timeout("timed out")

    monkeypatch.setattr(volume_module.requests, "get", spy)
    with pytest.raises(requests.Timeout):
        volume_module._get_with_retry(
            "http://example.invalid/x.nrrd",
            timeout=0.1,
            retries=0,
        )
    assert calls["n"] == 1
