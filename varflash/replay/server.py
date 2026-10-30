"""Tiny HTTP server for the replay GUI: static page, session list, and session files.

Supports HTTP Range requests, which browsers need to seek inside a video.
"""

from __future__ import annotations

import errno
import json
import logging
import mimetypes
import re
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .build import SESSIONS

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
CHUNK = 1 << 20


def list_sessions(root: Path) -> list[dict]:
    out = []
    for f in sorted(root.glob("*/session.json")):
        try:
            with f.open() as fh:
                head = json.loads(fh.read())
        except (OSError, ValueError):
            continue
        out.append({"name": f.parent.name, "title": head.get("title", f.parent.name), "note": head.get("note", "")})
    return out


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, root: Path, **kw):
        self.root = root
        super().__init__(*args, **kw)

    def log_message(self, fmt, *args):
        log.debug(fmt, *args)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path == "/api/sessions":
            return self._json(list_sessions(self.root))
        if path == "/":
            path = "/static/index.html"
        if path.startswith("/static/"):
            f = self._inside(STATIC, path.removeprefix("/static/"))
        elif path.startswith("/sessions/"):
            f = self._inside(self.root, path.removeprefix("/sessions/"))
        else:
            f = None
        if f is None or not f.is_file():
            return self.send_error(404)
        try:
            self._file(f)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser cancels range requests all the time while seeking

    @staticmethod
    def _inside(base: Path, rel: str) -> Path | None:
        f = (base / rel).resolve()
        return f if f.is_relative_to(base.resolve()) else None

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, f: Path):
        size = f.stat().st_size
        start, end = 0, size - 1
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
            else:
                start = max(size - int(m.group(2)), 0)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        ctype = "video/mp4" if f.suffix == ".mp4" else mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with f.open("rb") as fh:
            fh.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = fh.read(min(CHUNK, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)


def serve(host: str = "127.0.0.1", port: int = 8765, root: Path = SESSIONS) -> None:
    """Login nodes are shared, so if ``port`` is taken, use the next free one."""
    root.mkdir(parents=True, exist_ok=True)
    for p in range(port, port + 50):
        try:
            httpd = ThreadingHTTPServer((host, p), partial(Handler, root=root))
            break
        except OSError as e:
            if e.errno != errno.EADDRINUSE:
                raise
    else:
        raise OSError(f"no free port in {port}-{port + 49}")
    httpd.daemon_threads = True
    print(f"VARFlash replay: http://localhost:{p}/  (sessions in {root})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
