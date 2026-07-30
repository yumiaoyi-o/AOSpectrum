"""Local HTTP server for generated Band and Orbital artifacts."""

from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from aospectrum.errors import ArtifactError


def artifact_entry(root: str | Path) -> tuple[Path, str]:
    source = Path(root).expanduser().resolve()
    entry = "index.html"
    if not (source / entry).is_file():
        raise ArtifactError(f"{source} does not contain index.html")
    return source, entry


def serve_artifact(
    root: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    source, entry = artifact_entry(root)
    handler = partial(SimpleHTTPRequestHandler, directory=str(source))
    server = ThreadingHTTPServer((host, int(port)), handler)
    actual_host, actual_port = server.server_address[:2]
    print(f"Serving {source}")
    print(f"http://{actual_host}:{actual_port}/{entry}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
