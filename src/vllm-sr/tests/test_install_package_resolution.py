"""Exercise dev installation against a real pip HTTP cache and local index."""

import io
import os
import subprocess
import threading
import venv
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
OLD_VERSION = "0.3.0.dev20260101000100"
NEW_VERSION = "0.3.0.dev20260102000100"
DEPENDENCY = "installer_cache_dependency"


def _wheel(name: str, version: str, dependency: str = "") -> bytes:
    metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    if dependency:
        metadata += f"Requires-Dist: {dependency}==1.0.0\n"
    dist_info = f"{name}-{version}.dist-info"
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(f"{dist_info}/METADATA", metadata)
        archive.writestr(
            f"{dist_info}/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(f"{dist_info}/RECORD", "")
    return output.getvalue()


def test_dev_install_refreshes_cached_catalog_and_keeps_dependency_cache(
    tmp_path: Path,
) -> None:
    published_versions = [OLD_VERSION]
    dependency_downloads = []
    wheels = {
        f"/vllm_sr-{version}-py3-none-any.whl": _wheel("vllm_sr", version, DEPENDENCY)
        for version in (OLD_VERSION, NEW_VERSION)
    }
    dependency_path = f"/{DEPENDENCY}-1.0.0-py3-none-any.whl"
    wheels[dependency_path] = _wheel(DEPENDENCY, "1.0.0")

    class IndexHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/simple/vllm-sr/":
                # Reproduce a stale conditional response: cached clients retain
                # the old catalog while an unconditional fetch sees publication.
                if self.headers.get("If-None-Match") == '"catalog"':
                    self.send_response(304)
                    self.end_headers()
                    return
                paths = [
                    f"/vllm_sr-{version}-py3-none-any.whl"
                    for version in published_versions
                ]
            elif self.path == "/simple/installer-cache-dependency/":
                paths = [dependency_path]
            elif self.path in wheels:
                if self.path == dependency_path:
                    dependency_downloads.append(self.path)
                self._respond(wheels[self.path], "application/octet-stream")
                return
            else:
                self.send_error(404)
                return
            body = "".join(f'<a href="{path}">{path[1:]}</a>' for path in paths)
            self._respond(body.encode(), "text/html")

        def _respond(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "public, max-age=86400")
            self.send_header("ETag", '"catalog"')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), IndexHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        install_root = tmp_path / "install"
        venv.EnvBuilder(with_pip=True).create(install_root / "venv")
        python = install_root / "venv" / "bin" / "python"
        config = tmp_path / "pip.conf"
        config.write_text(
            "[global]\n"
            f"index-url = http://127.0.0.1:{server.server_port}/simple/\n"
            "extra-index-url =\n"
            "trusted-host = 127.0.0.1\n"
            f"cache-dir = {tmp_path / 'pip-cache'}\n"
            "disable-pip-version-check = true\n",
            encoding="utf-8",
        )
        temporary_downloads = tmp_path / "downloads"
        temporary_downloads.mkdir()
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("PIP_") and key != "PYTHONPATH"
        }
        env.update(
            PIP_CONFIG_FILE=str(config),
            TMPDIR=str(temporary_downloads),
            VLLM_SR_INSTALL_ROOT=str(install_root),
            VLLM_SR_INSTALL_CHANNEL="dev",
            VLLM_SR_PIP_SPEC="",
        )

        def run(*args: str) -> subprocess.CompletedProcess[str]:
            result = subprocess.run(
                args, env=env, text=True, capture_output=True, timeout=60, check=False
            )
            assert result.returncode == 0, result.stdout + result.stderr
            return result

        def pip(*args: str) -> str:
            return run(str(python), "-m", "pip", *args).stdout

        assert OLD_VERSION in pip("index", "versions", "--pre", "vllm-sr")
        pip("download", "--no-deps", "--dest", str(tmp_path / "seed"), DEPENDENCY)
        published_versions.append(NEW_VERSION)
        # Establish the failure condition using real pip, without stubbing its
        # flags or resolver: ordinary lookup still returns the cached catalog.
        stale_catalog = pip("index", "versions", "--pre", "vllm-sr")
        assert OLD_VERSION in stale_catalog
        assert NEW_VERSION not in stale_catalog

        source = (REPO_ROOT / "install.sh").read_text(encoding="utf-8")
        testable = tmp_path / "install.testable.sh"
        testable.write_text(
            "\n".join(
                line for line in source.splitlines() if not line.startswith("main ")
            ),
            encoding="utf-8",
        )
        run(
            "bash",
            "-c",
            'source "$1"; install_requested_package',
            "installer",
            str(testable),
        )
        installed = run(
            str(python),
            "-c",
            "from importlib.metadata import version; "
            "print(version('vllm-sr')); print(version('installer-cache-dependency'))",
        ).stdout.splitlines()
        assert installed == [NEW_VERSION, "1.0.0"]
        assert (
            len(dependency_downloads) == 1
        ), "dependency cache was unnecessarily bypassed"
        assert not list(
            temporary_downloads.iterdir()
        ), "temporary downloads were not removed"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
