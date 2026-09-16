#!/usr/bin/env python3

import argparse
import contextlib
import ctypes
import os
import subprocess
import sys
from ctypes import c_char_p, c_int, c_uint32

IN_CLOSE_WRITE = 0x00000008
IN_CREATE = 0x00000100
IN_MOVED_TO = 0x00000080
IN_ONLYDIR = 0x01000000
IN_NONBLOCK = 0x00000800
WATCH_MASK = IN_CREATE | IN_MOVED_TO | IN_CLOSE_WRITE


class InotifyWatcher:
    def __init__(self, path: str) -> None:
        self._libc = ctypes.CDLL("libc.so.6", use_errno=True)
        self._libc.inotify_init1.argtypes = [c_int]
        self._libc.inotify_init1.restype = c_int
        self._libc.inotify_add_watch.argtypes = [c_int, c_char_p, c_uint32]
        self._libc.inotify_add_watch.restype = c_int
        self.fd = self._libc.inotify_init1(IN_NONBLOCK)
        if self.fd < 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))
        watch_result = self._libc.inotify_add_watch(
            self.fd, path.encode("utf-8"), WATCH_MASK | IN_ONLYDIR
        )
        if watch_result < 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err), path)

    def fileno(self) -> int:
        return self.fd

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


def final_export_ready(final_dir: str) -> bool:
    return os.path.exists(os.path.join(final_dir, "model.pt")) and os.path.exists(
        os.path.join(final_dir, "config.json")
    )


def build_upload_command(args: argparse.Namespace) -> list[str]:
    pieces = [
        sys.executable,
        "-m",
        "training.model_embeddings.multimodal.large.upload",
        "--final-dir",
        args.final_dir,
        "--repo-id",
        args.repo_id,
        "--token-name",
        args.token_name,
    ]
    if args.max_samples is not None:
        pieces.extend(["--max-samples", str(args.max_samples)])
    if args.min_eval_top1 is not None:
        pieces.extend(["--min-eval-top1", str(args.min_eval_top1)])
    if args.max_eval_loss is not None:
        pieces.extend(["--max-eval-loss", str(args.max_eval_loss)])
    return pieces


def run_upload(args: argparse.Namespace) -> int:
    command = build_upload_command(args)
    print("[watcher] final export detected, launching module uploader", flush=True)
    result = subprocess.run(command, check=False)
    return result.returncode


def wait_for_final(args: argparse.Namespace) -> int:
    if final_export_ready(args.final_dir):
        print("[watcher] final export already present", flush=True)
        return run_upload(args)

    parent_dir = os.path.dirname(args.final_dir.rstrip(os.sep))
    os.makedirs(parent_dir, exist_ok=True)
    watcher = InotifyWatcher(parent_dir)
    try:
        print(f"[watcher] waiting for final export under {args.final_dir}", flush=True)
        while True:
            if final_export_ready(args.final_dir):
                return run_upload(args)
            os.read(watcher.fileno(), 4096)
    except BlockingIOError:
        import select  # noqa: PLC0415 - Linux fallback loaded only when needed

        poller = select.poll()
        poller.register(watcher.fileno(), select.POLLIN)
        while True:
            if final_export_ready(args.final_dir):
                return run_upload(args)
            poller.poll()
            with contextlib.suppress(BlockingIOError):
                os.read(watcher.fileno(), 4096)
    finally:
        watcher.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for the tri-encoder final export, then run gated eval and Hugging Face upload."
    )
    parser.add_argument(
        "--final-dir",
        required=True,
        help="Final export directory to watch for model.pt and config.json",
    )
    parser.add_argument(
        "--repo-id",
        default="llm-semantic-router/multi-modal-embed-large",
        help="Destination Hugging Face model repository",
    )
    parser.add_argument(
        "--token-name",
        default="model training",
        help="Named entry in ~/.cache/huggingface/stored_tokens used by the uploader",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optionally evaluate only the first N validation samples before upload",
    )
    parser.add_argument(
        "--min-eval-top1",
        type=float,
        default=0.85,
        help="Minimum eval_top1 required before upload",
    )
    parser.add_argument(
        "--max-eval-loss",
        type=float,
        default=0.45,
        help="Maximum eval_loss allowed before upload",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exit_code = wait_for_final(args)
    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
