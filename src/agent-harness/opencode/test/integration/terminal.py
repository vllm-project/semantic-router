"""Drive the installed native TUI through a private PTY using only fixture input."""

import fcntl
import json
import os
import pty
import re
import select
import signal
import struct
import subprocess
import sys
import termios
import time


def main():
    config = json.load(sys.stdin)
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 45, 160, 0, 0))
    env = {**os.environ, "TERM": "xterm-256color"}
    child = subprocess.Popen(
        ["opencode", "--server", config["url"], "--session", config["sessionID"]],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env=env,
        start_new_session=True,
    )
    os.close(slave)
    data = bytearray()
    started = time.monotonic()
    ready = None
    steps = list(config["steps"])
    try:
        while time.monotonic() - (ready or started) < config.get("duration", 15):
            if ready is None and b"vSR" in data:
                ready = time.monotonic()
            elapsed = (time.monotonic() - (ready or started)) * 1000
            while ready is not None and steps and elapsed >= steps[0]["at"]:
                step = steps.pop(0)
                if "resize" in step:
                    fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", *step["resize"], 0, 0))
                    os.killpg(child.pid, signal.SIGWINCH)
                if "keys" in step:
                    os.write(master, step["keys"].encode())
            if child.poll() is not None:
                break
            if select.select([master], [], [], 0.05)[0]:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                data.extend(chunk)
                if b"\x1b[6n" in chunk:
                    os.write(master, b"\x1b[1;1R")
    finally:
        child.terminate()
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
        os.close(master)
    text = data.decode(errors="replace")
    text = re.sub(r"\x1b\].*?(?:\x07|\x1b\\)", "", text, flags=re.S)
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    print(json.dumps({"text": text, "exit": child.returncode, "readyAfter": None if ready is None else ready - started}))


if __name__ == "__main__":
    main()
