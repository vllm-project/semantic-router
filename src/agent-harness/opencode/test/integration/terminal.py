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
            # The line vSR draws above the prompt; its colored mark splits "vSR" across style codes.
            if ready is None and "decisions · /vsr".encode() in data:
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
    screen = Screen(45, 160, config["steps"])
    frames = screen.replay(data.decode(errors="replace"))
    print(json.dumps({"text": "\n".join(frames), "last": "\n".join(screen.lines()), "ansi": screen.ansi(), "exit": child.returncode, "readyAfter": None if ready is None else ready - started}))


TOKEN = re.compile(r"\x1b[\]P_^X].*?(?:\x07|\x1b\\)|\x1b\[([0-?]*)[ -/]*([@-~])|\x1b[^\[\]]|.", re.S)


BASE16 = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), (192, 192, 192),
          (128, 128, 128), (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255)]


def xterm(n):
    """A 256-color index as `r;g;b`."""
    if n < 16:
        rgb = BASE16[n]
    elif n < 232:
        n -= 16
        rgb = tuple(0 if c == 0 else 55 + 40 * c for c in (n // 36, n // 6 % 6, n % 6))
    else:
        rgb = (8 + 10 * (n - 232),) * 3
    return ";".join(map(str, rgb))


class Screen:
    """The cells the renderer drew; it rewrites only changed cells, so raw output splices words."""

    def __init__(self, rows, cols, steps):
        # The grid holds the largest size the run resizes to, so a redraw after SIGWINCH stays in bounds.
        sizes = [(rows, cols)] + [tuple(step["resize"]) for step in steps if "resize" in step]
        self.resize(max(r for r, _ in sizes), max(c for _, c in sizes))

    def resize(self, rows, cols):
        self.rows, self.cols = rows, cols
        self.cells = [[" "] * cols for _ in range(rows)]
        self.styles = [[(None, None)] * cols for _ in range(rows)]
        self.style = (None, None)
        self.row = self.col = 0

    def lines(self):
        return ["".join(line).rstrip() for line in self.cells]

    def ansi(self):
        """The final screen with its truecolor foreground and background, to look at the colors drawn."""
        out = []
        for chars, styles in zip(self.cells, self.styles):
            line, last = "", None
            for char, style in zip(chars, styles):
                if style != last:
                    fg, bg = style
                    line += "\x1b[0" + (f";38;2;{fg}" if fg else "") + (f";48;2;{bg}" if bg else "") + "m"
                    last = style
                line += char
            out.append(line + "\x1b[0m")
        return "\n".join(out)

    def sgr(self, params):
        values = [int(v) if v.isdigit() else 0 for v in params.split(";")] if params else [0]
        fg, bg = self.style
        i = 0
        while i < len(values):
            v = values[i]
            if v == 0:
                fg = bg = None
            elif v in (38, 48) and len(values) > i + 4 and values[i + 1] == 2:
                color = ";".join(str(c) for c in values[i + 2 : i + 5])
                fg, bg = (color, bg) if v == 38 else (fg, color)
                i += 4
            elif v in (38, 48) and len(values) > i + 2 and values[i + 1] == 5:
                color = xterm(values[i + 2])
                fg, bg = (color, bg) if v == 38 else (fg, color)
                i += 2
            elif v == 39:
                fg = None
            elif v == 49:
                bg = None
            i += 1
        self.style = (fg, bg)

    def replay(self, text):
        frames, seen = [], set()

        def snap():
            for line in self.lines():
                if line.strip() and line not in seen:
                    seen.add(line)
                    frames.append(line)

        for match in TOKEN.finditer(text):
            token, params, final = match.group(0), match.group(1), match.group(2)
            if final is None:
                if token == "\r":
                    self.col = 0
                elif token == "\n":
                    self.row = min(self.row + 1, self.rows - 1)
                elif token == "\b":
                    self.col = max(self.col - 1, 0)
                elif token >= " " and len(token) == 1:
                    if self.col < self.cols:
                        self.cells[self.row][self.col] = token
                        self.styles[self.row][self.col] = self.style
                    self.col += 1
                continue
            private = params.startswith("?")
            values = [int(v) if v.isdigit() else 0 for v in params.lstrip("?").split(";")]
            n = max(values[0], 1)
            if private and final in "hl" and "1049" in params:
                snap()
                self.resize(self.rows, self.cols)
            elif private and final == "l" and "2026" in params:
                snap()
            elif private:
                continue
            elif final == "m":
                self.sgr(params)
            elif final in "Hf":
                self.row = min(max(values[0], 1), self.rows) - 1
                self.col = min(max(values[1] if len(values) > 1 else 1, 1), self.cols) - 1
            elif final == "A":
                self.row = max(self.row - n, 0)
            elif final == "B":
                self.row = min(self.row + n, self.rows - 1)
            elif final == "C":
                self.col = min(self.col + n, self.cols - 1)
            elif final == "D":
                self.col = max(self.col - n, 0)
            elif final == "G":
                self.col = min(n, self.cols) - 1
            elif final == "d":
                self.row = min(n, self.rows) - 1
            elif final == "X":
                self.cells[self.row][self.col : self.col + n] = [" "] * len(self.cells[self.row][self.col : self.col + n])
            elif final == "K":
                line = self.cells[self.row]
                start, end = {0: (self.col, self.cols), 1: (0, self.col + 1)}.get(values[0], (0, self.cols))
                line[start:end] = [" "] * (end - start)
            elif final == "J":
                if values[0] in (2, 3):
                    snap()
                    self.cells = [[" "] * self.cols for _ in range(self.rows)]
                    self.styles = [[(None, None)] * self.cols for _ in range(self.rows)]
                elif values[0] == 0:
                    self.cells[self.row][self.col :] = [" "] * (self.cols - self.col)
                    for line in self.cells[self.row + 1 :]:
                        line[:] = [" "] * self.cols
        snap()
        return frames


if __name__ == "__main__":
    main()
