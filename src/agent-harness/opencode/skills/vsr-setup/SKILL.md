# Set up vSR's decision runtime

vSR asks a small decision model about every child task, prompt, read, shell command and answer. This skill gets that model answering on this machine, or points vSR at one that already answers, and hands over to OpenCode's own controls. Run each command yourself, show what it prints, and ask the user only what they decide. Nothing here needs sudo; if a step seems to, stop and tell the user why.

The runtime is the same one the Claude Code plugin installs: `~/.local/share/vsr/venv`, port 8790, so one runtime serves every agent on this machine.

## Steps

1. **Look first.** Run these and summarize them in a few lines; they install nothing:

   - `curl -s --max-time 3 http://127.0.0.1:8790/health` — a runtime already runs here when this prints `"status":"ready"`; `"loading"` means one is still starting.
   - `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader` (or `rocm-smi --showproductname` on AMD) — a GPU, when there is one.
   - `python3 --version`, `df -h ~` and `free -h` (`sysctl hw.memsize` on macOS).

   When a runtime already answers, skip to step 5. Done when you know whether one answers, and what this machine has.

2. **Where the model runs.** Ask the user one question, the first option marked recommended by what step 1 found:

   - **This machine's GPU**: fastest; vela3-lite uses about 2 GB of GPU memory.
   - **This machine's CPU**: about 2.3 GB of RAM and 0.1 to 0.25 s per question on an x86 CPU with bf16, up to about 2 s on a laptop or Apple Silicon CPU; at that speed apps often step aside within their wait.
   - **A runtime on another machine**: nothing is installed. Tell the user plainly that everything the enabled apps check is sent to that host: each read and shell output, fetched page, prompt, child task, shell command and answer. Skip to step 5.

   Done when the user picked one. An Apple GPU cannot run vela3 yet, so a Mac runs it on the CPU.

3. **Install**, only when the user agrees to the download: about 200 MB of CPU torch, or 3 to 4 GB of CUDA or ROCm torch (5 to 6 GB on disk; a few minutes on a fast connection), plus about 1.8 GB of model weights on first start. Use a Python 3.10 or later; skip any command whose result already exists (`~/.local/share/vsr/venv/bin/vllm-srun` means it is installed). Each takes minutes, so run the pip commands in the background and wait for them.

   ```bash
   python3 -m venv ~/.local/share/vsr/venv
   ~/.local/share/vsr/venv/bin/python -m pip install --upgrade pip
   # CPU on Linux only: the CPU wheels save several GB of CUDA libraries.
   ~/.local/share/vsr/venv/bin/python -m pip install --upgrade 'torch>=2.4' --index-url https://download.pytorch.org/whl/cpu
   # GPU, or a Mac: torch from PyPI instead.
   ~/.local/share/vsr/venv/bin/python -m pip install --upgrade 'torch>=2.4'
   # The runtime, pinned to the commit vSR was tested against; drop `-c <lock>` on a Python older than 3.12.
   ~/.local/share/vsr/venv/bin/python -m pip install --upgrade -c https://raw.githubusercontent.com/vllm-project/semantic-router/579e0ca2a0b1b03580eada4213da84fe68db57c2/src/model-runtime/requirements-lock.txt 'vllm-srun @ git+https://github.com/vllm-project/semantic-router.git@579e0ca2a0b1b03580eada4213da84fe68db57c2#subdirectory=src/model-runtime'
   ```

   On failure, show the last pip lines and stop. For ROCm, torch from PyPI is a CUDA build: point the user to pytorch.org for the ROCm wheel index. Done when `~/.local/share/vsr/venv/bin/vllm-srun --help` prints its usage.

4. **Start it.** Ask whether it should run for this login only (it stops at logout or reboot) or keep running across reboots. For this login, with `cpu`, `cuda` or `rocm` as the device:

   ```bash
   nohup ~/.local/share/vsr/venv/bin/vllm-srun serve vllm-sr/vela3-lite --device <device> --host 127.0.0.1 --port 8790 >> ~/.local/share/vsr/runtime.log 2>&1 &
   ```

   Across reboots on Linux, write that command as the `ExecStart` of a systemd user unit at `~/.config/systemd/user/vsr-runtime.service` (with `Restart=on-failure` and the log appended to `~/.local/share/vsr/runtime.log`), show it to the user, and only on their yes run `systemctl --user daemon-reload && systemctl --user enable --now vsr-runtime`. On macOS, the same as a launchd agent in `~/Library/LaunchAgents`. Done when the process runs.

5. **Wait until it answers.** The first start downloads the model, which can take several minutes. Poll `curl -s --max-time 3 http://127.0.0.1:8790/health` every 10 seconds, up to 15 minutes, until it prints `"status":"ready"`. For a remote runtime, use the user's URL with `/v1/decisions` replaced by `/health`; a Router listener answers `"status":"ok"` there and a hosted API has no `/health`, so for those `/v1/models` listing a model means it answers. If it never answers, show the last lines of `~/.local/share/vsr/runtime.log`. Done when it is ready.

6. **Hand over.** Tell the user, in this order:

   - Run `/connect`, pick **vSR decision runtime**, then **Runtime without a token** for this machine (URL `http://127.0.0.1:8790/v1/decisions`), or **Runtime with a token** for one that needs a key. OpenCode keeps the token in its own credential store; never ask the user to paste it into the chat. Leave the model empty unless the runtime serves several.
   - Run `/vsr-check` in a session: it asks one case per enabled app and shows whether each decides as it does on vela3-lite, and how long each took.
   - `/vsr-apps` turns apps on or off; check is off by default, and on a CPU runtime a long answer can outlast its wait.

   Done when the user has these steps.
