#!/usr/bin/env bash
set -euo pipefail

MODE="${VLLM_SR_INSTALL_MODE:-serve}"
REQUESTED_RUNTIME="${VLLM_SR_RUNTIME:-auto}"
INSTALL_ROOT="${VLLM_SR_INSTALL_ROOT:-$HOME/.local/share/vllm-sr}"
BIN_DIR="${VLLM_SR_BIN_DIR:-$HOME/.local/bin}"
PIP_SPEC="${VLLM_SR_PIP_SPEC:-}"
REQUESTED_CHANNEL="${VLLM_SR_INSTALL_CHANNEL:-dev}"
PYTHON_BIN="${VLLM_SR_PYTHON:-}"
REQUESTED_PLATFORM="${VLLM_SR_INSTALL_PLATFORM:-${VLLM_SR_PLATFORM:-auto}}"
AUTO_LAUNCH="${VLLM_SR_INSTALL_AUTO_LAUNCH:-1}"

OS_NAME=""
SELECTED_RUNTIME=""
LAUNCH_PLATFORM=""
AUTO_LAUNCH_RAN="0"
STYLE_BOLD=""
COLOR_RESET=""
COLOR_ORANGE=""
COLOR_BLUE=""
COLOR_WHITE=""
COLOR_MUTED=""
COLOR_SUCCESS=""

DASHBOARD_URL="http://localhost:8700"

init_colors() {
  if [ ! -t 1 ] || [ -n "${NO_COLOR:-}" ]; then
    return
  fi

  STYLE_BOLD=$'\033[1m'
  COLOR_RESET=$'\033[0m'
  COLOR_ORANGE=$'\033[38;2;254;181;22m'
  COLOR_BLUE=$'\033[38;2;48;162;255m'
  COLOR_WHITE=$'\033[97m'
  COLOR_MUTED=$'\033[38;2;145;158;171m'
  COLOR_SUCCESS=$'\033[38;2;111;224;161m'
}

print_logo() {
  if [ "${VLLM_SR_NO_LOGO:-0}" = "1" ]; then
    return
  fi

  init_colors

  printf '\n'
  printf '%b\n' "  ${STYLE_BOLD}${COLOR_WHITE}       █     █     █▄   ▄█${COLOR_RESET}"
  printf '%b\n' "  ${STYLE_BOLD}${COLOR_WHITE} ▄▄ ▄█ █     █     █ ▀▄▀ █${COLOR_RESET}"
  printf '%b\n' "  ${STYLE_BOLD}${COLOR_WHITE}  █▄█▀ █     █     █     █${COLOR_RESET}"
  printf '%b\n' "  ${STYLE_BOLD}${COLOR_WHITE}   ▀▀  ▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀${COLOR_RESET}"
  printf '%b\n' "  ${STYLE_BOLD}${COLOR_WHITE}  Semantic Router${COLOR_RESET}"
  printf '%b\n' "  ${COLOR_MUTED}  local installer${COLOR_RESET}"
  printf '\n'
}

step() {
  printf '%b\n' "${COLOR_BLUE}[step]${COLOR_RESET} $*"
}

done_step() {
  printf '%b\n' "${COLOR_SUCCESS}[done]${COLOR_RESET} $*"
}

info() {
  printf '%b\n' "${COLOR_MUTED}[info]${COLOR_RESET} $*"
}

warn() {
  printf '%b\n' "${COLOR_ORANGE}[warn]${COLOR_RESET} $*" >&2
}

die() {
  printf '%b\n' "${COLOR_ORANGE}[error]${COLOR_RESET} $*" >&2
  exit 1
}

is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

is_falsey() {
  case "${1:-}" in
    0|false|FALSE|no|NO|off|OFF)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

auto_launch_enabled() {
  is_truthy "$AUTO_LAUNCH"
}

should_auto_launch() {
  auto_launch_enabled || return 1
  [ "$MODE" = "serve" ] || return 1
  [ "$REQUESTED_RUNTIME" != "skip" ] || return 1
}

describe_package_channel() {
  if [ -n "$PIP_SPEC" ]; then
    printf 'custom (--pip-spec override)\n'
    return
  fi

  printf '%s\n' "$REQUESTED_CHANNEL"
}

describe_package_selection() {
  if [ -n "$PIP_SPEC" ]; then
    printf '%s\n' "$PIP_SPEC"
    return
  fi

  case "$REQUESTED_CHANNEL" in
    dev)
      printf 'vllm-sr (--pre, latest development release)\n'
      ;;
    stable)
      printf 'vllm-sr (latest stable release)\n'
      ;;
  esac
}

print_install_plan() {
  local requested_runtime python_cmd python_version runtime_cmd package_manager
  requested_runtime="$REQUESTED_RUNTIME"
  if [ "$REQUESTED_RUNTIME" = "auto" ]; then
    case "$OS_NAME" in
      darwin)
        requested_runtime="auto -> docker/colima"
        ;;
      linux)
        requested_runtime="auto -> docker"
        ;;
    esac
  fi

  python_cmd="$(detect_python_candidate || true)"
  runtime_cmd="$(detect_existing_runtime || true)"
  package_manager="$(detect_package_manager_label)"

  printf '%b\n' "${COLOR_WHITE}Detected environment${COLOR_RESET}"
  printf '  platform     %s (%s)\n' "$(detect_os_label)" "$(uname -m)"
  printf '  package mgr  %s\n' "$package_manager"
  if [ -n "$python_cmd" ]; then
    python_version="$(python_version_of "$python_cmd")"
    printf '  python       %s (%s)\n' "$python_cmd" "$python_version"
  else
    printf '  python       not found (needs Python 3.10+)\n'
  fi
  if [ -n "$runtime_cmd" ]; then
    printf '  runtime      %s (ready)\n' "$runtime_cmd"
  else
    printf '  runtime      none detected\n'
  fi
  printf '\n'

  printf '%b\n' "${COLOR_WHITE}Install plan${COLOR_RESET}"
  printf '  mode         %s\n' "$MODE"
  printf '  runtime      %s\n' "$requested_runtime"
  printf '  launch       %s\n' "$(if should_auto_launch; then printf 'auto first-run'; else printf 'manual'; fi)"
  printf '  platform     %s\n' "$(display_platform_plan)"
  printf '  channel      %s\n' "$(describe_package_channel)"
  printf '  python deps  pip, setuptools, wheel\n'
  printf '  package      %s\n' "$(describe_package_selection)"
  printf '  system deps  %s\n' "$(describe_python_dependency_plan "$python_cmd")"
  printf '  runtime deps %s\n' "$(describe_runtime_dependency_plan "$runtime_cmd")"
  printf '  install root %s\n' "$INSTALL_ROOT"
  printf '  launcher     %s/vllm-sr\n' "$BIN_DIR"
  printf '\n'
}

usage() {
  cat <<'EOF'
Usage: install.sh [--mode cli|serve] [--runtime auto|docker|skip]
                  [--install-root PATH] [--bin-dir PATH]
                  [--channel stable|dev] [--pip-spec SPEC]
                  [--python PATH] [--platform PLATFORM] [--no-launch]

Installs the vLLM Semantic Router CLI into an isolated virtual environment and
links a launcher into ~/.local/bin by default.

Options:
  --mode cli|serve         Install the CLI only, or prepare a local runtime for
                           `vllm-sr serve` as well. Default: serve
  --runtime auto|docker|skip
                           Runtime strategy for serve mode. Default: auto
                           macOS auto -> docker via colima
                           Linux auto -> docker
  --install-root PATH      Installation root. Default:
                           ~/.local/share/vllm-sr
  --bin-dir PATH           Launcher directory. Default: ~/.local/bin
  --channel stable|dev     Package channel to install when --pip-spec is not
                           set. Default: dev
  --pip-spec SPEC          Explicit Python package spec to install. Overrides
                           --channel when set
  --python PATH            Explicit Python interpreter to use
  --platform PLATFORM      Platform hint for first-run serve. Use 'amd' for ROCm.
                           Default: auto
  --no-launch              Skip the installer's automatic first `vllm-sr serve`
                           and dashboard open step
  -h, --help               Show this help message

Environment overrides:
  VLLM_SR_INSTALL_MODE
  VLLM_SR_RUNTIME
  VLLM_SR_INSTALL_ROOT
  VLLM_SR_BIN_DIR
  VLLM_SR_INSTALL_CHANNEL
  VLLM_SR_PIP_SPEC
  VLLM_SR_PYTHON
  VLLM_SR_INSTALL_PLATFORM
  VLLM_SR_INSTALL_AUTO_LAUNCH
EOF
}

has_cmd() {
  command -v "$1" >/dev/null 2>&1
}

make_temp_log() {
  mktemp "${TMPDIR:-/tmp}/vllm-sr-install.XXXXXX"
}

run_quiet_step() {
  local label log_file
  label="$1"
  shift

  step "$label"
  log_file="$(make_temp_log)"

  if "$@" >"$log_file" 2>&1; then
    rm -f "$log_file"
    done_step "$label"
    return
  fi

  warn "$label failed. Command output follows:"
  cat "$log_file" >&2
  rm -f "$log_file"
  exit 1
}

run_as_root() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
    return
  fi

  if has_cmd sudo; then
    sudo "$@"
    return
  fi

  die "This step requires root or sudo: $*"
}

detect_os() {
  case "$(uname -s)" in
    Darwin)
      OS_NAME="darwin"
      ;;
    Linux)
      OS_NAME="linux"
      ;;
    *)
      die "Unsupported operating system. This installer supports macOS and Linux."
      ;;
  esac
}

detect_os_label() {
  case "$OS_NAME" in
    darwin)
      printf 'macOS\n'
      ;;
    linux)
      printf 'Linux\n'
      ;;
    *)
      printf '%s\n' "$OS_NAME"
      ;;
  esac
}

resolve_launch_platform() {
  if [ "$REQUESTED_PLATFORM" != "auto" ]; then
    printf '%s\n' "$REQUESTED_PLATFORM"
    return
  fi

  if [ -n "${VLLM_SR_PLATFORM:-}" ]; then
    printf '%s\n' "$VLLM_SR_PLATFORM"
    return
  fi

  if has_cmd rocm-smi || has_cmd rocminfo || [ -e /dev/kfd ] || [ -d /opt/rocm ]; then
    printf 'amd\n'
    return
  fi

  printf '\n'
}

display_platform_plan() {
  local platform
  platform="$(resolve_launch_platform)"
  if [ -n "$platform" ]; then
    if [ "$REQUESTED_PLATFORM" = "auto" ]; then
      printf '%s (auto-detected)\n' "$platform"
    else
      printf '%s\n' "$platform"
    fi
    return
  fi

  printf 'default\n'
}

detect_primary_ip() {
  local python_cmd
  python_cmd="$(detect_python_candidate || true)"
  if [ -z "$python_cmd" ]; then
    return 1
  fi

  "$python_cmd" -c '
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(("8.8.8.8", 80))
    print(s.getsockname()[0])
finally:
    s.close()
' 2>/dev/null
}

detect_host_label() {
  hostname -f 2>/dev/null || hostname 2>/dev/null || printf 'your-server\n'
}

is_remote_session() {
  [ -n "${SSH_CONNECTION:-}" ] || [ -n "${SSH_TTY:-}" ]
}

open_dashboard_url() {
  case "$OS_NAME" in
    darwin)
      open "$DASHBOARD_URL" >/dev/null 2>&1
      ;;
    linux)
      if has_cmd xdg-open; then
        xdg-open "$DASHBOARD_URL" >/dev/null 2>&1
        return
      fi
      if has_cmd gio; then
        gio open "$DASHBOARD_URL" >/dev/null 2>&1
        return
      fi
      return 1
      ;;
    *)
      return 1
      ;;
  esac
}

print_dashboard_access() {
  local primary_ip host_label
  primary_ip="$(detect_primary_ip || true)"
  host_label="$(detect_host_label)"

  printf '%b\n' "${COLOR_WHITE}Dashboard access${COLOR_RESET}"
  printf '  local        %s\n' "$DASHBOARD_URL"
  if [ -n "$primary_ip" ]; then
    printf '  network      http://%s:8700\n' "$primary_ip"
  fi
  printf '\n'

  if is_remote_session; then
    printf '%b\n' "${COLOR_WHITE}Remote access${COLOR_RESET}"
    printf '  ssh tunnel   ssh -L 8700:localhost:8700 %s@%s\n' "${USER:-user}" "$host_label"
    printf '  then open    %s\n' "$DASHBOARD_URL"
    printf '\n'
  fi
}

resolve_launch_dir() {
  local current_dir fallback_dir
  current_dir="$(pwd)"
  if [ -w "$current_dir" ]; then
    printf '%s\n' "$current_dir"
    return
  fi

  fallback_dir="${HOME:-$INSTALL_ROOT}"
  warn "Current directory is not writable. Falling back to $fallback_dir for the first run."
  printf '%s\n' "$fallback_dir"
}

print_restart_command() {
  if [ -z "$LAUNCH_PLATFORM" ]; then
    LAUNCH_PLATFORM="$(resolve_launch_platform)"
  fi
  if [ -n "$LAUNCH_PLATFORM" ]; then
    printf '  vllm-sr serve --platform %s\n' "$LAUNCH_PLATFORM"
  else
    printf '  vllm-sr serve\n'
  fi
  printf '  vllm-sr dashboard\n'
}

python_supports_vllm_sr() {
  "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' \
    >/dev/null 2>&1
}

python_version_of() {
  "$1" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null || printf 'unknown\n'
}

find_python() {
  if [ -n "$PYTHON_BIN" ]; then
    if ! has_cmd "$PYTHON_BIN"; then
      die "Requested Python interpreter not found: $PYTHON_BIN"
    fi
    if ! python_supports_vllm_sr "$PYTHON_BIN"; then
      die "Requested Python interpreter must be Python 3.10 or newer: $PYTHON_BIN"
    fi
    printf '%s\n' "$PYTHON_BIN"
    return
  fi

  for candidate in python3 python3.12 python3.11 python3.10 python; do
    if has_cmd "$candidate" && python_supports_vllm_sr "$candidate"; then
      printf '%s\n' "$candidate"
      return
    fi
  done

  return 1
}

detect_python_candidate() {
  find_python
}

detect_linux_pkg_manager() {
  for candidate in apt-get dnf yum; do
    if has_cmd "$candidate"; then
      printf '%s\n' "$candidate"
      return
    fi
  done

  return 1
}

detect_package_manager_label() {
  local pkg_manager
  case "$OS_NAME" in
    darwin)
      if has_cmd brew; then
        printf 'homebrew\n'
      else
        printf 'homebrew (not found)\n'
      fi
      ;;
    linux)
      pkg_manager="$(detect_linux_pkg_manager || true)"
      if [ -n "$pkg_manager" ]; then
        printf '%s\n' "$pkg_manager"
      else
        printf 'not detected\n'
      fi
      ;;
  esac
}

detect_existing_runtime() {
  if docker_ready; then
    printf 'docker\n'
    return
  fi

  return 1
}

describe_python_dependency_plan() {
  local python_cmd pkg_manager
  python_cmd="$1"
  if [ -n "$python_cmd" ]; then
    printf 'none (using %s)\n' "$python_cmd"
    return
  fi

  case "$OS_NAME" in
    darwin)
      printf 'python via Homebrew\n'
      ;;
    linux)
      pkg_manager="$(detect_linux_pkg_manager || true)"
      case "$pkg_manager" in
        apt-get)
          printf 'python3, python3-venv, python3-pip via apt-get\n'
          ;;
        dnf)
          printf 'python3, python3-pip via dnf\n'
          ;;
        yum)
          printf 'python3, python3-pip via yum\n'
          ;;
        *)
          printf 'python 3.10+ required (install manually)\n'
          ;;
      esac
      ;;
  esac
}

describe_runtime_dependency_plan() {
  local runtime_cmd preferred_runtime pkg_manager
  runtime_cmd="$1"

  if [ "$MODE" = "cli" ] || [ "$REQUESTED_RUNTIME" = "skip" ]; then
    printf 'none\n'
    return
  fi

  if [ -n "$runtime_cmd" ]; then
    printf 'none (using existing %s)\n' "$runtime_cmd"
    return
  fi

  preferred_runtime="$(choose_runtime_preference)"
  case "$OS_NAME:$preferred_runtime" in
    darwin:docker)
      printf 'docker, colima via Homebrew\n'
      ;;
    linux:docker)
      pkg_manager="$(detect_linux_pkg_manager || true)"
      case "$pkg_manager" in
        apt-get)
          printf 'docker.io via apt-get\n'
          ;;
        dnf)
          printf 'docker via dnf\n'
          ;;
        yum)
          printf 'docker via yum\n'
          ;;
        *)
          printf 'docker required (install manually)\n'
          ;;
      esac
      ;;
    *)
      printf 'none\n'
      ;;
  esac
}

ensure_homebrew() {
  if ! has_cmd brew; then
    die "Homebrew is required on macOS when Python or a container runtime must be installed automatically."
  fi
}

install_python() {
  step "Installing Python 3.10+"

  case "$OS_NAME" in
    darwin)
      ensure_homebrew
      brew install python
      ;;
    linux)
      local pkg_manager
      pkg_manager="$(detect_linux_pkg_manager)" || die \
        "No supported Linux package manager found. Install Python 3.10+ manually and re-run the installer."
      case "$pkg_manager" in
        apt-get)
          run_as_root apt-get update
          run_as_root apt-get install -y python3 python3-venv python3-pip
          ;;
        dnf)
          run_as_root dnf install -y python3 python3-pip
          ;;
        yum)
          run_as_root yum install -y python3 python3-pip
          ;;
      esac
      ;;
  esac

  done_step "Python 3.10+ is ready"
}

create_launcher() {
  local launcher_path executable_path
  launcher_path="$BIN_DIR/vllm-sr"
  executable_path="$INSTALL_ROOT/venv/bin/vllm-sr"

  mkdir -p "$BIN_DIR"
  cat >"$launcher_path" <<EOF
#!/usr/bin/env bash
set -euo pipefail

exec "$executable_path" "\$@"
EOF
  chmod +x "$launcher_path"
}

install_requested_package() {
  if [ -n "$PIP_SPEC" ]; then
    run_quiet_step \
      "Installing vLLM Semantic Router from $PIP_SPEC" \
      "$INSTALL_ROOT/venv/bin/python" -m pip install --disable-pip-version-check --upgrade --quiet "$PIP_SPEC"
    return
  fi

  case "$REQUESTED_CHANNEL" in
    dev)
      run_quiet_step \
        "Installing latest development vLLM Semantic Router release" \
        "$INSTALL_ROOT/venv/bin/python" -m pip install --disable-pip-version-check --upgrade --quiet --pre vllm-sr
      ;;
    stable)
      run_quiet_step \
        "Installing latest stable vLLM Semantic Router release" \
        "$INSTALL_ROOT/venv/bin/python" -m pip install --disable-pip-version-check --upgrade --quiet vllm-sr
      ;;
  esac
}

install_cli() {
  local python_cmd
  python_cmd="$(find_python)" || {
    info "Python 3.10+ was not found. The installer will add it now."
    install_python
    python_cmd="$(find_python)" || die "Unable to locate a Python 3.10+ interpreter after installation."
  }

  done_step "Using Python interpreter: $python_cmd"
  mkdir -p "$INSTALL_ROOT"
  step "Creating isolated environment at $INSTALL_ROOT/venv"
  "$python_cmd" -m venv "$INSTALL_ROOT/venv"
  done_step "Created isolated environment"
  run_quiet_step \
    "Bootstrapping installer Python tooling" \
    "$INSTALL_ROOT/venv/bin/python" -m pip install --disable-pip-version-check --upgrade --quiet pip setuptools wheel
  install_requested_package
  step "Writing launcher to $BIN_DIR/vllm-sr"
  create_launcher
  done_step "Launcher is ready"

  local version_output
  version_output="$("$BIN_DIR/vllm-sr" --version 2>/dev/null || true)"
  if [ -n "$version_output" ]; then
    done_step "Installed $version_output"
  else
    done_step "Installed vllm-sr"
  fi
}

docker_ready() {
  has_cmd docker && docker info >/dev/null 2>&1
}

choose_runtime_preference() {
  if [ "$REQUESTED_RUNTIME" != "auto" ]; then
    printf '%s\n' "$REQUESTED_RUNTIME"
    return
  fi

  case "$OS_NAME" in
    darwin)
      printf 'docker\n'
      ;;
    linux)
      printf 'docker\n'
      ;;
  esac
}

install_macos_docker_runtime() {
  ensure_homebrew
  step "Installing Docker CLI and Colima via Homebrew"
  brew install docker colima
  step "Starting Colima"
  colima start
  done_step "Docker + Colima runtime is ready"
}

install_linux_docker_runtime() {
  local pkg_manager
  pkg_manager="$(detect_linux_pkg_manager)" || die \
    "No supported Linux package manager found. Install Docker manually and re-run the installer."

  step "Installing Docker"
  case "$pkg_manager" in
    apt-get)
      run_as_root apt-get update
      run_as_root apt-get install -y docker.io
      ;;
    dnf)
      run_as_root dnf install -y docker
      ;;
    yum)
      run_as_root yum install -y docker
      ;;
  esac

  if has_cmd systemctl; then
    run_as_root systemctl enable --now docker || true
  fi

  if [ "$(id -u)" -ne 0 ]; then
    run_as_root usermod -aG docker "$USER" || true
  fi

  done_step "Docker runtime is ready"
}

write_runtime_env() {
  rm -f "$INSTALL_ROOT/runtime.env"
}

ensure_runtime() {
  if [ "$MODE" = "cli" ] || [ "$REQUESTED_RUNTIME" = "skip" ]; then
    SELECTED_RUNTIME=""
    write_runtime_env
    info "Runtime bootstrap skipped."
    return
  fi

  if docker_ready; then
    SELECTED_RUNTIME="docker"
    write_runtime_env
    done_step "Using existing Docker runtime"
    return
  fi

  case "$(choose_runtime_preference)" in
    docker)
      case "$OS_NAME" in
        darwin)
          install_macos_docker_runtime
          docker_ready || die "Docker is installed but not reachable. Try running 'colima start' and then 'vllm-sr serve'."
          SELECTED_RUNTIME="docker"
          ;;
        linux)
          install_linux_docker_runtime
          if docker_ready; then
            SELECTED_RUNTIME="docker"
          else
            die "Docker was installed but is not reachable from the current shell. Open a new shell or run 'newgrp docker', then start with 'vllm-sr serve'."
          fi
          ;;
      esac
      ;;
    *)
      die "Unsupported runtime selection: $(choose_runtime_preference)"
      ;;
  esac

  write_runtime_env
}

launch_first_session() {
  local launch_dir serve_log_file
  if ! should_auto_launch; then
    return
  fi

  LAUNCH_PLATFORM="$(resolve_launch_platform)"
  launch_dir="$(resolve_launch_dir)"

  if [ -n "$LAUNCH_PLATFORM" ]; then
    info "First-run serve command: vllm-sr serve --platform $LAUNCH_PLATFORM"
  else
    info "First-run serve command: vllm-sr serve"
  fi
  info "First-run dashboard command: vllm-sr dashboard"
  info "Starting the first local session. This can take a few minutes on the first image pull."

  step "Running first-time serve flow"
  serve_log_file="$(make_temp_log)"
  if (
    cd "$launch_dir"
    if [ -n "$LAUNCH_PLATFORM" ]; then
      "$BIN_DIR/vllm-sr" serve --platform "$LAUNCH_PLATFORM"
    else
      "$BIN_DIR/vllm-sr" serve
    fi
  ) >"$serve_log_file" 2>&1; then
    rm -f "$serve_log_file"
    AUTO_LAUNCH_RAN="1"
    done_step "First-time serve flow completed"
  else
    warn "Automatic first run did not complete. Command output follows:"
    cat "$serve_log_file" >&2
    rm -f "$serve_log_file"
    warn "Automatic first run did not complete. Retry with:"
    print_restart_command
    die "Installation finished, but the first vllm-sr serve run failed"
  fi

  step "Checking dashboard availability"
  if "$BIN_DIR/vllm-sr" dashboard --no-open >/dev/null 2>&1; then
    done_step "Dashboard is available"
  else
    warn "The dashboard command could not confirm a running session."
    print_dashboard_access
    die "Installation finished, but the dashboard did not come up cleanly"
  fi

  step "Opening dashboard"
  if open_dashboard_url; then
    done_step "Dashboard opened in your browser"
  else
    warn "Could not open a browser automatically."
    done_step "Dashboard is running"
    print_dashboard_access
  fi
}

print_path_hint() {
  local shell_path_placeholder
  shell_path_placeholder="$(printf '%s' "\$PATH")"
  case ":$PATH:" in
    *":$BIN_DIR:"*)
      return 1
      ;;
    *)
      warn "$BIN_DIR is not on PATH."
      printf '%b\n' "${COLOR_WHITE}Current shell${COLOR_RESET}"
      printf '  export PATH="%s:%s"\n' "$BIN_DIR" "$shell_path_placeholder"
      printf '\n'
      return 0
      ;;
  esac
}

print_next_steps() {
  local primary_ip host_label

  printf '\n'
  done_step "vLLM Semantic Router is ready"

  if [ -z "$LAUNCH_PLATFORM" ]; then
    LAUNCH_PLATFORM="$(resolve_launch_platform)"
  fi

  if print_path_hint; then
    info "Add the PATH export above before running vllm-sr."
  fi

  printf '%b\n' "${COLOR_WHITE}Ready${COLOR_RESET}"
  if [ "$MODE" = "serve" ]; then
    printf '  runtime      %s\n' "${SELECTED_RUNTIME:-not configured}"
  fi

  if [ "$AUTO_LAUNCH_RAN" = "1" ]; then
    primary_ip="$(detect_primary_ip || true)"
    printf '  dashboard    %s\n' "$DASHBOARD_URL"
    if [ -n "$primary_ip" ]; then
      printf '  network      http://%s:8700\n' "$primary_ip"
    fi
    printf '  stop         vllm-sr stop\n'
    if [ -n "$LAUNCH_PLATFORM" ]; then
      printf '  restart      vllm-sr serve --platform %s\n' "$LAUNCH_PLATFORM"
    else
      printf '  restart      vllm-sr serve\n'
    fi
    if is_remote_session; then
      host_label="$(detect_host_label)"
      printf '  tunnel       ssh -L 8700:localhost:8700 %s@%s\n' "${USER:-user}" "$host_label"
    fi
  else
    printf '  verify       vllm-sr --version\n'
    if [ "$MODE" = "serve" ]; then
      if [ -n "$LAUNCH_PLATFORM" ]; then
        printf '  start        vllm-sr serve --platform %s\n' "$LAUNCH_PLATFORM"
      else
        printf '  start        vllm-sr serve\n'
      fi
      printf '  open         vllm-sr dashboard\n'
    fi
  fi
  printf '\n'
}

parse_args() {
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --mode)
        [ "$#" -ge 2 ] || die "Missing value for --mode"
        MODE="$2"
        shift 2
        ;;
      --runtime)
        [ "$#" -ge 2 ] || die "Missing value for --runtime"
        REQUESTED_RUNTIME="$2"
        shift 2
        ;;
      --install-root)
        [ "$#" -ge 2 ] || die "Missing value for --install-root"
        INSTALL_ROOT="$2"
        shift 2
        ;;
      --bin-dir)
        [ "$#" -ge 2 ] || die "Missing value for --bin-dir"
        BIN_DIR="$2"
        shift 2
        ;;
      --pip-spec)
        [ "$#" -ge 2 ] || die "Missing value for --pip-spec"
        PIP_SPEC="$2"
        shift 2
        ;;
      --channel)
        [ "$#" -ge 2 ] || die "Missing value for --channel"
        REQUESTED_CHANNEL="$2"
        shift 2
        ;;
      --python)
        [ "$#" -ge 2 ] || die "Missing value for --python"
        PYTHON_BIN="$2"
        shift 2
        ;;
      --platform)
        [ "$#" -ge 2 ] || die "Missing value for --platform"
        REQUESTED_PLATFORM="$2"
        shift 2
        ;;
      --no-launch)
        AUTO_LAUNCH="0"
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "Unknown argument: $1"
        ;;
    esac
  done
}

validate_args() {
  case "$MODE" in
    cli|serve)
      ;;
    *)
      die "--mode must be 'cli' or 'serve'"
      ;;
  esac

  case "$REQUESTED_RUNTIME" in
    auto|docker|skip)
      ;;
    *)
      die "--runtime must be one of: auto, docker, skip"
      ;;
  esac

  case "$REQUESTED_CHANNEL" in
    stable|dev)
      ;;
    *)
      die "--channel must be one of: stable, dev"
      ;;
  esac

  if ! is_truthy "$AUTO_LAUNCH" && ! is_falsey "$AUTO_LAUNCH"; then
    die "VLLM_SR_INSTALL_AUTO_LAUNCH must be a boolean-like value (1/0, true/false, yes/no)"
  fi
}

main() {
  parse_args "$@"
  validate_args
  detect_os
  print_logo
  print_install_plan
  install_cli
  ensure_runtime
  launch_first_session
  print_next_steps
}

main "$@"
