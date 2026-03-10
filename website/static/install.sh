#!/usr/bin/env bash
set -euo pipefail

MODE="${VLLM_SR_INSTALL_MODE:-serve}"
REQUESTED_RUNTIME="${VLLM_SR_RUNTIME:-auto}"
INSTALL_ROOT="${VLLM_SR_INSTALL_ROOT:-$HOME/.local/share/vllm-sr}"
BIN_DIR="${VLLM_SR_BIN_DIR:-$HOME/.local/bin}"
PIP_SPEC="${VLLM_SR_PIP_SPEC:-vllm-sr}"
PYTHON_BIN="${VLLM_SR_PYTHON:-}"

OS_NAME=""
SELECTED_RUNTIME=""
COLOR_RESET=""
COLOR_ORANGE=""
COLOR_BLUE=""

init_colors() {
  if [ ! -t 1 ] || [ -n "${NO_COLOR:-}" ]; then
    return
  fi

  COLOR_RESET=$'\033[0m'
  COLOR_ORANGE=$'\033[38;2;254;181;22m'
  COLOR_BLUE=$'\033[38;2;48;162;255m'
}

print_logo() {
  if [ "${VLLM_SR_NO_LOGO:-0}" = "1" ]; then
    return
  fi

  init_colors
  printf '%b\n' "${COLOR_ORANGE}######################${COLOR_RESET}            ${COLOR_BLUE}########################${COLOR_RESET}"
  printf '%b\n' " ${COLOR_ORANGE}####################${COLOR_RESET}          ${COLOR_BLUE}########################${COLOR_RESET}"
  printf '%b\n' "  ${COLOR_ORANGE}##################${COLOR_RESET}        ${COLOR_BLUE}########################${COLOR_RESET}"
  printf '%b\n' "   ${COLOR_ORANGE}################${COLOR_RESET}      ${COLOR_BLUE}########################${COLOR_RESET}"
  printf '%b\n' "    ${COLOR_ORANGE}##############${COLOR_RESET}    ${COLOR_BLUE}########################${COLOR_RESET}"
  printf '%b\n' "     ${COLOR_ORANGE}############${COLOR_BLUE}############################${COLOR_RESET}"
  printf '%b\n' "      ${COLOR_ORANGE}########${COLOR_BLUE}############################${COLOR_RESET}"
  printf '%b\n' "       ${COLOR_ORANGE}####${COLOR_BLUE}############################${COLOR_RESET}"
  printf '\n'
}

log() {
  printf '[vllm-sr] %s\n' "$*"
}

warn() {
  printf '[vllm-sr] warning: %s\n' "$*" >&2
}

die() {
  printf '[vllm-sr] error: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: install.sh [--mode cli|serve] [--runtime auto|docker|podman|skip]
                  [--install-root PATH] [--bin-dir PATH] [--pip-spec SPEC]
                  [--python PATH]

Installs the vLLM Semantic Router CLI into an isolated virtual environment and
links a launcher into ~/.local/bin by default.

Options:
  --mode cli|serve         Install the CLI only, or prepare a local runtime for
                           `vllm-sr serve` as well. Default: serve
  --runtime auto|docker|podman|skip
                           Runtime strategy for serve mode. Default: auto
                           macOS auto -> docker via colima
                           Linux auto -> podman
  --install-root PATH      Installation root. Default:
                           ~/.local/share/vllm-sr
  --bin-dir PATH           Launcher directory. Default: ~/.local/bin
  --pip-spec SPEC          Python package spec to install. Default: vllm-sr
  --python PATH            Explicit Python interpreter to use
  -h, --help               Show this help message

Environment overrides:
  VLLM_SR_INSTALL_MODE
  VLLM_SR_RUNTIME
  VLLM_SR_INSTALL_ROOT
  VLLM_SR_BIN_DIR
  VLLM_SR_PIP_SPEC
  VLLM_SR_PYTHON
EOF
}

has_cmd() {
  command -v "$1" >/dev/null 2>&1
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

python_supports_vllm_sr() {
  "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' \
    >/dev/null 2>&1
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

detect_linux_pkg_manager() {
  for candidate in apt-get dnf yum; do
    if has_cmd "$candidate"; then
      printf '%s\n' "$candidate"
      return
    fi
  done

  return 1
}

ensure_homebrew() {
  if ! has_cmd brew; then
    die "Homebrew is required on macOS when Python or a container runtime must be installed automatically."
  fi
}

install_python() {
  log "Installing Python 3.10+"

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
}

create_launcher() {
  local launcher_path runtime_env_path executable_path
  launcher_path="$BIN_DIR/vllm-sr"
  runtime_env_path="$INSTALL_ROOT/runtime.env"
  executable_path="$INSTALL_ROOT/venv/bin/vllm-sr"

  mkdir -p "$BIN_DIR"
  cat >"$launcher_path" <<EOF
#!/usr/bin/env bash
set -euo pipefail

if [ -f "$runtime_env_path" ]; then
  # shellcheck disable=SC1090
  . "$runtime_env_path"
fi

exec "$executable_path" "\$@"
EOF
  chmod +x "$launcher_path"
}

install_cli() {
  local python_cmd
  python_cmd="$(find_python)" || {
    install_python
    python_cmd="$(find_python)" || die "Unable to locate a Python 3.10+ interpreter after installation."
  }

  log "Using Python interpreter: $python_cmd"
  mkdir -p "$INSTALL_ROOT"
  "$python_cmd" -m venv "$INSTALL_ROOT/venv"
  "$INSTALL_ROOT/venv/bin/python" -m pip install --upgrade pip setuptools wheel
  "$INSTALL_ROOT/venv/bin/python" -m pip install --upgrade "$PIP_SPEC"
  create_launcher

  local version_output
  version_output="$("$BIN_DIR/vllm-sr" --version 2>/dev/null || true)"
  if [ -n "$version_output" ]; then
    log "Installed $version_output"
  else
    log "Installed vllm-sr"
  fi
}

docker_ready() {
  has_cmd docker && docker info >/dev/null 2>&1
}

podman_ready() {
  has_cmd podman && podman info >/dev/null 2>&1
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
      printf 'podman\n'
      ;;
  esac
}

install_macos_docker_runtime() {
  ensure_homebrew
  log "Installing Docker CLI and Colima via Homebrew"
  brew install docker colima
  log "Starting Colima"
  colima start
}

install_macos_podman_runtime() {
  ensure_homebrew
  log "Installing Podman via Homebrew"
  brew install podman
  if ! podman machine inspect >/dev/null 2>&1; then
    podman machine init
  fi
  log "Starting Podman machine"
  podman machine start
}

install_linux_podman_runtime() {
  local pkg_manager
  pkg_manager="$(detect_linux_pkg_manager)" || die \
    "No supported Linux package manager found. Install Podman manually and re-run the installer."

  log "Installing Podman"
  case "$pkg_manager" in
    apt-get)
      run_as_root apt-get update
      run_as_root apt-get install -y podman uidmap slirp4netns
      ;;
    dnf)
      run_as_root dnf install -y podman
      ;;
    yum)
      run_as_root yum install -y podman
      ;;
  esac
}

install_linux_docker_runtime() {
  local pkg_manager
  pkg_manager="$(detect_linux_pkg_manager)" || die \
    "No supported Linux package manager found. Install Docker manually and re-run the installer."

  log "Installing Docker"
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
}

write_runtime_env() {
  local runtime_env_path
  runtime_env_path="$INSTALL_ROOT/runtime.env"

  case "$SELECTED_RUNTIME" in
    podman)
      printf 'export CONTAINER_RUNTIME=podman\n' >"$runtime_env_path"
      ;;
    docker|'')
      rm -f "$runtime_env_path"
      ;;
  esac
}

ensure_runtime() {
  if [ "$MODE" = "cli" ] || [ "$REQUESTED_RUNTIME" = "skip" ]; then
    SELECTED_RUNTIME=""
    write_runtime_env
    return
  fi

  if docker_ready; then
    SELECTED_RUNTIME="docker"
    write_runtime_env
    return
  fi

  if podman_ready; then
    SELECTED_RUNTIME="podman"
    write_runtime_env
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
    podman)
      case "$OS_NAME" in
        darwin)
          install_macos_podman_runtime
          podman_ready || die "Podman is installed but not reachable. Try running 'podman machine start' and then 'vllm-sr serve'."
          ;;
        linux)
          install_linux_podman_runtime
          podman_ready || die "Podman is installed but not reachable from the current shell."
          ;;
      esac
      SELECTED_RUNTIME="podman"
      ;;
    *)
      die "Unsupported runtime selection: $(choose_runtime_preference)"
      ;;
  esac

  write_runtime_env
}

print_path_hint() {
  local shell_path_placeholder
  shell_path_placeholder="$(printf '%s' "\$PATH")"
  case ":$PATH:" in
    *":$BIN_DIR:"*)
      ;;
    *)
      warn "$BIN_DIR is not on PATH."
      printf 'export PATH="%s:%s"\n' "$BIN_DIR" "$shell_path_placeholder"
      ;;
  esac
}

print_next_steps() {
  log "Installation complete."
  print_path_hint

  printf '\n'
  printf 'Next steps:\n'
  printf '  %s --version\n' "$BIN_DIR/vllm-sr"

  if [ "$MODE" = "serve" ]; then
    printf '  %s serve\n' "$BIN_DIR/vllm-sr"
    printf '  %s serve --platform amd\n' "$BIN_DIR/vllm-sr"
    if [ "$SELECTED_RUNTIME" = "podman" ]; then
      printf '\n'
      printf 'Runtime:\n'
      printf '  This installation is pinned to Podman via %s/runtime.env\n' "$INSTALL_ROOT"
    fi
  fi
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
      --python)
        [ "$#" -ge 2 ] || die "Missing value for --python"
        PYTHON_BIN="$2"
        shift 2
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
    auto|docker|podman|skip)
      ;;
    *)
      die "--runtime must be one of: auto, docker, podman, skip"
      ;;
  esac
}

main() {
  print_logo
  parse_args "$@"
  validate_args
  detect_os
  install_cli
  ensure_runtime
  print_next_steps
}

main "$@"
