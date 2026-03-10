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
COLOR_WHITE=""
COLOR_MUTED=""
COLOR_SUCCESS=""

init_colors() {
  if [ ! -t 1 ] || [ -n "${NO_COLOR:-}" ]; then
    return
  fi

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
  printf '%b\n' "${COLOR_ORANGE}##${COLOR_WHITE}          ${COLOR_BLUE}##  ${COLOR_WHITE}##        ##        ##      ##    ######    ########${COLOR_RESET}"
  printf '%b\n' "${COLOR_ORANGE} ##${COLOR_WHITE}        ${COLOR_BLUE}##   ${COLOR_WHITE}##        ##        ###    ###   ##    ##   ##    ##${COLOR_RESET}"
  printf '%b\n' "${COLOR_ORANGE}  ##${COLOR_WHITE}      ${COLOR_BLUE}##    ${COLOR_WHITE}##        ##        ####  ####   ##         ##    ##${COLOR_RESET}"
  printf '%b\n' "${COLOR_ORANGE}   ##${COLOR_WHITE}    ${COLOR_BLUE}##     ${COLOR_WHITE}##        ##        ## #### ##    ####     ########${COLOR_RESET}"
  printf '%b\n' "${COLOR_ORANGE}    ##${COLOR_WHITE}  ${COLOR_BLUE}##      ${COLOR_WHITE}##        ##        ##  ##  ##       ##    ##  ##${COLOR_RESET}"
  printf '%b\n' "${COLOR_ORANGE}     ##${COLOR_BLUE}##       ${COLOR_WHITE}##        ##        ##      ##   ##    ##   ##   ##${COLOR_RESET}"
  printf '%b\n' "${COLOR_ORANGE}      ${COLOR_BLUE}##        ${COLOR_WHITE}########  ########  ##      ##    ######    ##    ##${COLOR_RESET}"
  printf '%b\n' "${COLOR_MUTED}vLLM Semantic Router local installer${COLOR_RESET}"
  printf '%b\n' "${COLOR_MUTED}One command for the CLI, runtime, and first local bring-up.${COLOR_RESET}"
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

print_install_plan() {
  local requested_runtime python_cmd python_version runtime_cmd package_manager
  requested_runtime="$REQUESTED_RUNTIME"
  if [ "$REQUESTED_RUNTIME" = "auto" ]; then
    case "$OS_NAME" in
      darwin)
        requested_runtime="auto -> docker/colima"
        ;;
      linux)
        requested_runtime="auto -> podman"
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
  printf '  python deps  pip, setuptools, wheel\n'
  printf '  package      %s\n' "$PIP_SPEC"
  printf '  system deps  %s\n' "$(describe_python_dependency_plan "$python_cmd")"
  printf '  runtime deps %s\n' "$(describe_runtime_dependency_plan "$runtime_cmd")"
  printf '  install root %s\n' "$INSTALL_ROOT"
  printf '  launcher     %s/vllm-sr\n' "$BIN_DIR"
  printf '\n'
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

  if podman_ready; then
    printf 'podman\n'
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
    darwin:podman)
      printf 'podman via Homebrew\n'
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
    linux:podman)
      pkg_manager="$(detect_linux_pkg_manager || true)"
      case "$pkg_manager" in
        apt-get)
          printf 'podman, uidmap, slirp4netns via apt-get\n'
          ;;
        dnf)
          printf 'podman via dnf\n'
          ;;
        yum)
          printf 'podman via yum\n'
          ;;
        *)
          printf 'podman required (install manually)\n'
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
  run_quiet_step \
    "Installing vLLM Semantic Router from $PIP_SPEC" \
    "$INSTALL_ROOT/venv/bin/python" -m pip install --disable-pip-version-check --upgrade --quiet "$PIP_SPEC"
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
  step "Installing Docker CLI and Colima via Homebrew"
  brew install docker colima
  step "Starting Colima"
  colima start
  done_step "Docker + Colima runtime is ready"
}

install_macos_podman_runtime() {
  ensure_homebrew
  step "Installing Podman via Homebrew"
  brew install podman
  if ! podman machine inspect >/dev/null 2>&1; then
    step "Initializing Podman machine"
    podman machine init
  fi
  step "Starting Podman machine"
  podman machine start
  done_step "Podman runtime is ready"
}

install_linux_podman_runtime() {
  local pkg_manager
  pkg_manager="$(detect_linux_pkg_manager)" || die \
    "No supported Linux package manager found. Install Podman manually and re-run the installer."

  step "Installing Podman"
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

  done_step "Podman runtime is ready"
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
    info "Runtime bootstrap skipped."
    return
  fi

  if docker_ready; then
    SELECTED_RUNTIME="docker"
    write_runtime_env
    done_step "Using existing Docker runtime"
    return
  fi

  if podman_ready; then
    SELECTED_RUNTIME="podman"
    write_runtime_env
    done_step "Using existing Podman runtime"
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
      printf '  export PATH="%s:%s"\n' "$BIN_DIR" "$shell_path_placeholder"
      ;;
  esac
}

print_next_steps() {
  printf '\n'
  printf '%b\n' "${COLOR_SUCCESS}vLLM Semantic Router is installed.${COLOR_RESET}"
  printf '%b\n' "${COLOR_WHITE}What you have${COLOR_RESET}"
  printf '  cli          %s/vllm-sr\n' "$BIN_DIR"
  printf '  workspace     %s\n' "$INSTALL_ROOT"
  if [ "$MODE" = "serve" ]; then
    printf '  runtime       %s\n' "${SELECTED_RUNTIME:-not configured}"
  fi
  printf '\n'

  print_path_hint

  printf '%b\n' "${COLOR_WHITE}Next steps${COLOR_RESET}"
  printf '  %s --version\n' "$BIN_DIR/vllm-sr"

  if [ "$MODE" = "serve" ]; then
    printf '  %s serve\n' "$BIN_DIR/vllm-sr"
    printf '  %s serve --platform amd\n' "$BIN_DIR/vllm-sr"
    if [ "$SELECTED_RUNTIME" = "podman" ]; then
      printf '\n'
      printf '%b\n' "${COLOR_WHITE}Runtime${COLOR_RESET}"
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
  parse_args "$@"
  validate_args
  detect_os
  print_logo
  print_install_plan
  install_cli
  ensure_runtime
  print_next_steps
}

main "$@"
