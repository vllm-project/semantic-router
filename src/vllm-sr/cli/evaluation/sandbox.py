"""Fail-closed Linux sandbox for the fixed Dashboard evaluation worker."""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import os
import resource
import stat
from dataclasses import dataclass
from pathlib import Path

_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1
_LANDLOCK_REQUIRED_ABI = 4

_FS_EXECUTE = 1 << 0
_FS_WRITE_FILE = 1 << 1
_FS_READ_FILE = 1 << 2
_FS_READ_DIR = 1 << 3
_FS_REMOVE_DIR = 1 << 4
_FS_REMOVE_FILE = 1 << 5
_FS_MAKE_CHAR = 1 << 6
_FS_MAKE_DIR = 1 << 7
_FS_MAKE_REG = 1 << 8
_FS_MAKE_SOCK = 1 << 9
_FS_MAKE_FIFO = 1 << 10
_FS_MAKE_BLOCK = 1 << 11
_FS_MAKE_SYM = 1 << 12
_FS_REFER = 1 << 13
_FS_TRUNCATE = 1 << 14
_FS_READ = _FS_READ_FILE | _FS_READ_DIR
_FS_WRITE = (
    _FS_READ
    | _FS_WRITE_FILE
    | _FS_REMOVE_DIR
    | _FS_REMOVE_FILE
    | _FS_MAKE_DIR
    | _FS_MAKE_REG
    | _FS_MAKE_FIFO
    | _FS_MAKE_SYM
    | _FS_REFER
    | _FS_TRUNCATE
)
_FS_HANDLED = _FS_WRITE | _FS_EXECUTE | _FS_MAKE_CHAR | _FS_MAKE_SOCK | _FS_MAKE_BLOCK

_NET_BIND_TCP = 1 << 0
_NET_CONNECT_TCP = 1 << 1
_NET_HANDLED = _NET_BIND_TCP | _NET_CONNECT_TCP

_PR_SET_DUMPABLE = 4
_PR_SET_NO_NEW_PRIVS = 38
_SCMP_ACT_ALLOW = 0x7FFF0000
_SCMP_ACT_ERRNO = 0x00050000
_SCMP_CMP_MASKED_EQ = 7
_CLONE_THREAD = 0x00010000
_WORKER_TASK_HEADROOM = 256
_DENIED_SYSCALLS = (
    "execve",
    "execveat",
    "fork",
    "vfork",
    "ptrace",
    "mount",
    "umount2",
    "pivot_root",
    "chroot",
    "unshare",
    "setns",
    "bpf",
    "perf_event_open",
    "keyctl",
    "add_key",
    "request_key",
    "open_by_handle_at",
    "process_vm_readv",
    "process_vm_writev",
    "pidfd_getfd",
    "kexec_load",
    "init_module",
    "finit_module",
    "delete_module",
    "reboot",
    "swapon",
    "swapoff",
    "userfaultfd",
    "io_uring_setup",
    "io_uring_enter",
    "io_uring_register",
    "socket",
    "socketpair",
    "connect",
    "bind",
    "listen",
    "accept",
    "accept4",
    "sendto",
    "recvfrom",
    "sendmsg",
    "recvmsg",
    "sendmmsg",
    "recvmmsg",
    "shutdown",
    "getsockname",
    "getpeername",
    "setsockopt",
    "getsockopt",
    "kill",
    "tkill",
    "tgkill",
    "chmod",
    "fchmod",
    "fchmodat",
    "fchmodat2",
    "chown",
    "fchown",
    "fchownat",
    "lchown",
    "utime",
    "utimes",
    "futimesat",
    "utimensat",
    "setxattr",
    "lsetxattr",
    "fsetxattr",
    "removexattr",
    "lremovexattr",
    "fremovexattr",
    "ioctl",
)


class SandboxUnavailableError(RuntimeError):
    """The host cannot provide the mandatory worker isolation contract."""


class _RulesetAttr(ctypes.Structure):
    _fields_ = [
        ("handled_access_fs", ctypes.c_uint64),
        ("handled_access_net", ctypes.c_uint64),
    ]


class _PathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
    ]


class _SeccompArgCompare(ctypes.Structure):
    _fields_ = [
        ("arg", ctypes.c_uint),
        ("op", ctypes.c_uint),
        ("datum_a", ctypes.c_uint64),
        ("datum_b", ctypes.c_uint64),
    ]


@dataclass(frozen=True)
class WorkerSandboxPolicy:
    writable_root: Path
    suite_store: Path
    readable_roots: tuple[Path, ...]
    cpu_seconds: int
    address_space_bytes: int = 4 * 1024 * 1024 * 1024
    output_bytes: int = 1024 * 1024 * 1024


def apply_worker_sandbox(policy: WorkerSandboxPolicy) -> None:
    """Constrain resources, filesystem, processes, and all network syscalls."""

    _validate_policy(policy)
    os.umask(0o077)
    os.chdir(policy.writable_root)
    _apply_resource_limits(policy)
    _apply_seccomp()
    _apply_landlock(policy)


def _validate_policy(policy: WorkerSandboxPolicy) -> None:
    if policy.cpu_seconds < 1 or policy.address_space_bytes < 512 * 1024 * 1024:
        raise SandboxUnavailableError("worker resource policy is invalid")
    if policy.output_bytes < 1:
        raise SandboxUnavailableError("worker output limit is invalid")
    for path in (policy.writable_root, policy.suite_store, *policy.readable_roots):
        if not path.is_absolute() or path.is_symlink() or not path.exists():
            raise SandboxUnavailableError("worker sandbox path is missing or unsafe")


def _apply_resource_limits(policy: WorkerSandboxPolicy) -> None:
    task_limit = _worker_task_limit()
    limits = {
        resource.RLIMIT_CORE: 0,
        resource.RLIMIT_FSIZE: policy.output_bytes,
        resource.RLIMIT_NOFILE: 128,
        resource.RLIMIT_NPROC: task_limit,
        resource.RLIMIT_CPU: policy.cpu_seconds,
        resource.RLIMIT_AS: policy.address_space_bytes,
    }
    for key, value in limits.items():
        _soft, hard = resource.getrlimit(key)
        effective = value if hard == resource.RLIM_INFINITY else min(value, hard)
        resource.setrlimit(key, (effective, effective))
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
        raise SandboxUnavailableError("could not disable worker process dumps")
    if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise SandboxUnavailableError("could not enable worker no-new-privileges")


def _worker_task_limit() -> int:
    """Return a per-UID task ceiling that leaves bounded worker headroom.

    Linux applies RLIMIT_NPROC to every task owned by the real UID, not just to
    descendants of this worker. A fixed low ceiling therefore fails closed on
    a busy shared host before the worker creates even one legitimate thread.
    Count the UID's existing tasks before isolation and reserve a fixed amount
    of additional capacity; seccomp separately prevents process creation.
    """

    try:
        real_uid = os.getuid()
        current_tasks = 0
        for process in Path("/proc").iterdir():
            if not process.name.isdecimal():
                continue
            try:
                status = (process / "status").read_text(encoding="utf-8")
                owner_line = next(
                    line for line in status.splitlines() if line.startswith("Uid:")
                )
                if int(owner_line.split()[1]) != real_uid:
                    continue
                current_tasks += sum(
                    1 for task in (process / "task").iterdir() if task.name.isdecimal()
                )
            except (FileNotFoundError, PermissionError, StopIteration, ValueError):
                continue
    except OSError as exc:
        raise SandboxUnavailableError("could not inspect worker task usage") from exc
    if current_tasks < 1:
        raise SandboxUnavailableError("could not determine worker task usage")
    return current_tasks + _WORKER_TASK_HEADROOM


def _landlock_syscalls() -> tuple[int, int, int]:
    # Linux uses the generic numbering on every Dashboard target architecture.
    if os.uname().machine not in {
        "x86_64",
        "aarch64",
        "arm64",
        "riscv64",
        "s390x",
        "ppc64",
        "ppc64le",
        "i386",
        "i686",
    }:
        raise SandboxUnavailableError(
            "worker architecture has no reviewed Landlock syscall contract"
        )
    return 444, 445, 446


def _checked_syscall(libc: ctypes.CDLL, number: int, *args: object) -> int:
    result = int(libc.syscall(number, *args))
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def _apply_landlock(policy: WorkerSandboxPolicy) -> None:
    create_ruleset, add_rule, restrict_self = _landlock_syscalls()
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        abi = _checked_syscall(
            libc,
            create_ruleset,
            ctypes.c_void_p(),
            ctypes.c_size_t(),
            ctypes.c_uint(_LANDLOCK_CREATE_RULESET_VERSION),
        )
    except OSError as exc:
        raise SandboxUnavailableError(
            "Landlock is unavailable for the evaluation worker"
        ) from exc
    if abi < _LANDLOCK_REQUIRED_ABI:
        raise SandboxUnavailableError(
            f"Landlock ABI {_LANDLOCK_REQUIRED_ABI} is required for worker TCP isolation"
        )
    attr = _RulesetAttr(_FS_HANDLED, _NET_HANDLED)
    try:
        ruleset_fd = _checked_syscall(
            libc,
            create_ruleset,
            ctypes.byref(attr),
            ctypes.sizeof(attr),
            ctypes.c_uint(),
        )
    except OSError as exc:
        raise SandboxUnavailableError(
            "could not create worker Landlock ruleset"
        ) from exc
    try:
        _add_path_rule(libc, add_rule, ruleset_fd, policy.writable_root, _FS_WRITE)
        _add_path_rule(libc, add_rule, ruleset_fd, policy.suite_store, _FS_READ)
        for path in policy.readable_roots:
            _add_path_rule(libc, add_rule, ruleset_fd, path, _FS_READ)
        _checked_syscall(
            libc,
            restrict_self,
            ruleset_fd,
            ctypes.c_uint(),
        )
    except OSError as exc:
        raise SandboxUnavailableError(
            "could not enforce worker Landlock rules"
        ) from exc
    finally:
        os.close(ruleset_fd)


def _add_path_rule(
    libc: ctypes.CDLL,
    add_rule: int,
    ruleset_fd: int,
    path: Path,
    access: int,
) -> None:
    descriptor = os.open(path, os.O_PATH | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        effective_access = access
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            effective_access &= (
                _FS_EXECUTE | _FS_WRITE_FILE | _FS_READ_FILE | _FS_TRUNCATE
            )
        if effective_access == 0:
            raise SandboxUnavailableError(
                "worker path rule grants no applicable access"
            )
        attr = _PathBeneathAttr(effective_access, descriptor, 0)
        _checked_syscall(
            libc,
            add_rule,
            ruleset_fd,
            ctypes.c_int(_LANDLOCK_RULE_PATH_BENEATH),
            ctypes.byref(attr),
            ctypes.c_uint(),
        )
    finally:
        os.close(descriptor)


def _configure_seccomp_library(seccomp: ctypes.CDLL) -> None:
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
    seccomp.seccomp_rule_add_array.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(_SeccompArgCompare),
    ]
    seccomp.seccomp_rule_add_array.restype = ctypes.c_int
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_load.restype = ctypes.c_int
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]


def _install_seccomp_rules(seccomp: ctypes.CDLL, context: int, deny: int) -> None:
    for name in _DENIED_SYSCALLS:
        _seccomp_rule(seccomp, context, deny, name)
    _seccomp_rule(
        seccomp,
        context,
        _SCMP_ACT_ERRNO | errno.ENOSYS,
        "clone3",
    )
    _seccomp_rule(
        seccomp,
        context,
        deny,
        "clone",
        _SeccompArgCompare(0, _SCMP_CMP_MASKED_EQ, _CLONE_THREAD, 0),
    )


def _apply_seccomp() -> None:
    seccomp = _load_seccomp_library()
    _configure_seccomp_library(seccomp)
    context = seccomp.seccomp_init(_SCMP_ACT_ALLOW)
    if not context:
        raise SandboxUnavailableError("could not initialize worker seccomp policy")
    deny = _SCMP_ACT_ERRNO | errno.EPERM
    try:
        _install_seccomp_rules(seccomp, context, deny)
        if seccomp.seccomp_load(context) != 0:
            raise SandboxUnavailableError("could not load worker seccomp policy")
    finally:
        seccomp.seccomp_release(context)


def _load_seccomp_library() -> ctypes.CDLL:
    """Load libseccomp without depending on PATH inside the sealed worker env."""

    discovered = ctypes.util.find_library("seccomp")
    candidates = tuple(
        dict.fromkeys(
            candidate
            for candidate in (discovered, "libseccomp.so.2", "libseccomp.so")
            if candidate
        )
    )
    for candidate in candidates:
        try:
            return ctypes.CDLL(candidate, use_errno=True)
        except OSError:
            continue
    raise SandboxUnavailableError("libseccomp is required for the evaluation worker")


def _seccomp_rule(
    seccomp: ctypes.CDLL,
    context: int,
    action: int,
    name: str,
    *comparisons: _SeccompArgCompare,
) -> None:
    syscall_number = seccomp.seccomp_syscall_resolve_name(name.encode("ascii"))
    if syscall_number < 0:
        return
    if comparisons:
        array = (_SeccompArgCompare * len(comparisons))(*comparisons)
        pointer = ctypes.cast(array, ctypes.POINTER(_SeccompArgCompare))
    else:
        pointer = ctypes.POINTER(_SeccompArgCompare)()
    if (
        seccomp.seccomp_rule_add_array(
            context,
            ctypes.c_uint32(action),
            syscall_number,
            len(comparisons),
            pointer,
        )
        != 0
    ):
        raise SandboxUnavailableError(f"could not add worker seccomp rule for {name}")
