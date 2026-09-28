#!/usr/bin/env bash
# Leak guard for the Decision 2.0 program: fails when text about to be (or
# already) pushed contains an IPv4 address, a node address or hostname, a
# credential pattern, or a literal value of the private secrets file.
#
# Usage:
#   check_no_private.sh [--strict]                          added lines of the staged diff (default)
#   check_no_private.sh [--strict] [--] <path>...           files, directories recursively ("-" = stdin)
#   check_no_private.sh [--strict] --rev <rev> [<path>...]  blobs of a commit's tree
#   check_no_private.sh [--strict] --log <range> [<path>...]
#       every commit in <range>: author, committer, message and added lines
#       (merges: only lines their conflict resolution introduced)
# <path> arguments of --rev/--log are repository-relative pathspecs.
# Run it on the staged changes before every commit.
#
# Output is one line per finding, "<where>:<line>: <category>[,<category>]",
# where <where> is a path, "<rev>:<path>", "<commit>:<path>",
# "<commit>:(message)", "<commit>:(author)" or "<commit>:(committer)". The
# matched text is never printed; a path that itself matches is printed with
# the match replaced by [REDACTED]. Exit status: 0 clean, 1 findings, 2 usage
# or git error.
#
# Private values are read at runtime and never printed:
#   ${DEV2_NODES_FILE:-~/.config/decision2/nodes.env}            <alias>=<user@host>
#   ${DEV2_NODE_NAMES_FILE:-~/.config/decision2/node-names.env}  <alias>=<other name>, e.g. short hostnames
#   ${DEV2_SECRETS_FILE:-~/.config/decision2/secrets.env}        KEY=value
# node-names.env is optional. A missing nodes.env or secrets.env disables its
# check with a warning; --strict makes that an error.
#
# Categories:
#   node-address    a nodes.env or node-names.env value (at least 4 characters),
#                   its host part, or a node IPv4 written with - or _ separators
#   secret-value    a secrets.env value (at least 8 characters), or the host of a URL value
#   ipv4            any other IPv4 address except 0.0.0.0/8, 127.0.0.0/8, 240.0.0.0/4
#                   and 192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24 (documentation),
#                   none of which can name a machine; octets above 255 and dotted runs
#                   of more than four numbers (versions) are not addresses
#   hf-token        hf_ + 20 or more letters/digits
#   apikey-token    apikey_ + 12 or more token characters
#   jev-live-token  jv_live_ + 12 or more token characters
#   github-token    ghp_/gho_/ghs_/ghu_/ghr_ + 16 or more, or github_pat_ + 20 or more
#   private-key     a PEM/OpenSSH/PGP "BEGIN ... PRIVATE KEY" header
# Token bodies made of one repeated character (hf_xxxx..., placeholders) are ignored.
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,/^set -euo pipefail$/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
  exit 0
fi

IFS= read -r -d '' guard_py <<'PY' || true
import bisect
import codecs
import ipaddress
import os
import re
import subprocess
import sys

PROG = "check_no_private"
USAGE = (
    "usage: check_no_private.sh [--strict] [--] [<path>...]\n"
    "       check_no_private.sh [--strict] --rev <rev> [<path>...]\n"
    "       check_no_private.sh [--strict] --log <range> [<path>...]"
)
FLAGS = re.ASCII
NEWLINE = re.compile(r"\n")
IPV4 = re.compile(r"(?<![0-9])(?<![0-9]\.)([0-9]{1,3}(?:\.[0-9]{1,3}){3})(?!\.?[0-9])", FLAGS)
ALLOWED_NETWORKS = [
    ipaddress.ip_network(net)
    for net in (
        "0.0.0.0/8",
        "127.0.0.0/8",
        "240.0.0.0/4",
        "192.0.2.0/24",
        "198.51.100.0/24",
        "203.0.113.0/24",
    )
]
TOKEN_PATTERNS = [
    ("hf-token", re.compile(r"hf_([A-Za-z0-9]{20,})", FLAGS)),
    ("apikey-token", re.compile(r"apikey_([A-Za-z0-9_-]{12,})", FLAGS)),
    ("jev-live-token", re.compile(r"jv_live_([A-Za-z0-9_-]{12,})", FLAGS)),
    ("github-token", re.compile(r"gh[oprsu]_([A-Za-z0-9]{16,})|github_pat_([A-Za-z0-9_]{20,})", FLAGS)),
    ("private-key", re.compile(r"-{5}BEGIN[A-Z0-9 ]* PRIVATE KEY", FLAGS)),
]
HUNK = re.compile(rb"^@@ -[0-9]+(?:,([0-9]+))? \+([0-9]+)(?:,([0-9]+))? @@")


def fail(message):
    print(f"{PROG}: {message}", file=sys.stderr)
    sys.exit(2)


def warn(message):
    print(f"{PROG}: warning: {message}", file=sys.stderr)


def read_pairs(env_var, default, what, strict, optional=False):
    path = os.path.expanduser(os.environ.get(env_var) or default)
    if not os.path.isfile(path):
        if optional:
            return []
        if strict:
            fail(f"{what} file not found: {path}")
        warn(f"{what} file not found ({path}); its check is disabled")
        return []
    pairs = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            key, sep, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if not sep or not key:
                warn(f"{path}:{number} is not KEY=value; ignored")
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            pairs.append((key, value))
    return pairs


def host_part(value):
    host = re.sub(r"^[A-Za-z][A-Za-z0-9+.-]*://", "", value)
    host = host.rsplit("@", 1)[-1].split("/", 1)[0]
    if host.count(":") == 1:
        host = host.split(":", 1)[0]
    return host


def is_ipv4(text):
    return re.fullmatch(r"[0-9]{1,3}(?:\.[0-9]{1,3}){3}", text) is not None


def bounded(terms):
    alternatives = sorted({re.escape(term) for term in terms}, key=len, reverse=True)
    if not alternatives:
        return None
    return re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(alternatives) + r")(?![A-Za-z0-9])", FLAGS | re.IGNORECASE)


def load_detectors(strict):
    node_terms = set()
    for env_var, default, what, optional in (
        ("DEV2_NODES_FILE", "~/.config/decision2/nodes.env", "nodes", False),
        ("DEV2_NODE_NAMES_FILE", "~/.config/decision2/node-names.env", "node names", True),
    ):
        for key, value in read_pairs(env_var, default, what, strict, optional):
            host = host_part(value)
            candidates = {value, host}
            if is_ipv4(host):
                candidates |= {host.replace(".", "-"), host.replace(".", "_")}
            elif "." in host:
                candidates.add(host.split(".", 1)[0])
            for term in candidates:
                if len(term) >= 4:
                    node_terms.add(term)
                elif term == value:
                    warn(f"{what} entry {key} is shorter than 4 characters; not matched")
    secret_values, secret_hosts = set(), set()
    for key, value in read_pairs("DEV2_SECRETS_FILE", "~/.config/decision2/secrets.env", "secrets", strict):
        if len(value) < 8:
            warn(f"secrets entry {key} is shorter than 8 characters; not matched")
            continue
        secret_values.add(value)
        if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value):
            host = host_part(value)
            if len(host) >= 4:
                secret_hosts.add(host)
    detectors = []
    node_re = bounded(node_terms)
    if node_re:
        detectors.append(("node-address", node_re))
    if secret_values:
        alternatives = sorted((re.escape(v) for v in secret_values), key=len, reverse=True)
        detectors.append(("secret-value", re.compile("|".join(alternatives))))
    host_re = bounded(secret_hosts)
    if host_re:
        detectors.append(("secret-value", host_re))
    return detectors


def flagged_ipv4(text):
    octets = [int(part) for part in text.split(".")]
    if max(octets) > 255:
        return False
    address = ipaddress.IPv4Address(".".join(map(str, octets)))
    return not any(address in network for network in ALLOWED_NETWORKS)


def placeholder(match):
    body = next((group for group in match.groups() if group), "")
    return bool(body) and len(set(body.lower())) == 1


def matches(text, detectors):
    """Yield (offset, end, category) for every finding in text."""
    for category, regex in detectors:
        for match in regex.finditer(text):
            yield match.start(), match.end(), category
    for match in IPV4.finditer(text):
        if flagged_ipv4(match.group(1)):
            yield match.start(), match.end(), "ipv4"
    for category, regex in TOKEN_PATTERNS:
        for match in regex.finditer(text):
            if not placeholder(match):
                yield match.start(), match.end(), category


def scan_text(text, detectors):
    """Return {line number: set of categories} for text."""
    found = {}
    newlines = None
    for start, _end, category in matches(text, detectors):
        if newlines is None:
            newlines = [m.end() for m in NEWLINE.finditer(text)]
        found.setdefault(bisect.bisect_right(newlines, start) + 1, set()).add(category)
    return found


def redact(label, detectors):
    spans = sorted((start, end) for start, end, _ in matches(label, detectors))
    if not spans:
        return label, set()
    categories = {category for _, _, category in matches(label, detectors)}
    out, last = [], 0
    for start, end in spans:
        if start >= last:
            out.append(label[last:start])
            out.append("[REDACTED]")
            last = end
        else:
            last = max(last, end)
    out.append(label[last:])
    return "".join(out), categories


def decode(data):
    return data.decode("latin-1")


def git(args, cwd, check=True):
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True)
    if check and result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip().splitlines()
        fail(f"git {args[0]} failed: {detail[-1] if detail else result.returncode}")
    return result


def unquote_path(raw):
    raw = raw.rstrip(b"\t")
    if raw.startswith(b'"') and raw.endswith(b'"'):
        raw = codecs.escape_decode(raw[1:-1])[0]
    return raw.decode("utf-8", "surrogateescape")


def diff_git_path(rest):
    """Path of an "a/<path> b/<path>" pair from a diff --git header (renames are off)."""
    if rest.startswith(b'"'):
        end = 1
        while end < len(rest) and rest[end] != ord('"'):
            end += 2 if rest[end] == ord("\\") else 1
        path = unquote_path(rest[: end + 1])
    else:
        path = rest[: 2 + (len(rest) - 5) // 2].decode("utf-8", "surrogateescape")
    return path[2:] if path.startswith("a/") else path


def added_lines(patch):
    """Parse a -U0 unified diff into [(path, [(new line number, text)])].

    Every added or changed path is listed, also when it has no hunk (empty
    files, mode changes), so that its name is checked; deleted paths are not.
    """
    files, order = {}, []
    path = None
    old_left = new_left = new_line = 0
    for line in patch.split(b"\n"):
        if old_left > 0 or new_left > 0:
            if line.startswith(b"+") and new_left > 0:
                files[path].append((new_line, decode(line[1:])))
                new_line += 1
                new_left -= 1
                continue
            if line.startswith(b"-") and old_left > 0:
                old_left -= 1
                continue
            if line.startswith(b" "):
                new_line += 1
                new_left -= 1
                old_left -= 1
                continue
            if line.startswith(b"\\"):
                continue
            old_left = new_left = 0
        if line.startswith(b"diff --git "):
            path = diff_git_path(line[len(b"diff --git "):])
            if path not in files:
                files[path] = []
                order.append(path)
        elif line.startswith(b"diff "):
            path = None
        elif line.startswith(b"deleted file mode ") and path is not None:
            if not files[path]:
                del files[path]
                order.remove(path)
            path = None
        elif line.startswith(b"+++ "):
            target = line[4:]
            if target == b"/dev/null":
                path = None
            else:
                path = unquote_path(target)
                if path.startswith("b/"):
                    path = path[2:]
                if path not in files:
                    files[path] = []
                    order.append(path)
        elif line.startswith(b"@@ ") and path is not None:
            hunk = HUNK.match(line)
            if hunk:
                old_left = int(hunk.group(1)) if hunk.group(1) is not None else 1
                new_line = int(hunk.group(2))
                new_left = int(hunk.group(3)) if hunk.group(3) is not None else 1
    return [(name, files[name]) for name in order]


DIFF_ARGS = ["--text", "--unified=0", "--no-renames", "--no-color", "--no-ext-diff", "--no-textconv", "--src-prefix=a/", "--dst-prefix=b/"]


class Report:
    def __init__(self, detectors):
        self.detectors = detectors
        self.findings = 0
        self.scanned = 0

    def emit(self, where, found):
        if not found:
            return
        for line in sorted(found):
            print(f"{where}:{line}: {','.join(sorted(found[line]))}")
            self.findings += 1

    def label(self, prefix, path):
        """Report a path that itself leaks, and return a printable form of it."""
        shown, categories = redact(path, self.detectors)
        if categories:
            print(f"{prefix}{shown}:(path): {','.join(sorted(categories))}")
            self.findings += 1
        return prefix + shown

    def text(self, where, text):
        self.scanned += 1
        self.emit(where, scan_text(text, self.detectors))

    def lines(self, where, numbered):
        self.scanned += 1
        if not numbered:
            return
        found = scan_text("\n".join(text for _, text in numbered), self.detectors)
        self.emit(where, {numbered[index - 1][0]: categories for index, categories in found.items()})


def toplevel():
    result = git(["rev-parse", "--show-toplevel"], cwd=None, check=False)
    if result.returncode != 0:
        fail("not inside a git repository")
    return result.stdout.decode().strip()


def scan_staged(report):
    top = toplevel()
    patch = git(["-c", "core.quotePath=false", "diff", "--cached", *DIFF_ARGS], cwd=top).stdout
    for path, numbered in added_lines(patch):
        report.lines(report.label("", path), numbered)


def walk(paths):
    for path in paths:
        if os.path.isdir(path) and not os.path.islink(path):
            for root, dirs, files in os.walk(path):
                dirs[:] = sorted(d for d in dirs if d != ".git")
                for name in sorted(files):
                    yield os.path.join(root, name)
        elif os.path.lexists(path):
            yield path
        else:
            fail(f"no such file or directory: {path}")


def scan_paths(report, paths):
    if "-" in paths:
        report.text("(stdin)", decode(sys.stdin.buffer.read()))
        paths = [path for path in paths if path != "-"]
    for path in walk(paths):
        where = report.label("", path)
        if os.path.islink(path):
            report.text(where, os.readlink(path))
            continue
        with open(path, "rb") as handle:
            report.text(where, decode(handle.read()))


def scan_rev(report, rev, paths):
    top = toplevel()
    commit = git(["rev-parse", "--verify", "--quiet", rev + "^{commit}"], cwd=top).stdout.decode().strip()
    listing = git(["ls-tree", "-r", "-z", commit, "--", *paths], cwd=top).stdout
    blobs = []
    for record in listing.split(b"\0"):
        if not record:
            continue
        meta, _, path = record.partition(b"\t")
        _mode, kind, obj = meta.split(b" ")
        if kind == b"blob":
            blobs.append((path.decode("utf-8", "surrogateescape"), obj))
    batch = subprocess.Popen(["git", "cat-file", "--batch"], cwd=top, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for path, obj in blobs:
            batch.stdin.write(obj + b"\n")
            batch.stdin.flush()
            header = batch.stdout.readline().split()
            size = int(header[2])
            data = batch.stdout.read(size)
            batch.stdout.read(1)
            report.text(report.label(f"{rev}:", path), decode(data))
    finally:
        batch.stdin.close()
        batch.wait()


def scan_log(report, revision_range, paths):
    top = toplevel()
    commits = git(["rev-list", "--reverse", revision_range], cwd=top).stdout.decode().split()
    merge_mode = "--diff-merges=remerge"
    for commit in commits:
        meta = git(["show", "-s", "--format=%an <%ae>%x00%cn <%ce>%x00%B", commit], cwd=top).stdout
        author, committer, message = (decode(part) for part in meta.split(b"\0", 2))
        report.text(f"{commit}:(author)", author)
        report.text(f"{commit}:(committer)", committer)
        report.text(f"{commit}:(message)", message)
        args = ["-c", "core.quotePath=false", "show", "--format=", "--patch", *DIFF_ARGS, merge_mode, commit, "--", *paths]
        result = git(args, cwd=top, check=False)
        if result.returncode != 0 and merge_mode == "--diff-merges=remerge":
            merge_mode = "--diff-merges=first-parent"
            args[args.index("--diff-merges=remerge")] = merge_mode
            result = git(args, cwd=top, check=False)
        if result.returncode != 0:
            fail(f"git show {commit} failed")
        for path, numbered in added_lines(result.stdout):
            report.lines(report.label(f"{commit}:", path), numbered)


def main(argv):
    strict = False
    mode, target = "staged", None
    paths = []
    args = list(argv)
    while args:
        arg = args.pop(0)
        if arg == "--strict":
            strict = True
        elif arg in ("--rev", "--log"):
            if mode != "staged" or not args:
                fail(USAGE)
            mode, target = arg[2:], args.pop(0)
        elif arg == "--":
            paths.extend(args)
            args = []
        elif arg.startswith("-") and arg != "-":
            fail(f"unknown option {arg}\n{USAGE}")
        else:
            paths.append(arg)
    report = Report(load_detectors(strict))
    if mode == "rev":
        scan_rev(report, target, paths)
    elif mode == "log":
        scan_log(report, target, paths)
    elif paths:
        scan_paths(report, paths)
    else:
        scan_staged(report)
    sys.stdout.flush()
    if report.findings:
        print(f"{PROG}: {report.findings} finding(s); matched text is not shown", file=sys.stderr)
        return 1
    print(f"{PROG}: clean ({report.scanned} item(s) scanned)", file=sys.stderr)
    return 0


sys.exit(main(sys.argv[1:]))
PY

exec python3 -c "$guard_py" "$@"
