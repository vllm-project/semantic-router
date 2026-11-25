"""Bind frozen raw development instructions to the prompts used by a solver."""

from pathlib import Path
from string import Formatter

if __package__:
    from .loop_evidence import check, digest, sha
    from .schedule import parse
else:
    from loop_evidence import check, digest, sha
    from schedule import parse


def source_file(root, name):
    check(
        isinstance(name, str) and name and "\\" not in name, "invalid dev source path"
    )
    rel = Path(name)
    check(not rel.is_absolute() and ".." not in rel.parts, "invalid dev source path")
    path = (root / rel).resolve(strict=True)
    check(
        path.is_relative_to(root) and path.is_file(),
        "source outside dev manifest directory",
    )
    return path


def prompt_identities(path, fingerprint, study, allocated):
    raw = path.read_bytes()
    check(digest(raw) == sha(fingerprint), "target dev manifest fingerprint mismatch")
    doc = parse(raw)
    check(
        isinstance(doc, dict)
        and set(doc) == {"schema", "study", "split", "template", "tasks"}
        and doc["schema"]
        in ("recipe-lab-target-dev-manifest-v1", "recipe-lab-target-dev-manifest-v2")
        and doc["study"] == study
        and doc["split"] == "dev",
        "incompatible target dev manifest",
    )
    swe = doc["schema"] == "recipe-lab-target-dev-manifest-v1"
    template = doc["template"]
    check(
        isinstance(template, dict)
        and set(template) == {"file", "sha256"}
        and (not swe or template["file"] == "prompt-template.txt"),
        "invalid template reference",
    )
    tasks = doc["tasks"]
    file_key = "statement_file" if swe else "instruction_file"
    keys = {"instruction_sha256", file_key}
    if swe:
        keys.add("repo")
    check(
        isinstance(tasks, list)
        and all(
            isinstance(t, dict)
            and set(t) == keys
            and (not swe or (isinstance(t["repo"], str) and t["repo"]))
            for t in tasks
        ),
        "invalid dev task binding",
    )
    ids = {sha(t["instruction_sha256"]) for t in tasks}
    check(
        ids == allocated and len(ids) == len(tasks) and ids,
        "manifest must exactly match the frozen development allocation",
    )
    # No template or task text is opened until the entire manifest is pinned
    # and all of its raw identities have been checked against the dev allocation.
    root = path.resolve().parent
    template_bytes = source_file(root, template["file"]).read_bytes()
    check(digest(template_bytes) == sha(template["sha256"]), "solver template changed")
    fmt = template_bytes.decode("utf-8")
    if swe:
        # The original SWE adapter reads the template in universal-newline mode.
        fmt = fmt.replace("\r\n", "\n").replace("\r", "\n")
    fields = {
        (key, spec, conversion)
        for _, key, spec, conversion in Formatter().parse(fmt)
        if key is not None
    }
    expected = ("repo", "problem_statement") if swe else ("instruction",)
    check(fields == {(key, "", None) for key in expected}, "unexpected template fields")
    identities = {}
    for task in tasks:
        statement = source_file(root, task[file_key]).read_bytes()
        check(
            digest(statement) == task["instruction_sha256"],
            "dev statement fingerprint mismatch",
        )
        text = statement.decode("utf-8")
        bindings = (
            {"repo": task["repo"], "problem_statement": text}
            if swe
            else {"instruction": text}
        )
        prompt = fmt.format(**bindings)
        identity = digest(prompt.encode("utf-8"))
        check(identity not in identities, "duplicate rendered prompt")
        identities[identity] = {"raw": task["instruction_sha256"]}
        if swe:
            identities[identity]["repo"] = task["repo"]
    return identities, {
        "method": "swe_solver_template" if swe else "instruction_template",
        **({} if swe else {"template_file": template["file"]}),
        "manifest_sha256": digest(raw),
        "template_sha256": template["sha256"],
    }
