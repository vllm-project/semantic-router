#!/usr/bin/env python3
"""Authorize explicit previews and upload prebuilt static files as Netlify drafts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

API_PAGE_SIZE = 100


class PreviewError(RuntimeError):
    """A safe, credential-free error for workflow logs and summaries."""


class GitHubClient:
    def request(self, endpoint: str, *, method: str = "GET", payload=None):
        command = ["gh", "api", endpoint, "--method", method]
        if payload is not None:
            command += ["--input", "-"]
        result = subprocess.run(
            command,
            input=json.dumps(payload) if payload is not None else None,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise PreviewError("GitHub API request failed; check workflow permissions.")
        return json.loads(result.stdout) if result.stdout.strip() else None


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class NetlifyClient:
    def __init__(self, token: str):
        self.token = token
        self.opener = build_opener(NoRedirect)

    def request(self, endpoint: str, *, method: str = "GET", payload=None, data=None):
        content_type = "application/octet-stream"
        if payload is not None:
            data = json.dumps(payload).encode()
            content_type = "application/json"
        request = Request(
            f"https://api.netlify.com/api/v1/{endpoint}",
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": content_type,
                "User-Agent": "vllm-sr-preview",
            },
        )
        try:
            with self.opener.open(request, timeout=30) as response:
                result = json.load(response)
                if not isinstance(result, dict):
                    raise ValueError("Expected a Netlify API object")
                return result
        except HTTPError as error:
            raise PreviewError(f"Netlify API returned HTTP {error.code}.") from None
        except (URLError, OSError, ValueError):
            raise PreviewError(
                "Netlify API request failed or returned invalid JSON."
            ) from None


def summary(message: str) -> None:
    print(message)
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(path, "a", encoding="utf-8") as output:
            output.write(message + "\n")


def outputs(values: dict) -> None:
    if path := os.environ.get("GITHUB_OUTPUT"):
        with open(path, "a", encoding="utf-8") as output:
            for key, value in values.items():
                output.write(f"{key}={value}\n")


def run_url(repo: str) -> str:
    return (
        f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{repo}"
        f"/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    )


def status(
    client, repo: str, number: int, sha: str, state: str, message: str, url=None
):
    client.request(
        f"repos/{repo}/statuses/{sha}",
        method="POST",
        payload={
            "context": f"netlify-preview/PR-{number}",
            "state": state,
            "description": message,
            "target_url": url or run_url(repo),
        },
    )


def current_pull(client, repo: str, number: int):
    pull = client.request(f"repos/{repo}/pulls/{number}")
    if (
        pull.get("state") != "open"
        or pull.get("base", {}).get("ref") != "main"
        or pull.get("base", {}).get("repo", {}).get("full_name") != repo
    ):
        return None
    return pull


def already_requested(client, repo: str, number: int, sha: str) -> bool:
    """Statuses are newest first; inspect every page until this PR's latest status."""
    page = 1
    while True:
        entries = client.request(
            f"repos/{repo}/commits/{sha}/statuses?per_page={API_PAGE_SIZE}&page={page}"
        )
        for entry in entries:
            if entry.get("context") != f"netlify-preview/PR-{number}":
                continue
            if entry.get("state") == "success":
                return True
            if entry.get("state") != "pending":
                return False
            prefix = run_url(repo).rsplit("/", 1)[0] + "/"
            target = entry.get("target_url") or ""
            if (
                target == run_url(repo)
                and int(os.environ.get("GITHUB_RUN_ATTEMPT", "1")) > 1
            ):
                return False  # A cancelled prior attempt can leave its status pending.
            if target.startswith(prefix) and target[len(prefix) :].isdigit():
                run = client.request(
                    f"repos/{repo}/actions/runs/{target[len(prefix):]}"
                )
                return run.get("status") != "completed"
            return True
        if len(entries) < API_PAGE_SIZE:
            return False
        page += 1


def credentials() -> tuple[str, str]:
    token, site = os.environ.get("NETLIFY_AUTH_TOKEN"), os.environ.get(
        "NETLIFY_SITE_ID"
    )
    if not token or not site:
        raise PreviewError(
            "Configure NETLIFY_AUTH_TOKEN and NETLIFY_SITE_ID before requesting a preview."
        )
    return token, site


def authorize(client, event: dict) -> dict:
    result = {"build": "false"}
    outputs(result)
    issue, comment = event.get("issue", {}), event.get("comment", {})
    actor = event.get("sender", {}).get("login")
    repo = os.environ["GITHUB_REPOSITORY"]
    if (
        os.environ.get("GITHUB_EVENT_NAME") != "issue_comment"
        or event.get("action") != "created"
        or not issue.get("pull_request")
        or comment.get("body") != "/netlify"
        or not actor
        or comment.get("user", {}).get("login") != actor
        or event.get("repository", {}).get("full_name") != repo
    ):
        summary("No preview requested: use a new `/netlify` comment on an open PR.")
        return result
    permission = client.request(
        f"repos/{repo}/collaborators/{quote(actor, safe='')}/permission"
    )
    if permission.get("permission") not in {"write", "maintain", "admin"}:
        summary("Preview skipped: the commenter must have repository write access.")
        return result
    number = issue["number"]
    pull = current_pull(client, repo, number)
    if pull is None:
        summary(
            "Preview skipped: the PR must be open and target this repository's main branch."
        )
        return result
    sha, head_repo = pull["head"]["sha"], (pull["head"].get("repo") or {}).get(
        "full_name", ""
    )
    if not re.fullmatch(r"[0-9a-f]{40}", sha) or not re.fullmatch(
        r"[\w.-]+/[\w.-]+", head_repo
    ):
        raise PreviewError("The PR head is unavailable or invalid.")
    credentials()
    if already_requested(client, repo, number, sha):
        summary(
            f"Preview skipped: PR #{number} already has a pending or successful preview for `{sha[:12]}`."
        )
        return result
    status(client, repo, number, sha, "pending", "Building requested Netlify preview")
    result = {"build": "true", "sha": sha, "number": number, "repository": head_repo}
    outputs(result)
    summary(f"Authorized Netlify preview for PR #{number} at `{sha[:12]}`.")
    return result


def static_files(directory: Path) -> tuple[dict, dict]:
    """Only regular static files cross from the untrusted build into this job."""
    if directory.is_symlink() or not directory.is_dir():
        raise PreviewError("Static build directory is missing or is a symlink.")
    files, by_digest = {}, {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise PreviewError("Static build contains a symlink or a non-regular file.")
        if path.is_dir():
            continue
        name = "/" + path.relative_to(directory).as_posix()
        if any(character in name for character in "?#\r\n\\"):
            raise PreviewError("Static build contains an unsupported file name.")
        hasher = hashlib.sha1()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        files[name], by_digest[digest] = digest, (name, path)
    if "/index.html" not in files:
        raise PreviewError("Static build must contain index.html.")
    return files, by_digest


def wait_ready(client, deploy: dict, *, attempts=100, delay=3) -> dict:
    deploy_id = quote(deploy["id"], safe="")
    for attempt in range(attempts):
        if deploy.get("state") == "ready":
            return deploy
        if deploy.get("state") in {"error", "rejected"}:
            raise PreviewError("Netlify could not process the draft deployment.")
        if attempt:
            time.sleep(delay)
        deploy = client.request(f"deploys/{deploy_id}")
    if deploy.get("state") == "ready":
        return deploy
    raise PreviewError("Timed out waiting for the Netlify draft deployment.")


def deploy_static(client, site: str, directory: Path, number: int, sha: str) -> str:
    files, by_digest = static_files(directory)
    # Documented file-digest API: draft prevents updating the production deploy.
    # https://docs.netlify.com/api-and-cli-guides/api-guides/get-started-with-api/
    query = urlencode({"title": f"PR #{number} ({sha[:12]})"})
    deploy = client.request(
        f"sites/{quote(site, safe='')}/deploys?{query}",
        method="POST",
        payload={
            "files": files,
            "draft": True,
            "branch": f"deploy-preview-{number}",
        },
    )
    deploy_id = quote(deploy["id"], safe="")
    for digest in set(deploy.get("required", [])):
        if digest not in by_digest:
            raise PreviewError(
                "Netlify requested a file outside the static build manifest."
            )
        name, path = by_digest[digest]
        client.request(
            f"deploys/{deploy_id}/files/{quote(name.lstrip('/'), safe='/')}",
            method="PUT",
            data=path.read_bytes(),
        )
    deploy = wait_ready(client, deploy)
    url = deploy.get("deploy_ssl_url") or deploy.get("deploy_url") or ""
    parsed = urlparse(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or any(c.isspace() for c in url)
    ):
        raise PreviewError("Netlify did not return a valid draft preview URL.")
    return url


def publish(client, directory: Path, netlify=None) -> None:
    repo, sha = os.environ["GITHUB_REPOSITORY"], os.environ["SHA"]
    number = int(os.environ["PR_NUMBER"])
    state = "failure"
    try:
        if os.environ.get("BUILD_RESULT") != "success":
            raise PreviewError(
                "Website build did not succeed; request /netlify again after fixing it."
            )
        pull = current_pull(client, repo, number)
        if pull is None or pull["head"]["sha"] != sha:
            state = "error"
            raise PreviewError(
                "PR closed or head changed; request /netlify for the current revision."
            )
        token, site = credentials()
        url = deploy_static(
            netlify or NetlifyClient(token), site, directory, number, sha
        )
        status(client, repo, number, sha, "success", "Netlify preview ready", url)
        summary(
            f"Netlify preview for PR #{number} (`{sha[:12]}`): [Open preview]({url})"
        )
    except (PreviewError, OSError, ValueError, KeyError, TypeError):
        status(
            client,
            repo,
            number,
            sha,
            state,
            "Netlify preview failed; see workflow logs",
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("authorize").add_argument("--event", type=Path, required=True)
    commands.add_parser("publish").add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    try:
        client = GitHubClient()
        if args.command == "authorize":
            authorize(client, json.loads(args.event.read_text()))
        else:
            publish(client, args.directory)
    except PreviewError as error:
        summary(str(error))
        return 1
    except (OSError, ValueError, KeyError, TypeError):
        summary(
            "Preview workflow failed: invalid input, API response, or unreadable build artifact."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
