"""GitHub transport fixture with independent sibling-attempt visibility."""

from __future__ import annotations

import copy
from collections import Counter

import test_pr_ci_status as status


class PendingClient(status.Client):
    def __init__(self):
        super().__init__()
        self.plan.update(deferred_images=[])
        self.refresh_report()
        self.run.update(repository={"full_name": status.REPO}, check_suite_id=20)
        self.community = {
            **copy.deepcopy(self.run),
            "id": 202,
            "workflow_id": 3,
            "path": ".github/workflows/community.yml",
            "name": "Community",
            "run_attempt": 2,
            "check_suite_id": 19,
        }
        self.checks = [
            {
                "id": 80,
                "name": "PR Gate",
                "head_sha": status.HEAD,
                "app": {"id": 15368},
                "external_id": "semantic-router-ci:3:101:1",
                "status": "in_progress",
                "conclusion": None,
                "check_suite": {"id": 19},
                "output": {"title": "Waiting for committer approval — /ci run"},
                "run_attempt": 1,
            }
        ]
        self.visible = set()
        self.new_check_visible = True
        self.reads = Counter()
        self.on_read = None
        self.after_write = None
        self.extra_pulls = []
        self.post_attempts = 0
        self.post_error = None
        self.post_before_error = None

    def event(self):
        return {"action": "completed", "workflow_run": copy.deepcopy(self.community)}

    def request(self, endpoint, **options):
        assert options.get("method", "GET") == "GET"
        path = endpoint.removeprefix(f"repos/{status.REPO}/")
        self.reads[path] += 1
        if self.on_read:
            self.on_read(path, self)
        if path == "actions/runs/202":
            return copy.deepcopy(self.community)
        if path == "actions/runs/102":
            return {**copy.deepcopy(self.run), "id": 102}
        if path == "actions/workflows/community.yml":
            return {"id": 3, "path": ".github/workflows/community.yml"}
        if path.startswith("pulls?"):
            return copy.deepcopy([self.pull, *self.extra_pulls])
        if path.startswith("pulls/"):
            for pull in self.extra_pulls:
                if path == f"pulls/{pull['number']}":
                    return copy.deepcopy(pull)
        if path.startswith("check-runs/"):
            identity = int(path.rsplit("/", 1)[1])
            return copy.deepcopy(next(c for c in self.checks if c["id"] == identity))
        if path.startswith(f"commits/{status.HEAD}/check-runs?"):
            items = self.checks
            if "filter=latest" in path:
                items = [c for c in items if c["id"] in self.visible]
            return {"check_runs": copy.deepcopy(items)}
        return super().request(endpoint, **options)

    def write(self, endpoint, payload, **options):
        if "/statuses/" in endpoint:
            return super().write(endpoint, payload, **options)
        assert "/check-runs" in endpoint, "Pending recovery must not write comments"
        if endpoint.endswith("/check-runs"):
            self.post_attempts += 1
            if self.post_before_error:
                error, self.post_before_error = self.post_before_error, None
                raise error
        value = super().write(endpoint, copy.deepcopy(payload), **options)
        if endpoint.endswith("/check-runs"):
            self.checks[-1]["check_suite"] = {"id": 19}
            if self.new_check_visible:
                self.visible.add(value["id"])
            if self.post_error:
                error, self.post_error = self.post_error, None
                raise error
            value = copy.deepcopy(self.checks[-1])
        if self.after_write:
            self.after_write(endpoint, payload, self)
        return value
