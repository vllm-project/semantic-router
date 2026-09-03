// SPDX-License-Identifier: Apache-2.0
// Tests for unassign-inactive.js. All GitHub API calls are mocked.

"use strict";

const {
  getLastHumanActivityDate,
  buildWarningComment,
  buildUnassignComment,
  WARNING_MARKER,
  WARN_LABEL,
  processAssignee,
  warningCommentExists,
  findLinkedPRActivity,
  ensureLabel,
  run,
} = require("../unassign-inactive");

// -- Helpers ----------------------------------------------------------------

function daysAgo(n) {
  const d = new Date();
  d.setDate(d.getDate() - n);
  return d.toISOString();
}

function makeIssue(overrides = {}) {
  return {
    number: 42,
    title: "Test issue",
    created_at: daysAgo(60),
    labels: [],
    assignees: [{ login: "alice" }],
    pull_request: undefined,
    ...overrides,
  };
}

function makeOctokit(overrides = {}) {
  return {
    paginate: jest.fn(async (fn, params) => {
      const result = await fn(params);
      return result.data || result || [];
    }),
    rest: {
      issues: {
        getLabel: jest.fn().mockResolvedValue({}),
        createLabel: jest.fn().mockResolvedValue({}),
        addLabels: jest.fn().mockResolvedValue({}),
        removeLabel: jest.fn().mockResolvedValue({}),
        createComment: jest.fn().mockResolvedValue({}),
        removeAssignees: jest.fn().mockResolvedValue({}),
        listComments: jest.fn().mockResolvedValue({ data: [] }),
        listForRepo: jest.fn().mockResolvedValue({ data: [] }),
        listEventsForTimeline: jest.fn().mockResolvedValue({ data: [] }),
      },
      search: {
        issuesAndPullRequests: jest.fn().mockResolvedValue({ data: { items: [] } }),
      },
    },
    ...overrides,
  };
}

// -- getLastHumanActivityDate -----------------------------------------------

describe("getLastHumanActivityDate", () => {
  test("returns null for empty timeline", () => {
    expect(getLastHumanActivityDate([], "alice")).toBeNull();
  });

  test("picks up human comment by the correct assignee", () => {
    const ts = daysAgo(5);
    const timeline = [
      { event: "commented", actor: { login: "alice", type: "User" }, created_at: ts },
    ];
    const result = getLastHumanActivityDate(timeline, "alice");
    expect(result).not.toBeNull();
    expect(result.toISOString()).toBe(new Date(ts).toISOString());
  });

  test("ignores comments by a different user", () => {
    const timeline = [
      { event: "commented", actor: { login: "bob", type: "User" }, created_at: daysAgo(3) },
    ];
    expect(getLastHumanActivityDate(timeline, "alice")).toBeNull();
  });

  test("ignores bot comments", () => {
    const timeline = [
      { event: "commented", actor: { login: "github-actions[bot]", type: "Bot" }, created_at: daysAgo(1) },
    ];
    expect(getLastHumanActivityDate(timeline, "github-actions[bot]")).toBeNull();
  });

  test("picks cross-referenced PR authored by the assignee", () => {
    const ts = daysAgo(7);
    const timeline = [
      {
        event: "cross-referenced",
        actor: { login: "alice", type: "User" },
        created_at: ts,
        source: {
          issue: {
            user: { login: "alice" },
            pull_request: { url: "https://api.github.com/repos/x/y/pulls/1" },
          },
        },
      },
    ];
    expect(getLastHumanActivityDate(timeline, "alice")).not.toBeNull();
  });

  test("ignores cross-referenced PR by a different author", () => {
    const timeline = [
      {
        event: "cross-referenced",
        actor: { login: "carol", type: "User" },
        created_at: daysAgo(2),
        source: {
          issue: {
            user: { login: "carol" },
            pull_request: { url: "https://api.github.com/repos/x/y/pulls/2" },
          },
        },
      },
    ];
    expect(getLastHumanActivityDate(timeline, "alice")).toBeNull();
  });

  test("returns the most recent event when multiple exist", () => {
    const older = daysAgo(20);
    const newer = daysAgo(5);
    const timeline = [
      { event: "commented", actor: { login: "alice", type: "User" }, created_at: older },
      { event: "commented", actor: { login: "alice", type: "User" }, created_at: newer },
    ];
    const result = getLastHumanActivityDate(timeline, "alice");
    expect(result.toISOString()).toBe(new Date(newer).toISOString());
  });

  test("picks up reassignment by maintainer", () => {
    const ts = daysAgo(2);
    const timeline = [
      { event: "assigned", actor: { login: "maintainer", type: "User" }, assignee: { login: "alice" }, created_at: ts },
    ];
    const result = getLastHumanActivityDate(timeline, "alice");
    expect(result.toISOString()).toBe(new Date(ts).toISOString());
  });

  test("picks up warning label removal by maintainer", () => {
    const ts = daysAgo(1);
    const timeline = [
      { event: "unlabeled", actor: { login: "maintainer", type: "User" }, label: { name: WARN_LABEL }, created_at: ts },
    ];
    const result = getLastHumanActivityDate(timeline, "alice");
    expect(result.toISOString()).toBe(new Date(ts).toISOString());
  });
});

// -- Comment builders -------------------------------------------------------

describe("buildWarningComment", () => {
  test("contains marker, assignee mention, and day count", () => {
    const body = buildWarningComment("alice", 15, 30);
    expect(body).toContain(WARNING_MARKER);
    expect(body).toContain("@alice");
    expect(body).toContain("15 day");
  });
});

describe("buildUnassignComment", () => {
  test("contains marker, assignee mention, and day count", () => {
    const body = buildUnassignComment("alice", 30);
    expect(body).toContain(WARNING_MARKER);
    expect(body).toContain("@alice");
    expect(body).toContain("30");
  });
});

// -- warningCommentExists ---------------------------------------------------

describe("warningCommentExists", () => {
  test("true when matching comment exists and is newer than last activity", async () => {
    const octokit = makeOctokit();
    octokit.paginate.mockResolvedValue([
      { body: `${WARNING_MARKER}\n@alice some warning text`, created_at: daysAgo(5) },
    ]);
    const lastActivity = new Date(daysAgo(10));
    expect(await warningCommentExists(octokit, "o", "r", 42, "alice", lastActivity)).toBe(true);
  });

  test("false when matching comment exists but is older than last activity", async () => {
    const octokit = makeOctokit();
    octokit.paginate.mockResolvedValue([
      { body: `${WARNING_MARKER}\n@alice some warning text`, created_at: daysAgo(15) },
    ]);
    const lastActivity = new Date(daysAgo(10));
    expect(await warningCommentExists(octokit, "o", "r", 42, "alice", lastActivity)).toBe(false);
  });

  test("false when no matching comment", async () => {
    const octokit = makeOctokit();
    octokit.paginate.mockResolvedValue([{ body: "unrelated comment" }]);
    expect(await warningCommentExists(octokit, "o", "r", 42, "alice")).toBe(false);
  });

  test("false when comment is for a different assignee", async () => {
    const octokit = makeOctokit();
    octokit.paginate.mockResolvedValue([
      { body: `${WARNING_MARKER}\n@bob some warning text` },
    ]);
    expect(await warningCommentExists(octokit, "o", "r", 42, "alice")).toBe(false);
  });

  test("false on API error (safe default)", async () => {
    const octokit = makeOctokit();
    octokit.paginate.mockRejectedValue(new Error("API error"));
    expect(await warningCommentExists(octokit, "o", "r", 42, "alice")).toBe(false);
  });
});

// -- findLinkedPRActivity ---------------------------------------------------

describe("findLinkedPRActivity", () => {
  test("hasActivity=true when PR body mentions the issue", async () => {
    const octokit = makeOctokit();
    octokit.rest.search.issuesAndPullRequests.mockResolvedValue({
      data: {
        items: [{
          pull_request: { url: "https://..." },
          body: "closes #42 — implements the feature",
          updated_at: daysAgo(3),
        }],
      },
    });
    const result = await findLinkedPRActivity(octokit, "o", "r", 42, "alice");
    expect(result.hasActivity).toBe(true);
    expect(result.date).not.toBeNull();
  });

  test("hasActivity=false with no matching PRs", async () => {
    const octokit = makeOctokit();
    octokit.rest.search.issuesAndPullRequests.mockResolvedValue({ data: { items: [] } });
    const result = await findLinkedPRActivity(octokit, "o", "r", 42, "alice");
    expect(result.hasActivity).toBe(false);
  });

  test("hasActivity=false when body does not mention issue number", async () => {
    const octokit = makeOctokit();
    octokit.rest.search.issuesAndPullRequests.mockResolvedValue({
      data: {
        items: [{
          pull_request: { url: "https://..." },
          body: "general improvement",
          updated_at: daysAgo(2),
        }],
      },
    });
    const result = await findLinkedPRActivity(octokit, "o", "r", 42, "alice");
    expect(result.hasActivity).toBe(false);
  });

  test("hasActivity=false when body mentions a longer issue number (#420)", async () => {
    const octokit = makeOctokit();
    octokit.rest.search.issuesAndPullRequests.mockResolvedValue({
      data: {
        items: [{
          pull_request: { url: "https://..." },
          body: "fixes #420 — different issue entirely",
          updated_at: daysAgo(2),
        }],
      },
    });
    const result = await findLinkedPRActivity(octokit, "o", "r", 42, "alice");
    expect(result.hasActivity).toBe(false);
  });

  test("fail-closed on API error", async () => {
    const octokit = makeOctokit();
    octokit.rest.search.issuesAndPullRequests.mockRejectedValue(new Error("rate limited"));
    const result = await findLinkedPRActivity(octokit, "o", "r", 42, "alice");
    expect(result.hasActivity).toBe(true);
    expect(result.error).toBe(true);
  });
});

// -- processAssignee (state machine) ----------------------------------------

describe("processAssignee", () => {
  const baseConfig = { warnAfterDays: 15, unassignAfterDays: 30, dryRun: false };

  function buildOctokitWithTimeline(timelineEvents, comments = []) {
    const octokit = makeOctokit();
    octokit.paginate.mockImplementation(async (fn) => {
      if (fn === octokit.rest.issues.listEventsForTimeline) return timelineEvents;
      if (fn === octokit.rest.issues.listComments) return comments;
      return [];
    });
    return octokit;
  }

  test("skips active assignee", async () => {
    const timeline = [
      { event: "commented", actor: { login: "alice", type: "User" }, created_at: daysAgo(3) },
    ];
    const octokit = buildOctokitWithTimeline(timeline);
    const action = await processAssignee(octokit, "o", "r", makeIssue(), "alice", baseConfig);
    expect(action).toBe("skipped");
    expect(octokit.rest.issues.createComment).not.toHaveBeenCalled();
    expect(octokit.rest.issues.removeAssignees).not.toHaveBeenCalled();
  });

  test("warns assignee inactive 15–29 days", async () => {
    const octokit = buildOctokitWithTimeline([]);
    octokit.rest.search.issuesAndPullRequests.mockResolvedValue({ data: { items: [] } });
    const issue = makeIssue({ created_at: daysAgo(20) });
    const action = await processAssignee(octokit, "o", "r", issue, "alice", baseConfig);
    expect(action).toBe("warned");
    expect(octokit.rest.issues.createComment).toHaveBeenCalledTimes(1);
    const body = octokit.rest.issues.createComment.mock.calls[0][0].body;
    expect(body).toContain(WARNING_MARKER);
    expect(body).toContain("@alice");
    expect(octokit.rest.issues.addLabels).toHaveBeenCalledWith(
      expect.objectContaining({ labels: [WARN_LABEL] })
    );
    expect(octokit.rest.issues.removeAssignees).not.toHaveBeenCalled();
  });

  test("no double-warn when label already present", async () => {
    const octokit = buildOctokitWithTimeline([]);
    octokit.rest.search.issuesAndPullRequests.mockResolvedValue({ data: { items: [] } });
    octokit.paginate.mockImplementation(async (fn) => {
      if (fn === octokit.rest.issues.listComments) {
        return [{ body: `${WARNING_MARKER}\n@alice old warning` }];
      }
      return [];
    });
    const issue = makeIssue({ labels: [{ name: WARN_LABEL }], created_at: daysAgo(20) });
    const action = await processAssignee(octokit, "o", "r", issue, "alice", baseConfig);
    expect(action).toBe("skipped");
    expect(octokit.rest.issues.createComment).not.toHaveBeenCalled();
  });

  test("unassigns after 30 days inactivity", async () => {
    const octokit = buildOctokitWithTimeline([]);
    octokit.rest.search.issuesAndPullRequests.mockResolvedValue({ data: { items: [] } });
    const issue = makeIssue({ labels: [{ name: WARN_LABEL }], created_at: daysAgo(35) });
    const action = await processAssignee(octokit, "o", "r", issue, "alice", baseConfig);
    expect(action).toBe("unassigned");
    expect(octokit.rest.issues.removeAssignees).toHaveBeenCalledWith(
      expect.objectContaining({ assignees: ["alice"] })
    );
    expect(octokit.rest.issues.createComment).toHaveBeenCalled();
  });

  test("clears warning when activity resumes", async () => {
    const timeline = [
      { event: "commented", actor: { login: "alice", type: "User" }, created_at: daysAgo(2) },
    ];
    const octokit = buildOctokitWithTimeline(timeline);
    const issue = makeIssue({ labels: [{ name: WARN_LABEL }], created_at: daysAgo(25) });
    const action = await processAssignee(octokit, "o", "r", issue, "alice", baseConfig);
    expect(action).toBe("warning-cleared");
    expect(octokit.rest.issues.removeLabel).toHaveBeenCalledWith(
      expect.objectContaining({ name: WARN_LABEL })
    );
    expect(octokit.rest.issues.removeAssignees).not.toHaveBeenCalled();
  });

  test("fail-closed: timeline fetch error → skip removal", async () => {
    const octokit = makeOctokit();
    octokit.paginate.mockImplementation(async (fn) => {
      if (fn === octokit.rest.issues.listEventsForTimeline) throw new Error("network error");
      return [];
    });
    const action = await processAssignee(
      octokit, "o", "r", makeIssue({ created_at: daysAgo(40) }), "alice", baseConfig
    );
    expect(action).toBe("error");
    expect(octokit.rest.issues.removeAssignees).not.toHaveBeenCalled();
  });

  test("fail-closed: PR search error → skip removal", async () => {
    const octokit = buildOctokitWithTimeline([]);
    octokit.rest.search.issuesAndPullRequests.mockRejectedValue(new Error("rate limited"));
    const action = await processAssignee(
      octokit, "o", "r", makeIssue({ created_at: daysAgo(40) }), "alice", baseConfig
    );
    expect(action).toBe("error");
    expect(octokit.rest.issues.removeAssignees).not.toHaveBeenCalled();
  });

  test("per-assignee: active alice does not shield inactive bob", async () => {
    const timeline = [
      { event: "commented", actor: { login: "alice", type: "User" }, created_at: daysAgo(2) },
    ];
    const octokit = buildOctokitWithTimeline(timeline);
    octokit.rest.search.issuesAndPullRequests.mockResolvedValue({ data: { items: [] } });

    const issue = makeIssue({
      labels: [{ name: WARN_LABEL }],
      created_at: daysAgo(40),
      assignees: [{ login: "alice" }, { login: "bob" }],
    });

    const aliceAction = await processAssignee(octokit, "o", "r", issue, "alice", baseConfig);
    expect(aliceAction).toBe("warning-cleared");

    // Reset mocks for bob: no activity.
    octokit.paginate.mockImplementation(async (fn) => {
      if (fn === octokit.rest.issues.listEventsForTimeline) return [];
      if (fn === octokit.rest.issues.listComments) return [];
      return [];
    });
    octokit.rest.search.issuesAndPullRequests.mockResolvedValue({ data: { items: [] } });

    const bobAction = await processAssignee(octokit, "o", "r", issue, "bob", baseConfig);
    expect(bobAction).toBe("unassigned");
    expect(octokit.rest.issues.removeAssignees).toHaveBeenCalledWith(
      expect.objectContaining({ assignees: ["bob"] })
    );
  });

  test("linked PR by assignee counts as active", async () => {
    const octokit = buildOctokitWithTimeline([]);
    octokit.rest.search.issuesAndPullRequests.mockResolvedValue({
      data: {
        items: [{
          pull_request: { url: "https://..." },
          body: "fixes #42 adding the feature",
          updated_at: daysAgo(3),
        }],
      },
    });
    const action = await processAssignee(
      octokit, "o", "r", makeIssue({ created_at: daysAgo(20) }), "alice", baseConfig
    );
    expect(action).toBe("skipped");
  });

  test("dry-run does not mutate state", async () => {
    const octokit = buildOctokitWithTimeline([]);
    octokit.rest.search.issuesAndPullRequests.mockResolvedValue({ data: { items: [] } });
    const action = await processAssignee(
      octokit, "o", "r", makeIssue({ created_at: daysAgo(35) }), "alice",
      { ...baseConfig, dryRun: true }
    );
    expect(action).toBe("unassigned");
    expect(octokit.rest.issues.removeAssignees).not.toHaveBeenCalled();
    expect(octokit.rest.issues.createComment).not.toHaveBeenCalled();
  });
});

// -- ensureLabel ------------------------------------------------------------

describe("ensureLabel", () => {
  test("no-op when label exists", async () => {
    const octokit = makeOctokit();
    await ensureLabel(octokit, "o", "r", WARN_LABEL);
    expect(octokit.rest.issues.createLabel).not.toHaveBeenCalled();
  });

  test("creates label on 404", async () => {
    const octokit = makeOctokit();
    const err = new Error("Not Found");
    err.status = 404;
    octokit.rest.issues.getLabel.mockRejectedValue(err);
    await ensureLabel(octokit, "o", "r", WARN_LABEL, "e4e669", "desc");
    expect(octokit.rest.issues.createLabel).toHaveBeenCalledWith(
      expect.objectContaining({ name: WARN_LABEL, color: "e4e669" })
    );
  });

  test("logs and continues on non-404 error", async () => {
    const octokit = makeOctokit();
    const err = new Error("Server Error");
    err.status = 500;
    octokit.rest.issues.getLabel.mockRejectedValue(err);
    const spy = jest.spyOn(console, "warn").mockImplementation(() => {});
    await ensureLabel(octokit, "o", "r", WARN_LABEL);
    expect(octokit.rest.issues.createLabel).not.toHaveBeenCalled();
    spy.mockRestore();
  });
});

// -- run (integration) ------------------------------------------------------

describe("run", () => {
  const ctx = { repo: { owner: "vllm-project", repo: "semantic-router" } };

  test("no-op when no assigned issues", async () => {
    const octokit = makeOctokit();
    octokit.paginate.mockImplementation(async (fn) => {
      if (fn === octokit.rest.issues.listForRepo) return [];
      return [];
    });
    await run(octokit, ctx, { warnAfterDays: 15, unassignAfterDays: 30 });
    expect(octokit.rest.issues.createComment).not.toHaveBeenCalled();
    expect(octokit.rest.issues.removeAssignees).not.toHaveBeenCalled();
  });

  test("skips pull requests returned by listForRepo", async () => {
    const octokit = makeOctokit();
    octokit.paginate.mockImplementation(async (fn) => {
      if (fn === octokit.rest.issues.listForRepo) {
        return [{
          number: 99, title: "A PR",
          pull_request: { url: "https://..." },
          assignees: [{ login: "alice" }],
          labels: [{ name: "accepted" }],
          created_at: daysAgo(40),
        }];
      }
      return [];
    });
    await run(octokit, ctx, { warnAfterDays: 15, unassignAfterDays: 30 });
    expect(octokit.rest.issues.removeAssignees).not.toHaveBeenCalled();
  });

  test("dry-run reports actions without mutations", async () => {
    const octokit = makeOctokit();
    octokit.paginate.mockImplementation(async (fn) => {
      if (fn === octokit.rest.issues.listForRepo) {
        return [makeIssue({ created_at: daysAgo(40), assignees: [{ login: "alice" }] })];
      }
      if (fn === octokit.rest.issues.listEventsForTimeline) return [];
      if (fn === octokit.rest.issues.listComments) return [];
      return [];
    });
    octokit.rest.search.issuesAndPullRequests.mockResolvedValue({ data: { items: [] } });
    await run(octokit, ctx, { warnAfterDays: 15, unassignAfterDays: 30, dryRun: true });
    expect(octokit.rest.issues.removeAssignees).not.toHaveBeenCalled();
    expect(octokit.rest.issues.createComment).not.toHaveBeenCalled();
  });
});
