// SPDX-License-Identifier: Apache-2.0
// Tests for unassign-inactive.js. Every GitHub API call is faked; no network.

"use strict";

const fs = require("fs");
const os = require("os");
const path = require("path");

const {
  run,
  evaluateAssignee,
  collectIssueSignals,
  collectPullRequestActivity,
  readTransitionState,
  buildWarningComment,
  buildRemovalComment,
  warningMarker,
  removalMarker,
  MAX_LINKED_PULL_REQUESTS,
} = require("../unassign-inactive");

// -- Fixtures ---------------------------------------------------------------

const NOW = new Date("2026-06-01T00:00:00.000Z");
const DAY_MS = 24 * 60 * 60 * 1000;

function daysAgo(n) {
  return new Date(NOW.getTime() - n * DAY_MS).toISOString();
}

const CONFIG = {
  warnAfterDays: 15,
  unassignAfterDays: 30,
  dryRun: false,
  now: NOW,
};

function issueFixture(overrides = {}) {
  return {
    number: 42,
    title: "Test issue",
    created_at: daysAgo(90),
    labels: [{ name: "accepted" }],
    assignees: [{ login: "alice" }],
    ...overrides,
  };
}

const WORKFLOW_BOT = { login: "github-actions[bot]", type: "Bot" };

function comment(login, kind, at, user = WORKFLOW_BOT) {
  const marker = kind === "warned" ? warningMarker(login) : removalMarker(login);
  return {
    id: `${kind}-${login}-${at}`,
    body: `${marker}\nbody text`,
    created_at: at,
    user,
  };
}

function timelineComment(login, at) {
  return { event: "commented", actor: { login, type: "User" }, created_at: at };
}

/**
 * Fake Octokit. `timelines` and `comments` are keyed by issue/PR number, so a
 * pull-request timeline is distinguishable from its issue's timeline.
 */
function makeGitHub({
  issues = [],
  timelines = {},
  comments = {},
  failListForRepo = false,
  failTimelineFor = [],
  failCommentsFor = [],
  failCreateComment = false,
  failRemoveAssignees = false,
} = {}) {
  const rest = {
    issues: {
      listForRepo: Symbol("listForRepo"),
      listEventsForTimeline: Symbol("listEventsForTimeline"),
      listComments: Symbol("listComments"),
      createComment: jest.fn(async () => {
        if (failCreateComment) throw new Error("comment write failed");
        return {};
      }),
      removeAssignees: jest.fn(async () => {
        if (failRemoveAssignees) throw new Error("remove failed");
        return {};
      }),
    },
  };

  const paginate = jest.fn(async (endpoint, params) => {
    if (endpoint === rest.issues.listForRepo) {
      if (failListForRepo) throw new Error("listing failed");
      return issues;
    }
    if (endpoint === rest.issues.listEventsForTimeline) {
      if (failTimelineFor.includes(params.issue_number)) {
        throw new Error("timeline failed");
      }
      return timelines[params.issue_number] || [];
    }
    if (endpoint === rest.issues.listComments) {
      if (failCommentsFor.includes(params.issue_number)) {
        throw new Error("comments failed");
      }
      return comments[params.issue_number] || [];
    }
    throw new Error("unexpected endpoint");
  });

  return { paginate, rest };
}

const CONTEXT = { repo: { owner: "o", repo: "r" } };

function crossReference({
  number,
  author,
  createdAt,
  referencedAt,
  actor = author,
  repositoryFullName = "o/r",
}) {
  return {
    event: "cross-referenced",
    actor: { login: actor, type: "User" },
    created_at: referencedAt,
    source: {
      type: "issue",
      issue: {
        number,
        created_at: createdAt,
        user: { login: author },
        pull_request: { url: `https://api.github.com/pulls/${number}` },
        repository: { full_name: repositoryFullName },
      },
    },
  };
}

// -- collectIssueSignals ----------------------------------------------------

describe("collectIssueSignals", () => {
  test("returns nothing for an empty timeline", () => {
    const signals = collectIssueSignals([], "alice");
    expect(signals.lastActivityAt).toBeNull();
    expect(signals.linkedPullRequests).toEqual([]);
  });

  test("credits the assignee's own comment", () => {
    const at = daysAgo(5);
    const signals = collectIssueSignals([timelineComment("alice", at)], "alice");
    expect(signals.lastActivityAt.toISOString()).toBe(new Date(at).toISOString());
  });

  test("ignores another contributor's comment", () => {
    const signals = collectIssueSignals([timelineComment("bob", daysAgo(1))], "alice");
    expect(signals.lastActivityAt).toBeNull();
  });

  test("matches logins case-insensitively", () => {
    const signals = collectIssueSignals([timelineComment("Alice", daysAgo(4))], "alice");
    expect(signals.lastActivityAt).not.toBeNull();
  });

  test("ignores bot writes even when the login matches", () => {
    const timeline = [
      {
        event: "commented",
        actor: { login: "github-actions[bot]", type: "Bot" },
        created_at: daysAgo(1),
      },
    ];
    expect(collectIssueSignals(timeline, "github-actions[bot]").lastActivityAt).toBeNull();
  });

  test("ignores label churn by the assignee", () => {
    const timeline = [
      {
        event: "labeled",
        actor: { login: "alice", type: "User" },
        label: { name: "in-progress" },
        created_at: daysAgo(1),
      },
      {
        event: "milestoned",
        actor: { login: "alice", type: "User" },
        created_at: daysAgo(1),
      },
    ];
    expect(collectIssueSignals(timeline, "alice").lastActivityAt).toBeNull();
  });

  test("records assignment, including assignment performed by automation", () => {
    const at = daysAgo(3);
    const timeline = [
      {
        event: "assigned",
        actor: { login: "triage-bot[bot]", type: "Bot" },
        assignee: { login: "alice" },
        created_at: at,
      },
    ];
    const signals = collectIssueSignals(timeline, "alice");
    expect(signals.lastActivityAt.toISOString()).toBe(new Date(at).toISOString());
  });

  test("does not credit an assignment of somebody else", () => {
    const timeline = [
      {
        event: "assigned",
        actor: { login: "maintainer", type: "User" },
        assignee: { login: "bob" },
        created_at: daysAgo(1),
      },
    ];
    expect(collectIssueSignals(timeline, "alice").lastActivityAt).toBeNull();
  });

  test("keeps the most recent of several qualifying events", () => {
    const timeline = [
      timelineComment("alice", daysAgo(20)),
      timelineComment("alice", daysAgo(4)),
      timelineComment("alice", daysAgo(11)),
    ];
    const signals = collectIssueSignals(timeline, "alice");
    expect(signals.lastActivityAt.toISOString()).toBe(new Date(daysAgo(4)).toISOString());
  });

  test("collects linked pull requests regardless of who referenced them", () => {
    const timeline = [
      crossReference({
        number: 100,
        author: "alice",
        createdAt: daysAgo(9),
        referencedAt: daysAgo(9),
        actor: "carol",
      }),
    ];
    const signals = collectIssueSignals(timeline, "alice");
    expect(signals.linkedPullRequests).toHaveLength(1);
    expect(signals.linkedPullRequests[0]).toMatchObject({
      number: 100,
      authorLogin: "alice",
    });
    // Carol's reference is not Alice's activity on its own.
    expect(signals.lastActivityAt).toBeNull();
  });

  test("ignores a cross-reference whose source is an issue, not a pull request", () => {
    const timeline = [
      {
        event: "cross-referenced",
        actor: { login: "alice", type: "User" },
        created_at: daysAgo(2),
        source: { type: "issue", issue: { number: 7, user: { login: "alice" } } },
      },
    ];
    // Alice made the reference, so it counts as her activity, but no pull
    // request is queued for inspection.
    const signals = collectIssueSignals(timeline, "alice");
    expect(signals.linkedPullRequests).toEqual([]);
    expect(signals.lastActivityAt).not.toBeNull();
  });
});

// -- collectPullRequestActivity ---------------------------------------------

describe("collectPullRequestActivity", () => {
  const pullRequest = {
    number: 100,
    authorLogin: "alice",
    createdAt: new Date(daysAgo(40)),
    repositoryFullName: "o/r",
  };

  test("fails closed when the pull-request timeline cannot be read", async () => {
    const github = makeGitHub({ failTimelineFor: [100] });
    const result = await collectPullRequestActivity(github, "o", "r", pullRequest, "alice");
    expect(result.error).toBe(true);
  });

  test("credits opening the pull request to its author", async () => {
    const github = makeGitHub({ timelines: { 100: [] } });
    const result = await collectPullRequestActivity(github, "o", "r", pullRequest, "alice");
    expect(result.lastActivityAt.toISOString()).toBe(pullRequest.createdAt.toISOString());
  });

  test("ignores traffic from other people on the assignee's pull request", async () => {
    const github = makeGitHub({
      timelines: {
        100: [
          timelineComment("bob", daysAgo(1)),
          {
            event: "reviewed",
            actor: { login: "carol", type: "User" },
            submitted_at: daysAgo(2),
          },
          {
            event: "labeled",
            actor: { login: "ci[bot]", type: "Bot" },
            created_at: daysAgo(1),
          },
        ],
      },
    });
    const result = await collectPullRequestActivity(github, "o", "r", pullRequest, "alice");
    // Only the pull request's own creation date survives.
    expect(result.lastActivityAt.toISOString()).toBe(pullRequest.createdAt.toISOString());
  });

  test("credits the assignee's commits on their own pull request", async () => {
    const at = daysAgo(3);
    const github = makeGitHub({
      timelines: {
        100: [{ event: "committed", author: { name: "Alice A", date: at } }],
      },
    });
    const result = await collectPullRequestActivity(github, "o", "r", pullRequest, "alice");
    expect(result.lastActivityAt.toISOString()).toBe(new Date(at).toISOString());
  });

  test("does not credit commits on somebody else's pull request", async () => {
    const foreign = { ...pullRequest, authorLogin: "bob" };
    const github = makeGitHub({
      timelines: {
        100: [{ event: "committed", author: { name: "Bob B", date: daysAgo(1) } }],
      },
    });
    const result = await collectPullRequestActivity(github, "o", "r", foreign, "alice");
    expect(result.lastActivityAt).toBeNull();
  });

  test("credits the assignee's review on somebody else's pull request", async () => {
    const foreign = { ...pullRequest, authorLogin: "bob" };
    const at = daysAgo(2);
    const github = makeGitHub({
      timelines: {
        100: [{ event: "reviewed", actor: { login: "alice", type: "User" }, submitted_at: at }],
      },
    });
    const result = await collectPullRequestActivity(github, "o", "r", foreign, "alice");
    expect(result.lastActivityAt.toISOString()).toBe(new Date(at).toISOString());
  });
});

// -- readTransitionState ----------------------------------------------------

describe("readTransitionState", () => {
  test("finds a current warning", () => {
    const state = readTransitionState(
      [comment("alice", "warned", daysAgo(5))],
      "alice",
      new Date(daysAgo(20))
    );
    expect(state.warnedAt).not.toBeNull();
    expect(state.supersededWarningAt).toBeNull();
  });

  test("treats a warning older than the last activity as superseded", () => {
    const state = readTransitionState(
      [comment("alice", "warned", daysAgo(25))],
      "alice",
      new Date(daysAgo(10))
    );
    expect(state.warnedAt).toBeNull();
    expect(state.supersededWarningAt).not.toBeNull();
  });

  test("does not read another assignee's warning", () => {
    const state = readTransitionState(
      [comment("bob", "warned", daysAgo(5))],
      "alice",
      new Date(daysAgo(20))
    );
    expect(state.warnedAt).toBeNull();
  });

  test("does not confuse a removal notice with a warning", () => {
    const state = readTransitionState(
      [comment("alice", "removed", daysAgo(5))],
      "alice",
      new Date(daysAgo(20))
    );
    expect(state.warnedAt).toBeNull();
    expect(state.removalNoticeAt).not.toBeNull();
  });

  test("ignores a login that is a prefix of the assignee's", () => {
    const state = readTransitionState(
      [comment("alice-b", "warned", daysAgo(5))],
      "alice",
      new Date(daysAgo(20))
    );
    expect(state.warnedAt).toBeNull();
  });

  test("matches a marker written for a differently-cased login", () => {
    const state = readTransitionState(
      [comment("Alice", "warned", daysAgo(5))],
      "alice",
      new Date(daysAgo(20))
    );
    expect(state.warnedAt).not.toBeNull();
  });

  test("ignores a marker quoted by a human, which the workflow did not write", () => {
    const quoted = comment("alice", "warned", daysAgo(2), { login: "alice", type: "User" });
    const state = readTransitionState(
      [comment("alice", "warned", daysAgo(20)), quoted],
      "alice",
      new Date(daysAgo(40))
    );
    expect(state.warnedAt.toISOString()).toBe(new Date(daysAgo(20)).toISOString());
  });
});

// -- Comment bodies ---------------------------------------------------------

describe("comment bodies", () => {
  test("the warning carries its marker, the mention, and both windows", () => {
    const body = buildWarningComment("alice", 20, 10);
    expect(body).toContain(warningMarker("alice"));
    expect(body).toContain("@alice");
    expect(body).toContain("20 days"); // inactivity so far
    expect(body).toContain("10 days"); // grace window that remains
    expect(body).not.toContain(removalMarker("alice"));
  });

  test("the removal notice carries a different marker and cites the warning", () => {
    const body = buildRemovalComment("alice", 34, new Date("2026-05-10T00:00:00Z"));
    expect(body).toContain(removalMarker("alice"));
    expect(body).not.toContain(warningMarker("alice"));
    expect(body).toContain("2026-05-10");
  });
});

// -- evaluateAssignee: the state machine ------------------------------------

describe("evaluateAssignee", () => {
  test("leaves a recently active assignee alone", async () => {
    const github = makeGitHub({ timelines: { 42: [timelineComment("alice", daysAgo(3))] } });
    const record = await evaluateAssignee(github, "o", "r", issueFixture(), "alice", CONFIG);
    expect(record.action).toBe("skipped");
    expect(github.rest.issues.createComment).not.toHaveBeenCalled();
    expect(github.rest.issues.removeAssignees).not.toHaveBeenCalled();
  });

  test("warns an assignee past the warning threshold", async () => {
    const github = makeGitHub({ timelines: { 42: [] } });
    const record = await evaluateAssignee(github, "o", "r", issueFixture({ created_at: daysAgo(20) }), "alice", CONFIG);
    expect(record.action).toBe("warned");
    expect(github.rest.issues.createComment).toHaveBeenCalledTimes(1);
    expect(github.rest.issues.createComment.mock.calls[0][0].body).toContain(
      warningMarker("alice")
    );
    expect(github.rest.issues.removeAssignees).not.toHaveBeenCalled();
  });

  test("warns rather than removes an assignee first seen past the removal threshold", async () => {
    const github = makeGitHub({ timelines: { 42: [] } });
    const record = await evaluateAssignee(github, "o", "r", issueFixture({ created_at: daysAgo(400) }), "alice", CONFIG);
    expect(record.action).toBe("warned");
    expect(github.rest.issues.removeAssignees).not.toHaveBeenCalled();
  });

  test("does not remove until the full grace window has passed since the warning", async () => {
    const github = makeGitHub({
      timelines: { 42: [] },
      comments: { 42: [comment("alice", "warned", daysAgo(1))] },
    });
    const record = await evaluateAssignee(github, "o", "r", issueFixture({ created_at: daysAgo(400) }), "alice", CONFIG);
    expect(record.action).toBe("skipped");
    expect(record.reason).toContain("grace left");
    expect(github.rest.issues.removeAssignees).not.toHaveBeenCalled();
  });

  test("does not warn twice while a warning stands", async () => {
    const github = makeGitHub({
      timelines: { 42: [] },
      comments: { 42: [comment("alice", "warned", daysAgo(2))] },
    });
    const record = await evaluateAssignee(github, "o", "r", issueFixture({ created_at: daysAgo(20) }), "alice", CONFIG);
    expect(record.action).toBe("skipped");
    expect(github.rest.issues.createComment).not.toHaveBeenCalled();
  });

  test("removes once inactive past the threshold and warned beyond the grace window", async () => {
    const github = makeGitHub({
      timelines: { 42: [] },
      comments: { 42: [comment("alice", "warned", daysAgo(16))] },
    });
    const record = await evaluateAssignee(github, "o", "r", issueFixture({ created_at: daysAgo(40) }), "alice", CONFIG);
    expect(record.action).toBe("unassigned");
    expect(github.rest.issues.removeAssignees).toHaveBeenCalledWith(
      expect.objectContaining({ assignees: ["alice"] })
    );
  });

  test("notifies before removing", async () => {
    const order = [];
    const github = makeGitHub({
      timelines: { 42: [] },
      comments: { 42: [comment("alice", "warned", daysAgo(16))] },
    });
    github.rest.issues.createComment.mockImplementation(async () => {
      order.push("comment");
      return {};
    });
    github.rest.issues.removeAssignees.mockImplementation(async () => {
      order.push("remove");
      return {};
    });
    await evaluateAssignee(github, "o", "r", issueFixture({ created_at: daysAgo(40) }), "alice", CONFIG);
    expect(order).toEqual(["comment", "remove"]);
  });

  test("keeps the assignment when the removal notice cannot be posted", async () => {
    const github = makeGitHub({
      timelines: { 42: [] },
      comments: { 42: [comment("alice", "warned", daysAgo(16))] },
      failCreateComment: true,
    });
    const record = await evaluateAssignee(github, "o", "r", issueFixture({ created_at: daysAgo(40) }), "alice", CONFIG);
    expect(record.action).toBe("error");
    expect(github.rest.issues.removeAssignees).not.toHaveBeenCalled();
  });

  test("retries only the removal when the notice already landed", async () => {
    const github = makeGitHub({
      timelines: { 42: [] },
      comments: {
        42: [comment("alice", "warned", daysAgo(16)), comment("alice", "removed", daysAgo(1))],
      },
    });
    const record = await evaluateAssignee(github, "o", "r", issueFixture({ created_at: daysAgo(40) }), "alice", CONFIG);
    expect(record.action).toBe("unassigned");
    expect(github.rest.issues.createComment).not.toHaveBeenCalled();
    expect(github.rest.issues.removeAssignees).toHaveBeenCalledTimes(1);
  });

  test("reports an error when the removal call itself fails", async () => {
    const github = makeGitHub({
      timelines: { 42: [] },
      comments: { 42: [comment("alice", "warned", daysAgo(16))] },
      failRemoveAssignees: true,
    });
    const record = await evaluateAssignee(github, "o", "r", issueFixture({ created_at: daysAgo(40) }), "alice", CONFIG);
    expect(record.action).toBe("error");
  });

  test("cancels a pending warning when activity resumes", async () => {
    const github = makeGitHub({
      timelines: { 42: [timelineComment("alice", daysAgo(2))] },
      comments: { 42: [comment("alice", "warned", daysAgo(10))] },
    });
    const record = await evaluateAssignee(github, "o", "r", issueFixture({ created_at: daysAgo(40) }), "alice", CONFIG);
    expect(record.action).toBe("recovered");
    expect(github.rest.issues.removeAssignees).not.toHaveBeenCalled();
  });

  test("warns again after a cancelled cycle lapses a second time", async () => {
    const github = makeGitHub({
      timelines: { 42: [timelineComment("alice", daysAgo(20))] },
      comments: { 42: [comment("alice", "warned", daysAgo(30))] },
    });
    const record = await evaluateAssignee(github, "o", "r", issueFixture({ created_at: daysAgo(90) }), "alice", CONFIG);
    expect(record.action).toBe("warned");
    expect(github.rest.issues.createComment).toHaveBeenCalledTimes(1);
  });

  test("fails closed when the issue timeline cannot be read", async () => {
    const github = makeGitHub({ failTimelineFor: [42] });
    const record = await evaluateAssignee(github, "o", "r", issueFixture({ created_at: daysAgo(400) }), "alice", CONFIG);
    expect(record.action).toBe("error");
    expect(github.rest.issues.removeAssignees).not.toHaveBeenCalled();
    expect(github.rest.issues.createComment).not.toHaveBeenCalled();
  });

  test("fails closed when a linked pull request cannot be read", async () => {
    const github = makeGitHub({
      timelines: {
        42: [
          crossReference({
            number: 100,
            author: "alice",
            createdAt: daysAgo(50),
            referencedAt: daysAgo(50),
          }),
        ],
      },
      failTimelineFor: [100],
    });
    const record = await evaluateAssignee(github, "o", "r", issueFixture({ created_at: daysAgo(400) }), "alice", CONFIG);
    expect(record.action).toBe("error");
    expect(github.rest.issues.removeAssignees).not.toHaveBeenCalled();
  });

  test("fails closed when the warning lookup fails, rather than warning again", async () => {
    const github = makeGitHub({ timelines: { 42: [] }, failCommentsFor: [42] });
    const record = await evaluateAssignee(github, "o", "r", issueFixture({ created_at: daysAgo(400) }), "alice", CONFIG);
    expect(record.action).toBe("error");
    expect(github.rest.issues.createComment).not.toHaveBeenCalled();
    expect(github.rest.issues.removeAssignees).not.toHaveBeenCalled();
  });

  test("recent work on a linked pull request keeps the assignment", async () => {
    const github = makeGitHub({
      timelines: {
        42: [
          crossReference({
            number: 100,
            author: "alice",
            createdAt: daysAgo(50),
            referencedAt: daysAgo(50),
          }),
        ],
        100: [{ event: "committed", author: { name: "Alice", date: daysAgo(2) } }],
      },
    });
    const record = await evaluateAssignee(github, "o", "r", issueFixture({ created_at: daysAgo(400) }), "alice", CONFIG);
    expect(record.action).toBe("skipped");
  });

  test("other people's traffic on a linked pull request does not keep the assignment", async () => {
    const github = makeGitHub({
      timelines: {
        42: [
          crossReference({
            number: 100,
            author: "alice",
            createdAt: daysAgo(50),
            referencedAt: daysAgo(50),
          }),
        ],
        // Reviews, comments, and bot writes long after Alice stopped working.
        100: [
          timelineComment("bob", daysAgo(1)),
          { event: "reviewed", actor: { login: "carol", type: "User" }, submitted_at: daysAgo(1) },
          { event: "labeled", actor: { login: "ci[bot]", type: "Bot" }, created_at: daysAgo(1) },
        ],
      },
    });
    const record = await evaluateAssignee(github, "o", "r", issueFixture({ created_at: daysAgo(400) }), "alice", CONFIG);
    expect(record.action).toBe("warned");
    expect(Math.floor(record.inactiveDays)).toBe(50);
  });

  test("a pull request by another contributor does not shield the assignee", async () => {
    const github = makeGitHub({
      timelines: {
        42: [
          crossReference({
            number: 100,
            author: "bob",
            createdAt: daysAgo(1),
            referencedAt: daysAgo(1),
            actor: "bob",
          }),
        ],
        100: [],
      },
    });
    const record = await evaluateAssignee(github, "o", "r", issueFixture({ created_at: daysAgo(400) }), "alice", CONFIG);
    expect(record.action).toBe("warned");
  });

  test("skips a cross-repository reference instead of failing on it", async () => {
    const github = makeGitHub({
      timelines: {
        42: [
          crossReference({
            number: 7,
            author: "alice",
            createdAt: daysAgo(1),
            referencedAt: daysAgo(1),
            actor: "carol",
            repositoryFullName: "someone/else",
          }),
        ],
      },
    });
    const record = await evaluateAssignee(github, "o", "r", issueFixture({ created_at: daysAgo(400) }), "alice", CONFIG);
    expect(record.action).toBe("warned");
  });

  test("caps how many linked pull requests it inspects", async () => {
    const references = [];
    const timelines = { 42: references };
    for (let i = 0; i < MAX_LINKED_PULL_REQUESTS + 5; i += 1) {
      const number = 200 + i;
      references.push(
        crossReference({
          number,
          author: "alice",
          createdAt: daysAgo(60),
          referencedAt: daysAgo(60 - i),
          actor: "carol",
        })
      );
      timelines[number] = [];
    }
    const github = makeGitHub({ timelines });
    await evaluateAssignee(github, "o", "r", issueFixture({ created_at: daysAgo(400) }), "alice", CONFIG);

    const prTimelineReads = github.paginate.mock.calls.filter(
      ([endpoint, params]) =>
        endpoint === github.rest.issues.listEventsForTimeline && params.issue_number !== 42
    );
    expect(prTimelineReads).toHaveLength(MAX_LINKED_PULL_REQUESTS);
  });

  test("dry run reports the transition without mutating anything", async () => {
    const github = makeGitHub({
      timelines: { 42: [] },
      comments: { 42: [comment("alice", "warned", daysAgo(16))] },
    });
    const record = await evaluateAssignee(
      github,
      "o",
      "r",
      issueFixture({ created_at: daysAgo(40) }),
      "alice",
      { ...CONFIG, dryRun: true }
    );
    expect(record.action).toBe("unassigned");
    expect(github.rest.issues.createComment).not.toHaveBeenCalled();
    expect(github.rest.issues.removeAssignees).not.toHaveBeenCalled();
  });

  test("assignment by a maintainer restarts the clock", async () => {
    const github = makeGitHub({
      timelines: {
        42: [
          {
            event: "assigned",
            actor: { login: "maintainer", type: "User" },
            assignee: { login: "alice" },
            created_at: daysAgo(2),
          },
        ],
      },
      comments: { 42: [comment("alice", "warned", daysAgo(20))] },
    });
    const record = await evaluateAssignee(github, "o", "r", issueFixture({ created_at: daysAgo(400) }), "alice", CONFIG);
    expect(record.action).toBe("recovered");
  });
});

// -- run --------------------------------------------------------------------

describe("run", () => {
  let summaryPath;

  beforeEach(() => {
    summaryPath = path.join(
      fs.mkdtempSync(path.join(os.tmpdir(), "unassign-summary-")),
      "summary.md"
    );
  });

  function options(extra = {}) {
    return { warnAfterDays: 15, unassignAfterDays: 30, now: NOW, summaryPath, ...extra };
  }

  test("does nothing when no accepted issue is assigned", async () => {
    const github = makeGitHub({ issues: [] });
    const { stats } = await run(github, CONTEXT, options());
    expect(stats.evaluated).toBe(0);
    expect(github.rest.issues.createComment).not.toHaveBeenCalled();
  });

  test("rejects a configuration with no grace window", async () => {
    const github = makeGitHub({ issues: [] });
    await expect(run(github, CONTEXT, options({ unassignAfterDays: 15 }))).rejects.toThrow(
      /grace window/
    );
  });

  test("throws when the issue listing fails, so the job goes red", async () => {
    const github = makeGitHub({ failListForRepo: true });
    await expect(run(github, CONTEXT, options())).rejects.toThrow(/listing accepted issues/);
  });

  test("ignores pull requests returned by the issues endpoint", async () => {
    const github = makeGitHub({
      issues: [
        issueFixture({
          number: 99,
          pull_request: { url: "https://..." },
          created_at: daysAgo(400),
        }),
      ],
    });
    const { stats } = await run(github, CONTEXT, options());
    expect(stats.evaluated).toBe(0);
    expect(github.rest.issues.createComment).not.toHaveBeenCalled();
  });

  test("ignores an issue carrying an exempt label", async () => {
    const github = makeGitHub({
      issues: [
        issueFixture({
          created_at: daysAgo(400),
          labels: [{ name: "accepted" }, { name: "Hold" }],
        }),
      ],
    });
    const { stats } = await run(github, CONTEXT, options());
    expect(stats.issues).toBe(0);
    expect(github.rest.issues.createComment).not.toHaveBeenCalled();
  });

  test("ignores a locked issue, which cannot receive a warning", async () => {
    const github = makeGitHub({
      issues: [issueFixture({ created_at: daysAgo(400), locked: true })],
    });
    const { stats } = await run(github, CONTEXT, options());
    expect(stats.issues).toBe(0);
  });

  test("ignores an issue with no assignee", async () => {
    const github = makeGitHub({
      issues: [issueFixture({ created_at: daysAgo(400), assignees: [] })],
    });
    const { stats } = await run(github, CONTEXT, options());
    expect(stats.issues).toBe(0);
  });

  test("evaluates each assignee of a shared issue independently", async () => {
    const github = makeGitHub({
      issues: [
        issueFixture({
          created_at: daysAgo(400),
          assignees: [{ login: "alice" }, { login: "bob" }],
        }),
      ],
      timelines: { 42: [timelineComment("alice", daysAgo(1))] },
      // Bob was warned long enough ago to be removed; Alice is active.
      comments: { 42: [comment("bob", "warned", daysAgo(16))] },
    });

    const { stats, records } = await run(github, CONTEXT, options());

    expect(records.find((r) => r.assignee === "alice").action).toBe("skipped");
    expect(records.find((r) => r.assignee === "bob").action).toBe("unassigned");
    expect(stats.unassigned).toBe(1);
    expect(github.rest.issues.removeAssignees).toHaveBeenCalledTimes(1);
    expect(github.rest.issues.removeAssignees).toHaveBeenCalledWith(
      expect.objectContaining({ assignees: ["bob"] })
    );
  });

  test("one assignee's warning does not suppress another's", async () => {
    const github = makeGitHub({
      issues: [
        issueFixture({
          created_at: daysAgo(20),
          assignees: [{ login: "alice" }, { login: "bob" }],
        }),
      ],
      timelines: { 42: [] },
      comments: { 42: [comment("alice", "warned", daysAgo(2))] },
    });

    const { records } = await run(github, CONTEXT, options());

    expect(records.find((r) => r.assignee === "alice").action).toBe("skipped");
    expect(records.find((r) => r.assignee === "bob").action).toBe("warned");
    expect(github.rest.issues.createComment).toHaveBeenCalledTimes(1);
    expect(github.rest.issues.createComment.mock.calls[0][0].body).toContain("@bob");
  });

  test("a second run over the same state changes nothing further", async () => {
    const issues = [issueFixture({ created_at: daysAgo(20) })];
    const posted = [];
    const github = makeGitHub({ issues, timelines: { 42: [] }, comments: { 42: posted } });
    github.rest.issues.createComment.mockImplementation(async ({ body }) => {
      posted.push({ body, created_at: NOW.toISOString(), user: WORKFLOW_BOT });
      return {};
    });

    await run(github, CONTEXT, options());
    expect(github.rest.issues.createComment).toHaveBeenCalledTimes(1);

    await run(github, CONTEXT, options());
    expect(github.rest.issues.createComment).toHaveBeenCalledTimes(1);
  });

  test("counts a failed evaluation without touching the assignment", async () => {
    const github = makeGitHub({
      issues: [issueFixture({ created_at: daysAgo(400) })],
      failTimelineFor: [42],
    });
    const { stats } = await run(github, CONTEXT, options());
    expect(stats.errors).toBe(1);
    expect(github.rest.issues.removeAssignees).not.toHaveBeenCalled();
  });

  test("dry run leaves GitHub untouched", async () => {
    const github = makeGitHub({
      issues: [issueFixture({ created_at: daysAgo(400) })],
      timelines: { 42: [] },
    });
    const { stats } = await run(github, CONTEXT, options({ dryRun: true }));
    expect(stats.warned).toBe(1);
    expect(github.rest.issues.createComment).not.toHaveBeenCalled();
    expect(github.rest.issues.removeAssignees).not.toHaveBeenCalled();
  });

  test("writes a step summary explaining every transition", async () => {
    const github = makeGitHub({
      issues: [issueFixture({ created_at: daysAgo(20) })],
      timelines: { 42: [] },
    });
    await run(github, CONTEXT, options());

    const summary = fs.readFileSync(summaryPath, "utf8");
    expect(summary).toContain("Assignee inactivity sweep");
    expect(summary).toContain("#42");
    expect(summary).toContain("@alice");
    expect(summary).toContain("warned");
  });

  test("paginates every list read rather than taking the first page", async () => {
    const github = makeGitHub({
      issues: [issueFixture({ created_at: daysAgo(20) })],
      timelines: {
        42: [
          crossReference({
            number: 100,
            author: "alice",
            createdAt: daysAgo(20),
            referencedAt: daysAgo(20),
          }),
        ],
        100: [],
      },
    });
    await run(github, CONTEXT, options());

    const reads = github.paginate.mock.calls.map(([endpoint, params]) => [endpoint, params]);
    expect(reads.every(([, params]) => params.per_page === 100)).toBe(true);

    const [, listParams] = reads.find(([endpoint]) => endpoint === github.rest.issues.listForRepo);
    expect(listParams).toMatchObject({ state: "open", labels: "accepted" });

    const timelineReads = reads.filter(
      ([endpoint]) => endpoint === github.rest.issues.listEventsForTimeline
    );
    expect(timelineReads.map(([, params]) => params.issue_number)).toEqual([42, 100]);
    expect(
      reads.some(([endpoint]) => endpoint === github.rest.issues.listComments)
    ).toBe(true);
  });
});
