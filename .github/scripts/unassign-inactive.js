// SPDX-License-Identifier: Apache-2.0
// Per-assignee inactivity state machine for accepted issues. Activity is
// attributed to one assignee at a time, removal always follows a recorded
// warning, every read fails closed, and every write is marker-guarded so a
// half-finished run resumes rather than repeating itself.

"use strict";

const fs = require("fs");

const ACCEPTED_LABEL = "accepted";
const MARKER_NAMESPACE = "unassign-inactive";
const DAY_MS = 24 * 60 * 60 * 1000;

// Triage noise (labels, milestones, mentions, renames) is deliberately absent.
// `assigned` is handled separately: its actor is the maintainer, not the assignee.
const HUMAN_EVENT_TYPES = new Set([
  "commented",
  "committed",
  "reviewed",
  "review_requested",
  "head_ref_force_pushed",
  "ready_for_review",
  "referenced",
  "cross-referenced",
  "connected",
  "reopened",
]);

// -- Small helpers ----------------------------------------------------------

function normalizeLogin(login) {
  return typeof login === "string" ? login.trim().toLowerCase() : "";
}

function sameLogin(a, b) {
  const left = normalizeLogin(a);
  return left !== "" && left === normalizeLogin(b);
}

// `type` is absent on some timeline shapes, so check the login suffix too.
function isBotActor(actor) {
  return (
    !actor ||
    actor.type === "Bot" ||
    /\[bot\]$/i.test(typeof actor.login === "string" ? actor.login : "")
  );
}

function toDate(value) {
  if (!value) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

function laterOf(a, b) {
  if (!a) return b || null;
  if (!b) return a;
  return a > b ? a : b;
}

function daysBetween(later, earlier) {
  return (later.getTime() - earlier.getTime()) / DAY_MS;
}

function plural(count, word) {
  return `${count} ${word}${count === 1 ? "" : "s"}`;
}

// Reviews use submitted_at, commits use author.date.
function eventTimestamp(event) {
  return toDate(
    event.created_at ||
      event.submitted_at ||
      event.committed_at ||
      (event.author && event.author.date) ||
      (event.committer && event.committer.date)
  );
}

// Per-assignee and per-transition, so one assignee's warning is never mistaken
// for another's and a removal notice is never counted as a warning.
function warningMarker(login) {
  return `<!-- ${MARKER_NAMESPACE}:warned:${normalizeLogin(login)} -->`;
}

function removalMarker(login) {
  return `<!-- ${MARKER_NAMESPACE}:removed:${normalizeLogin(login)} -->`;
}

// -- GitHub reads (all paginated, all fail-closed) --------------------------

/** Every timeline event for an issue or pull request; null on failure. */
async function fetchTimeline(octokit, owner, repo, number) {
  try {
    return await octokit.paginate(octokit.rest.issues.listEventsForTimeline, {
      owner,
      repo,
      issue_number: number,
      per_page: 100,
    });
  } catch (err) {
    console.warn(`[fetchTimeline] #${number}: ${err.message}`);
    return null;
  }
}

/** Every issue comment; null on failure, never an empty "no warning" result. */
async function fetchComments(octokit, owner, repo, issueNumber) {
  try {
    return await octokit.paginate(octokit.rest.issues.listComments, {
      owner,
      repo,
      issue_number: issueNumber,
      per_page: 100,
    });
  } catch (err) {
    console.warn(`[fetchComments] #${issueNumber}: ${err.message}`);
    return null;
  }
}

/** Every commit on a pull request, with GitHub logins attached; null on failure. */
async function fetchPullRequestCommits(octokit, owner, repo, pullNumber) {
  try {
    return await octokit.paginate(octokit.rest.pulls.listCommits, {
      owner,
      repo,
      pull_number: pullNumber,
      per_page: 100,
    });
  } catch (err) {
    console.warn(`[fetchPullRequestCommits] #${pullNumber}: ${err.message}`);
    return null;
  }
}

// -- Activity attribution ---------------------------------------------------

/** Reduce an issue timeline to { lastActivityAt, linkedPullRequests }. */
function collectIssueSignals(timeline, assigneeLogin) {
  let lastActivityAt = null;
  const linkedPullRequests = new Map();

  for (const event of timeline || []) {
    const at = eventTimestamp(event);

    // Assignment restarts this assignee's clock. Read before the bot filter,
    // since automation may be what does the assigning.
    if (
      event.event === "assigned" &&
      sameLogin(event.assignee && event.assignee.login, assigneeLogin)
    ) {
      lastActivityAt = laterOf(lastActivityAt, at);
      continue;
    }

    // Collected whoever made the reference; whether the pull request is this
    // assignee's work is decided by inspecting the pull request itself.
    if (event.event === "cross-referenced") {
      const source = event.source && event.source.issue;
      if (source && source.pull_request && Number.isInteger(source.number)) {
        const existing = linkedPullRequests.get(source.number);
        const createdAt = toDate(source.created_at);
        if (!existing) {
          linkedPullRequests.set(source.number, {
            number: source.number,
            authorLogin: (source.user && source.user.login) || null,
            createdAt,
            repositoryFullName:
              (source.repository && source.repository.full_name) || null,
            referencedAt: at,
          });
        } else {
          existing.referencedAt = laterOf(existing.referencedAt, at);
        }
      }
    }

    const actor = event.actor || event.user;
    if (isBotActor(actor)) continue;
    if (!sameLogin(actor.login, assigneeLogin)) continue;
    if (!HUMAN_EVENT_TYPES.has(event.event)) continue;
    if (!at) continue;

    lastActivityAt = laterOf(lastActivityAt, at);
  }

  return {
    lastActivityAt,
    linkedPullRequests: Array.from(linkedPullRequests.values()),
  };
}

/**
 * The assignee's own most recent activity inside one pull request, or
 * { error: true }. Deliberately not the pull request's `updated_at`: that moves
 * on anyone's comment or CI write, keeping an inactive assignee assigned.
 */
async function collectPullRequestActivity(
  octokit,
  owner,
  repo,
  pullRequest,
  assigneeLogin
) {
  const authoredByAssignee = sameLogin(pullRequest.authorLogin, assigneeLogin);

  const timeline = await fetchTimeline(octokit, owner, repo, pullRequest.number);
  if (timeline === null) return { error: true };

  // Opening the pull request is itself the assignee's work.
  let lastActivityAt = authoredByAssignee ? pullRequest.createdAt : null;

  for (const event of timeline) {
    const at = eventTimestamp(event);
    if (!at) continue;

    // Timeline `committed` events carry only a git identity, so they are
    // attributed below from the commit list instead.
    if (event.event === "committed") continue;

    const actor = event.actor || event.user;
    if (isBotActor(actor)) continue;
    if (!sameLogin(actor.login, assigneeLogin)) continue;
    if (!HUMAN_EVENT_TYPES.has(event.event)) continue;

    lastActivityAt = laterOf(lastActivityAt, at);
  }

  const commits = await fetchPullRequestCommits(octokit, owner, repo, pullRequest.number);
  if (commits === null) return { error: true };

  for (const item of commits) {
    const detail = item.commit || {};
    const authored = sameLogin(item.author && item.author.login, assigneeLogin);
    const committed = sameLogin(item.committer && item.committer.login, assigneeLogin);
    if (!authored && !committed) continue;

    lastActivityAt = laterOf(
      lastActivityAt,
      toDate(
        (authored && detail.author && detail.author.date) ||
          (detail.committer && detail.committer.date) ||
          (detail.author && detail.author.date)
      )
    );
  }

  return { lastActivityAt };
}

/**
 * One assignee's last activity across the issue and every linked pull request,
 * or { error: true } if a read that could have proven activity failed. Scanning
 * stops only once activity newer than activeSince settles the outcome as
 * active, so no linked pull request is skipped while it could change it.
 */
async function resolveLastActivity(
  octokit,
  owner,
  repo,
  issue,
  assigneeLogin,
  activeSince
) {
  const timeline = await fetchTimeline(octokit, owner, repo, issue.number);
  if (timeline === null) return { error: true };

  const signals = collectIssueSignals(timeline, assigneeLogin);
  let lastActivityAt = signals.lastActivityAt;

  const repositoryFullName = `${owner}/${repo}`;
  const candidates = signals.linkedPullRequests
    // Cross-repository sources are unreadable with this token; the reference
    // event itself is still credited above when the assignee made it.
    .filter(
      (pr) =>
        !pr.repositoryFullName || pr.repositoryFullName === repositoryFullName
    )
    // Most recently referenced first, so the early exit below usually lands on
    // the first pull request.
    .sort(
      (a, b) =>
        (b.referencedAt ? b.referencedAt.getTime() : 0) -
        (a.referencedAt ? a.referencedAt.getTime() : 0)
    );

  for (const pullRequest of candidates) {
    if (activeSince && lastActivityAt && lastActivityAt >= activeSince) break;

    const activity = await collectPullRequestActivity(
      octokit,
      owner,
      repo,
      pullRequest,
      assigneeLogin
    );
    if (activity.error) return { error: true };
    lastActivityAt = laterOf(lastActivityAt, activity.lastActivityAt);
  }

  return { lastActivityAt };
}

// -- Warning / removal state ------------------------------------------------

/**
 * Newest notices addressed to one assignee, as
 * { warnedAt, supersededWarningAt, removalNoticeAt }. Only bot-authored
 * notices count, so quoting a marker cannot move the clock; a notice predating
 * the assignee's last activity belongs to a cycle they already cancelled.
 */
function readTransitionState(comments, assigneeLogin, lastActivityAt) {
  const warnTag = warningMarker(assigneeLogin);
  const removeTag = removalMarker(assigneeLogin);

  let warnedAt = null;
  let supersededWarningAt = null;
  let removalNoticeAt = null;

  for (const comment of comments || []) {
    const body = comment && comment.body;
    if (typeof body !== "string") continue;
    if (!comment.user || !isBotActor(comment.user)) continue;

    const createdAt = toDate(comment.created_at);
    if (!createdAt) continue;

    const superseded = Boolean(lastActivityAt) && createdAt < lastActivityAt;

    if (body.includes(warnTag)) {
      if (superseded) {
        supersededWarningAt = laterOf(supersededWarningAt, createdAt);
      } else {
        warnedAt = laterOf(warnedAt, createdAt);
      }
    } else if (body.includes(removeTag) && !superseded) {
      removalNoticeAt = laterOf(removalNoticeAt, createdAt);
    }
  }

  return { warnedAt, supersededWarningAt, removalNoticeAt };
}

function buildWarningComment(assigneeLogin, inactiveDays, graceDays) {
  return [
    warningMarker(assigneeLogin),
    `@${assigneeLogin} — this issue has had no activity from you for ` +
      `**${plural(inactiveDays, "day")}**.`,
    "",
    "If you are still working on it, leave a comment or push to a linked pull " +
      `request within the next **${plural(graceDays, "day")}** to keep the ` +
      "assignment. Otherwise it will be released so another contributor can " +
      "pick it up. The issue itself stays open either way.",
    "",
    "Comments, reviews, and commits from other people do not count here — " +
      "only your own activity on this issue or on a pull request linked to it.",
    "",
    "_Posted automatically by the assignee inactivity workflow. Maintainers " +
      "can extend or override any assignment._",
  ].join("\n");
}

function buildRemovalComment(assigneeLogin, inactiveDays, warnedAt) {
  const warnedLine = warnedAt
    ? `A warning was posted on ${warnedAt.toISOString().slice(0, 10)} and the ` +
      "grace period has now passed."
    : "A warning was posted earlier and the grace period has now passed.";
  return [
    removalMarker(assigneeLogin),
    `@${assigneeLogin} — releasing this assignment after ` +
      `**${plural(inactiveDays, "day")}** without activity from you. ` +
      warnedLine,
    "",
    "The issue stays open and `accepted`, so anyone can pick it up. If you are " +
      "still interested, comment here and a maintainer can reassign it to you.",
    "",
    "_Posted automatically by the assignee inactivity workflow._",
  ].join("\n");
}

// -- Per-assignee state machine ---------------------------------------------

/**
 * Evaluate and, unless dryRun, advance one (issue, assignee) pair to one of
 * "skipped" (active, or still inside the grace window), "recovered", "warned",
 * "unassigned", or "error" (a read or write failed; nothing was mutated).
 */
async function evaluateAssignee(octokit, owner, repo, issue, assigneeLogin, config) {
  const { warnAfterDays, unassignAfterDays, dryRun, now } = config;
  const issueNumber = issue.number;
  const graceDays = unassignAfterDays - warnAfterDays;

  const record = {
    issueNumber,
    assignee: assigneeLogin,
    action: "skipped",
    reason: "",
    lastActivityAt: null,
    inactiveDays: 0,
    warnedAt: null,
  };

  const activity = await resolveLastActivity(
    octokit,
    owner,
    repo,
    issue,
    assigneeLogin,
    new Date(now.getTime() - warnAfterDays * DAY_MS)
  );
  if (activity.error) {
    record.action = "error";
    record.reason = "activity lookup failed; left assigned";
    return record;
  }

  // Fallback only when the timeline yields nothing; normally the `assigned`
  // event supplies a later, fairer start.
  const lastActivityAt =
    activity.lastActivityAt || toDate(issue.created_at) || now;
  const inactiveDays = Math.max(0, daysBetween(now, lastActivityAt));

  record.lastActivityAt = lastActivityAt;
  record.inactiveDays = inactiveDays;

  const comments = await fetchComments(octokit, owner, repo, issueNumber);
  if (comments === null) {
    record.action = "error";
    record.reason = "warning lookup failed; left assigned";
    return record;
  }

  const state = readTransitionState(comments, assigneeLogin, lastActivityAt);
  record.warnedAt = state.warnedAt;

  // A: still active.
  if (inactiveDays < warnAfterDays) {
    if (state.supersededWarningAt) {
      record.action = "recovered";
      record.reason = "activity resumed; pending warning no longer applies";
    } else {
      record.reason = `active ${Math.floor(inactiveDays)}d ago`;
    }
    return record;
  }

  // B: inactive, not yet warned. A warning always precedes removal, even for an
  // assignee first seen long past the removal threshold.
  if (!state.warnedAt) {
    record.action = "warned";
    record.reason = `inactive ${Math.floor(inactiveDays)}d; warning posted`;
    if (dryRun) return record;

    try {
      await octokit.rest.issues.createComment({
        owner,
        repo,
        issue_number: issueNumber,
        body: buildWarningComment(
          assigneeLogin,
          Math.floor(inactiveDays),
          graceDays
        ),
      });
    } catch (err) {
      console.warn(`  [#${issueNumber}] warning comment failed: ${err.message}`);
      record.action = "error";
      record.reason = "warning comment failed; retried on the next run";
    }
    return record;
  }

  // C: warned. Removal needs the inactivity threshold *and* a full grace window
  // measured from the warning, so a warning posted today cannot remove today.
  const daysSinceWarning = Math.max(0, daysBetween(now, state.warnedAt));
  if (inactiveDays < unassignAfterDays || daysSinceWarning < graceDays) {
    // Both conditions must clear, so the wait is the longer of the two.
    const remaining = Math.max(
      unassignAfterDays - inactiveDays,
      graceDays - daysSinceWarning
    );
    record.reason =
      `warned ${Math.floor(daysSinceWarning)}d ago; ` +
      `${Math.max(0, Math.ceil(remaining))}d of grace left`;
    return record;
  }

  // D: removal. Notice first, so the contributor is always told; its marker
  // makes a resumed run skip straight to the removal it did not reach.
  record.action = "unassigned";
  record.reason = `inactive ${Math.floor(inactiveDays)}d; warned ${Math.floor(daysSinceWarning)}d ago`;
  if (dryRun) return record;

  if (!state.removalNoticeAt) {
    try {
      await octokit.rest.issues.createComment({
        owner,
        repo,
        issue_number: issueNumber,
        body: buildRemovalComment(
          assigneeLogin,
          Math.floor(inactiveDays),
          state.warnedAt
        ),
      });
    } catch (err) {
      console.warn(`  [#${issueNumber}] removal notice failed: ${err.message}`);
      record.action = "error";
      record.reason = "removal notice failed; assignment kept until it lands";
      return record;
    }
  }

  try {
    await octokit.rest.issues.removeAssignees({
      owner,
      repo,
      issue_number: issueNumber,
      assignees: [assigneeLogin],
    });
  } catch (err) {
    console.warn(`  [#${issueNumber}] removeAssignees failed: ${err.message}`);
    record.action = "error";
    record.reason = "removal failed; the notice is on record and will be reused";
  }
  return record;
}

// -- Reporting --------------------------------------------------------------

function formatDays(value) {
  return `${Math.floor(value)}d`;
}

function buildRunSummary(records, stats, config) {
  const { warnAfterDays, unassignAfterDays, dryRun, exemptLabels } = config;
  const lines = [
    "## Assignee inactivity sweep",
    "",
    `Warn after ${plural(warnAfterDays, "day")} · unassign after ` +
      `${plural(unassignAfterDays, "day")} · ` +
      `${dryRun ? "**dry run — nothing was mutated**" : "live"}`,
  ];

  if (exemptLabels.length > 0) {
    lines.push("", `Exempt labels: ${exemptLabels.map((l) => `\`${l}\``).join(", ")}`);
  }

  lines.push(
    "",
    `Evaluated ${plural(stats.evaluated, "assignment")} across ` +
      `${plural(stats.issues, "issue")}: ${stats.warned} warned, ` +
      `${stats.unassigned} unassigned, ${stats.recovered} recovered, ` +
      `${stats.skipped} unchanged, ${stats.errors} failed.`
  );

  const notable = records.filter((r) => r.action !== "skipped");
  if (notable.length > 0) {
    lines.push(
      "",
      "| Issue | Assignee | Action | Inactive | Why |",
      "| --- | --- | --- | --- | --- |"
    );
    for (const record of notable) {
      lines.push(
        `| #${record.issueNumber} | @${record.assignee} | ${record.action} | ` +
          `${formatDays(record.inactiveDays)} | ${record.reason} |`
      );
    }
  }

  if (stats.errors > 0) {
    lines.push(
      "",
      "Failed evaluations left their assignments untouched. The sweep is " +
        "idempotent — re-running it retries only what did not complete."
    );
  }

  return `${lines.join("\n")}\n`;
}

function writeStepSummary(text, summaryPath) {
  if (!summaryPath) return;
  try {
    fs.appendFileSync(summaryPath, text, "utf8");
  } catch (err) {
    console.warn(`[unassign-inactive] step summary write failed: ${err.message}`);
  }
}

// -- Entry point ------------------------------------------------------------

const ACTION_STATS = {
  skipped: "skipped",
  recovered: "recovered",
  warned: "warned",
  unassigned: "unassigned",
  error: "errors",
};

function parseExemptLabels(value) {
  const raw = Array.isArray(value) ? value : String(value || "").split(",");
  return raw.map((label) => String(label).trim().toLowerCase()).filter(Boolean);
}

function issueLabelNames(issue) {
  return (issue.labels || [])
    .map((label) => (typeof label === "string" ? label : label && label.name))
    .filter(Boolean)
    .map((name) => name.toLowerCase());
}

function issueAssignees(issue) {
  if (Array.isArray(issue.assignees) && issue.assignees.length > 0) {
    return issue.assignees;
  }
  return issue.assignee ? [issue.assignee] : [];
}

/**
 * Sweep every open `accepted` issue, advancing each assignee independently.
 * Throws when the run cannot be trusted at all; otherwise returns
 * { stats, records }, where a non-zero stats.errors means the caller should
 * fail the job so the next run retries those assignments.
 */
async function run(octokit, context, config = {}) {
  const warnAfterDays = Number(config.warnAfterDays ?? 15);
  const unassignAfterDays = Number(config.unassignAfterDays ?? 30);
  const dryRun = Boolean(config.dryRun);
  const now = config.now instanceof Date ? config.now : new Date();
  const exemptLabels = parseExemptLabels(config.exemptLabels ?? "hold");
  const summaryPath =
    config.summaryPath ?? process.env.GITHUB_STEP_SUMMARY ?? null;

  if (!Number.isFinite(warnAfterDays) || warnAfterDays < 1) {
    throw new Error(`warnAfterDays must be a positive number, got ${config.warnAfterDays}`);
  }
  if (!Number.isFinite(unassignAfterDays) || unassignAfterDays <= warnAfterDays) {
    throw new Error(
      `unassignAfterDays (${unassignAfterDays}) must be greater than ` +
        `warnAfterDays (${warnAfterDays}) so a warned contributor gets a grace window`
    );
  }

  const { owner, repo } = context.repo;

  console.log(
    `[unassign-inactive] warn=${warnAfterDays}d unassign=${unassignAfterDays}d ` +
      `dryRun=${dryRun} exempt=[${exemptLabels.join(", ")}]`
  );

  let issues;
  try {
    issues = await octokit.paginate(octokit.rest.issues.listForRepo, {
      owner,
      repo,
      state: "open",
      labels: ACCEPTED_LABEL,
      per_page: 100,
    });
  } catch (err) {
    throw new Error(`listing accepted issues failed: ${err.message}`);
  }

  const candidates = issues.filter((issue) => {
    if (issue.pull_request) return false;
    if (issueAssignees(issue).length === 0) return false;
    // A locked issue cannot receive the warning that must precede removal.
    if (issue.locked) {
      console.log(`[#${issue.number}] skipped: locked`);
      return false;
    }
    const labels = issueLabelNames(issue);
    const exempt = exemptLabels.find((label) => labels.includes(label));
    if (exempt) {
      console.log(`[#${issue.number}] skipped: exempt label "${exempt}"`);
      return false;
    }
    return true;
  });

  console.log(`[unassign-inactive] ${candidates.length} issue(s) in scope`);

  const records = [];
  const stats = {
    issues: candidates.length,
    evaluated: 0,
    warned: 0,
    unassigned: 0,
    recovered: 0,
    skipped: 0,
    errors: 0,
  };

  for (const issue of candidates) {
    console.log(`\n#${issue.number} ${issue.title}`);
    for (const assignee of issueAssignees(issue)) {
      const record = await evaluateAssignee(
        octokit,
        owner,
        repo,
        issue,
        assignee.login,
        { warnAfterDays, unassignAfterDays, dryRun, now }
      );
      records.push(record);
      stats.evaluated += 1;
      stats[ACTION_STATS[record.action]] += 1;
      console.log(`  @${record.assignee}: ${record.action} — ${record.reason}`);
      if (record.action === "error") {
        console.log(
          `::warning title=Inactivity sweep::#${record.issueNumber} @${record.assignee}: ${record.reason}`
        );
      }
    }
  }

  const summary = buildRunSummary(records, stats, {
    warnAfterDays,
    unassignAfterDays,
    dryRun,
    exemptLabels,
  });
  console.log(`\n${summary}`);
  writeStepSummary(summary, summaryPath);

  return { stats, records };
}

module.exports = {
  run,
  evaluateAssignee,
  resolveLastActivity,
  collectIssueSignals,
  collectPullRequestActivity,
  readTransitionState,
  fetchTimeline,
  fetchComments,
  buildWarningComment,
  buildRemovalComment,
  buildRunSummary,
  warningMarker,
  removalMarker,
};
