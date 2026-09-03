// SPDX-License-Identifier: Apache-2.0
// Per-assignee inactivity state machine for accepted issues.
// Uses the Timeline API for human-only activity detection. Bot writes,
// label churn, and automation do NOT reset the inactivity clock.
// All list operations are paginated. API errors fail-closed.

"use strict";

const WARN_LABEL = "unassign-warned";
const ACCEPTED_LABEL = "accepted";
const WARNING_MARKER = "<!-- unassign-inactive-warning -->";

// Timeline event types that signal real contributor activity.
const HUMAN_EVENT_TYPES = new Set([
  "commented",
  "committed",
  "reviewed",
  "review_requested",
  "head_ref_force_pushed",
  "base_ref_changed",
  "referenced",
  "cross-referenced",
]);

// -- Timeline API -----------------------------------------------------------

/** Fetch all timeline events for an issue, paginated. Returns null on error. */
async function fetchTimeline(octokit, owner, repo, issueNumber) {
  try {
    return await octokit.paginate(octokit.rest.issues.listEventsForTimeline, {
      owner,
      repo,
      issue_number: issueNumber,
      per_page: 100,
      headers: { accept: "application/vnd.github.mockingbird-preview+json" },
    });
  } catch (err) {
    console.warn(`[fetchTimeline] #${issueNumber}: ${err.message}`);
    return null;
  }
}

/**
 * Return the most recent Date of human activity by assigneeLogin on the
 * given timeline, or null if none found.
 */
function getLastHumanActivityDate(timeline, assigneeLogin) {
  let latest = null;

  for (const event of timeline) {
    const actor =
      event.actor || event.user || (event.author && { login: event.author.name });

    // Skip bot actors.
    if (!actor || actor.type === "Bot" || /\[bot\]$/.test(actor.login || "")) {
      continue;
    }

    let isQualifying = false;

    if (actor.login === assigneeLogin) {
      if (HUMAN_EVENT_TYPES.has(event.event)) {
        isQualifying = true;
      }
    } else {
      // Cross-referenced events: match when the source PR is authored by the assignee.
      if (event.event === "cross-referenced") {
        const source = event.source && event.source.issue;
        if (source && source.pull_request && source.user && source.user.login === assigneeLogin) {
          isQualifying = true;
        }
      }
    }

    // Manual maintainer overrides: reassignment or removing the warning label resets the clock
    if (event.event === "assigned" && event.assignee && event.assignee.login === assigneeLogin) {
      isQualifying = true;
    }
    if (event.event === "unlabeled" && event.label && event.label.name === WARN_LABEL) {
      isQualifying = true;
    }

    if (!isQualifying) continue;

    const ts = event.created_at || event.submitted_at || event.committed_at;
    if (!ts) continue;

    const d = new Date(ts);
    if (!latest || d > latest) latest = d;
  }

  return latest;
}

// -- Linked-PR detection (paginated, per-assignee, fail-closed) -------------

/**
 * Search for PRs authored by assigneeLogin that reference issueNumber.
 * Returns { hasActivity, date, error? }. Fails-closed on API error.
 */
async function findLinkedPRActivity(octokit, owner, repo, issueNumber, assigneeLogin) {
  const query = `repo:${owner}/${repo} is:pr author:${assigneeLogin}`;
  // Word-boundary pattern: #42 must not match #420.
  const issuePattern = new RegExp(`#${issueNumber}(?:\\D|$)`);
  let latest = null;
  let page = 1;

  try {
    while (true) { // eslint-disable-line no-constant-condition
      const { data } = await octokit.rest.search.issuesAndPullRequests({
        q: query,
        per_page: 100,
        page,
      });

      for (const pr of data.items) {
        if (!pr.pull_request) continue;
        if (!issuePattern.test(pr.body || "")) continue;

        const updated = new Date(pr.updated_at);
        if (!latest || updated > latest) latest = updated;
      }

      if (data.items.length < 100) break;
      page += 1;
    }

    return { hasActivity: latest !== null, date: latest };
  } catch (err) {
    console.warn(
      `[findLinkedPRActivity] ${assigneeLogin} on #${issueNumber}: ${err.message} (fail-closed)`
    );
    return { hasActivity: true, date: null, error: true };
  }
}

// -- Warning comment helpers ------------------------------------------------

/** Check if a warning comment already exists for this assignee since their last activity. */
async function warningCommentExists(octokit, owner, repo, issueNumber, assigneeLogin, lastHumanActivity) {
  try {
    const comments = await octokit.paginate(octokit.rest.issues.listComments, {
      owner,
      repo,
      issue_number: issueNumber,
      per_page: 100,
    });
    return comments.some((c) => {
      if (!c.body || !c.body.includes(WARNING_MARKER) || !c.body.includes(`@${assigneeLogin}`)) {
        return false;
      }
      if (lastHumanActivity && c.created_at) {
        if (new Date(c.created_at) < lastHumanActivity) {
          return false;
        }
      }
      return true;
    });
  } catch (err) {
    console.warn(`[warningCommentExists] #${issueNumber}: ${err.message}`);
    return false;
  }
}

function buildWarningComment(assigneeLogin, warnAfterDays, unassignAfterDays) {
  const remaining = unassignAfterDays - warnAfterDays;
  return (
    `${WARNING_MARKER}\n` +
    `@${assigneeLogin} — this issue has had no qualifying activity from you ` +
    `for **${warnAfterDays} days**.\n\n` +
    `If you are still working on this, please leave a comment or open a linked ` +
    `pull request within the next **${remaining} day${remaining !== 1 ? "s" : ""}** ` +
    `to retain your assignment. Otherwise, the assignment will be released so ` +
    `another contributor can pick it up.\n\n` +
    `_This message was posted automatically by the inactivity workflow. ` +
    `Maintainers can always manually extend or override assignments._`
  );
}

function buildUnassignComment(assigneeLogin, unassignAfterDays) {
  return (
    `${WARNING_MARKER}\n` +
    `@${assigneeLogin} — you have been unassigned from this issue after ` +
    `**${unassignAfterDays} days** of inactivity. The issue remains open and ` +
    `available for anyone to pick up.\n\n` +
    `If you are still interested in working on this, feel free to comment and ` +
    `a maintainer can reassign it to you.\n\n` +
    `_This action was taken automatically by the inactivity workflow._`
  );
}

// -- Per-assignee state machine ---------------------------------------------

/**
 * Process one (issue, assignee) pair.
 * Returns: "skipped" | "warned" | "unassigned" | "warning-cleared" | "error"
 */
async function processAssignee(octokit, owner, repo, issue, assigneeLogin, config) {
  const { warnAfterDays, unassignAfterDays, dryRun } = config;
  const issueNumber = issue.number;
  const now = new Date();

  console.log(`  [#${issueNumber}] evaluating @${assigneeLogin}`);

  // 1. Fetch timeline (fail-closed).
  const timeline = await fetchTimeline(octokit, owner, repo, issueNumber);
  if (timeline === null) {
    console.log(`  → skip: timeline fetch failed; assuming active`);
    return "error";
  }

  // 2. Compute last human activity date.
  let lastHumanActivity = getLastHumanActivityDate(timeline, assigneeLogin);

  const prActivity = await findLinkedPRActivity(octokit, owner, repo, issueNumber, assigneeLogin);
  if (prActivity.error) {
    console.log(`  → skip: PR search failed; assuming active`);
    return "error";
  }
  if (prActivity.hasActivity && prActivity.date) {
    if (!lastHumanActivity || prActivity.date > lastHumanActivity) {
      lastHumanActivity = prActivity.date;
    }
  }

  // Fallback: use issue creation.
  if (!lastHumanActivity) {
    lastHumanActivity = new Date(issue.created_at);
  }

  const daysSinceActivity = (now - lastHumanActivity) / (1000 * 60 * 60 * 24);
  
  const isWarned = await warningCommentExists(
    octokit, owner, repo, issueNumber, assigneeLogin, lastHumanActivity
  );

  console.log(
    `  last activity: ${lastHumanActivity.toISOString()} (${Math.floor(daysSinceActivity)}d ago), warned=${isWarned}`
  );

  // 3. State machine transitions.

  // A: Recent activity
  if (daysSinceActivity < warnAfterDays) {
    console.log(`  → skip: active`);
    return "skipped";
  }

  // B: Past unassign threshold — remove assignee.
  if (daysSinceActivity >= unassignAfterDays) {
    console.log(`  → unassign: removing @${assigneeLogin}`);
    if (!dryRun) {
      try {
        await octokit.rest.issues.removeAssignees({
          owner, repo, issue_number: issueNumber, assignees: [assigneeLogin],
        });
      } catch (err) {
        console.warn(`  removeAssignees failed: ${err.message}`);
        return "error";
      }

      // Best-effort: post notification (recoverable — assignee is already removed).
      try {
        await octokit.rest.issues.createComment({
          owner, repo, issue_number: issueNumber,
          body: buildUnassignComment(assigneeLogin, unassignAfterDays),
        });
      } catch (err) {
        console.warn(`  notify comment failed: ${err.message}`);
      }
    }
    return "unassigned";
  }

  // C: Between warn and unassign thresholds — warn if not already warned.
  if (!isWarned) {
    console.log(`  → warn: posting warning`);
    if (!dryRun) {
      try {
        await octokit.rest.issues.createComment({
          owner, repo, issue_number: issueNumber,
          body: buildWarningComment(assigneeLogin, warnAfterDays, unassignAfterDays),
        });
      } catch (err) {
        console.warn(`  warning comment failed: ${err.message}`);
        return "error";
      }

    }
    return "warned";
  }

  console.log(`  → skip: already warned; awaiting unassign threshold`);
  return "skipped";
}

// -- Entry point ------------------------------------------------------------

async function run(octokit, context, config = {}) {
  const { warnAfterDays = 15, unassignAfterDays = 30, dryRun = false } = config;
  const { owner, repo } = context.repo;

  console.log(
    `[unassign-inactive] start: warn=${warnAfterDays}d, unassign=${unassignAfterDays}d, dryRun=${dryRun}`
  );

  let issues;
  try {
    issues = await octokit.paginate(octokit.rest.issues.listForRepo, {
      owner, repo, state: "open", labels: ACCEPTED_LABEL, per_page: 100,
    });
  } catch (err) {
    console.error(`[unassign-inactive] listForRepo failed: ${err.message}`);
    process.exitCode = 1;
    return;
  }

  // Filter to actual issues (not PRs) with at least one assignee.
  const assignedIssues = issues.filter(
    (i) => i.assignees && i.assignees.length > 0 && !i.pull_request
  );

  console.log(`[unassign-inactive] ${assignedIssues.length} assigned accepted issue(s)`);

  const stats = { warned: 0, unassigned: 0, cleared: 0, skipped: 0, errors: 0 };

  for (const issue of assignedIssues) {
    console.log(`\nIssue #${issue.number}: "${issue.title}"`);

    for (const assignee of issue.assignees) {
      const action = await processAssignee(
        octokit, owner, repo, issue, assignee.login,
        { warnAfterDays, unassignAfterDays, dryRun }
      );

      switch (action) {
        case "warned":          stats.warned++;     break;
        case "unassigned":      stats.unassigned++; break;
        case "error":           stats.errors++;     break;
        default:                stats.skipped++;    break;
      }
    }
  }

  console.log("\n[unassign-inactive] summary:", stats);
  if (dryRun) console.log("[unassign-inactive] DRY-RUN: no GitHub state was mutated");
}

module.exports = {
  run,
  fetchTimeline,
  getLastHumanActivityDate,
  findLinkedPRActivity,
  warningCommentExists,
  buildWarningComment,
  buildUnassignComment,
  processAssignee,
  WARNING_MARKER,
  WARN_LABEL,
};
