// Testable extraction of unassign logic for unit tests.
// This module exposes a function 

async function run({ github, context, daysWarning = 15, daysUnassign = 30, now = new Date() }) {
  const owner = context.repo.owner;
  const repo = context.repo.repo;
  const millis = d => d * 24 * 60 * 60 * 1000;

  async function fetchTimeline(issue_number) {
    try {
      const timeline = await github.paginate(github.rest.issues.listEventsForTimeline, { owner, repo, issue_number, per_page: 100 });
      return { timeline };
    } catch (err) {
      return { error: err };
    }
  }

  function lastHumanActivityForAssignee(assignee, timeline) {
    let last = null;
    for (const ev of timeline || []) {
      if (!ev.actor || !ev.created_at) continue;
      const actor = ev.actor;
      if (actor.type !== 'User') continue;
      if (actor.login && actor.login.toLowerCase() === assignee.login.toLowerCase()) {
        const t = new Date(ev.created_at);
        if (!last || t > last) last = t;
      }
      if (ev.event === 'commented' && ev.actor && ev.actor.login && ev.actor.login.toLowerCase() === assignee.login.toLowerCase()) {
        const t = new Date(ev.created_at);
        if (!last || t > last) last = t;
      }
    }
    return last;
  }

  async function hasRecentLinkedPRActivity(issue_number) {
    const { timeline, error } = await fetchTimeline(issue_number);
    if (error) return { error };
    for (const ev of timeline || []) {
      if (ev.event === 'cross-referenced' && ev.source && ev.source.issue && ev.source.issue.pull_request) {
        const prNum = ev.source.issue.number;
        try {
          const { data: pr } = await github.rest.pulls.get({ owner, repo, pull_number: prNum });
          const prUpdated = new Date(pr.updated_at || pr.created_at);
          if (now - prUpdated < millis(daysWarning)) return { recent: true };
        } catch (err) {
          return { error: err };
        }
      }
    }
    return { recent: false };
  }

  const results = [];

  const issuesIterator = github.paginate.iterator(github.rest.issues.listForRepo, { owner, repo, state: 'open', per_page: 100 });
  for await (const { data: issues } of issuesIterator) {
    for (const issue of issues) {
      if (issue.pull_request) continue;
      const { timeline, error: timelineErr } = await fetchTimeline(issue.number);
      if (timelineErr) { results.push({ issue: issue.number, skipped: true, reason: 'timeline_error' }); continue; }
      const assignees = issue.assignees || [];
      if (!assignees || assignees.length === 0) { results.push({ issue: issue.number, skipped: true, reason: 'no_assignees' }); continue; }
      let anyAssigneeActive = false;
      let prCheckError = false;
      for (const a of assignees) {
        const lastHuman = lastHumanActivityForAssignee(a, timeline);
        if (lastHuman && (now - lastHuman < millis(daysWarning))) { anyAssigneeActive = true; break; }
        const { recent, error } = await hasRecentLinkedPRActivity(issue.number);
        if (error) { prCheckError = true; break; }
        if (recent) { anyAssigneeActive = true; break; }
      }
      if (prCheckError) { results.push({ issue: issue.number, skipped: true, reason: 'pr_check_error' }); continue; }
      if (anyAssigneeActive) { results.push({ issue: issue.number, action: 'keep' }); continue; }

      let mostRecentHuman = null;
      for (const a of assignees) {
        const t = lastHumanActivityForAssignee(a, timeline || []);
        if (t && (!mostRecentHuman || t > mostRecentHuman)) mostRecentHuman = t;
      }
      const refTime = mostRecentHuman || new Date(issue.created_at);
      const ageSinceHuman = now - refTime;

      if (ageSinceHuman >= millis(daysUnassign)) {
        // unassign
        try {
          await github.rest.issues.removeAssignees({ owner, repo, issue_number: issue.number, assignees: assignees.map(a => a.login) });
          try { await github.rest.issues.createComment({ owner, repo, issue_number: issue.number, body: 'Unassigned due to inactivity' }); } catch (e) {}
          try { await github.rest.issues.addLabels({ owner, repo, issue_number: issue.number, labels: ['assignment-inactive-unassigned'] }); } catch (e) {}
          results.push({ issue: issue.number, action: 'unassigned' });
        } catch (e) {
          results.push({ issue: issue.number, skipped: true, reason: 'unassign_failed' });
        }
      } else if (ageSinceHuman >= millis(daysWarning)) {
        try { await github.rest.issues.createComment({ owner, repo, issue_number: issue.number, body: 'Reminder: inactivity' }); } catch (e) { results.push({ issue: issue.number, skipped: true, reason: 'comment_failed' }); continue; }
        try { await github.rest.issues.addLabels({ owner, repo, issue_number: issue.number, labels: ['assignment-inactive-warning'] }); } catch (e) {}
        results.push({ issue: issue.number, action: 'warned' });
      } else {
        results.push({ issue: issue.number, action: 'noop' });
      }
    }
  }

  return results;
}

module.exports = { run };

