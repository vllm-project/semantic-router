function normalizeValue(value) {
  return String(value ?? '').trim().toLowerCase()
}

function pickEarlierDate(left, right) {
  if (!left) {
    return right
  }

  if (!right) {
    return left
  }

  return left < right ? left : right
}

function pickLaterDate(left, right) {
  if (!left) {
    return right
  }

  if (!right) {
    return left
  }

  return left > right ? left : right
}

function normalizedSet(values) {
  const normalized = new Set()

  for (const value of values ?? []) {
    const candidate = normalizeValue(value)

    if (candidate) {
      normalized.add(candidate)
    }
  }

  return normalized
}

const AMBIGUOUS = Symbol('ambiguous evidence')

function indexEvidence(index, evidence, entry) {
  for (const value of evidence) {
    const existing = index.get(value)

    if (existing === undefined) {
      index.set(value, entry)
    }
    else if (existing !== entry) {
      index.set(value, AMBIGUOUS)
    }
  }
}

function lookupEvidence(index, evidence) {
  for (const value of evidence) {
    const target = index.get(value)

    if (target && target !== AMBIGUOUS) {
      return target
    }
  }

  return null
}

/**
 * Merges leftover `email:*`-keyed contributor entries into a matching
 * `github:*`-keyed entry when the commit author evidence the script already
 * collected (author emails, author names) links them to the same human.
 *
 * This covers commits whose GitHub login resolution failed at generation time
 * (e.g. the author email was not yet linked to the GitHub account), which
 * otherwise splits one contributor into two ranked leaderboard entries.
 *
 * @param {Map<string, object>} byContributor entries keyed by identity key,
 *   each carrying `emails`/`names` evidence sets plus commit stats.
 * @returns {Map<string, object>} a new map with email-keyed entries folded
 *   into their matching github-keyed entry (commits summed, date range
 *   widened, github key kept). The input map is not mutated.
 */
export function mergeEmailKeyedContributors(byContributor) {
  const merged = new Map()

  for (const [key, entry] of byContributor) {
    merged.set(key, {
      ...entry,
      emails: normalizedSet(entry.emails),
      names: normalizedSet(entry.names),
    })
  }

  const targetByEmail = new Map()
  const targetByName = new Map()
  const targetByLogin = new Map()

  for (const entry of merged.values()) {
    if (!entry.key?.startsWith('github:') || entry.isBot) {
      continue
    }

    indexEvidence(targetByEmail, entry.emails, entry)
    indexEvidence(targetByName, entry.names, entry)

    const login = normalizeValue(entry.login)

    if (login) {
      indexEvidence(targetByLogin, [login], entry)
    }
  }

  for (const [key, entry] of merged) {
    if (!key.startsWith('email:') || entry.isBot) {
      continue
    }

    const target = lookupEvidence(targetByEmail, entry.emails)
      ?? lookupEvidence(targetByName, entry.names)
      ?? lookupEvidence(targetByLogin, entry.names)

    if (!target) {
      continue
    }

    target.commits += entry.commits
    target.firstCommitDate = pickEarlierDate(target.firstCommitDate, entry.firstCommitDate)
    target.latestCommitDate = pickLaterDate(target.latestCommitDate, entry.latestCommitDate)

    for (const email of entry.emails) {
      target.emails.add(email)
    }

    for (const name of entry.names) {
      target.names.add(name)
    }

    merged.delete(key)
  }

  return merged
}
