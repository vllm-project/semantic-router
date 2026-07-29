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
 * `github:*`-keyed entry when the entry's commit author name equals the
 * GitHub login of exactly one non-bot `github:*` entry.
 *
 * This covers commits whose GitHub login resolution failed at generation time
 * (e.g. a deleted account whose commits resolve to `ghost`), which otherwise
 * splits one contributor into two ranked leaderboard entries. Broader
 * email/name-evidence matching was deliberately dropped: git author names are
 * user-controlled and collide (two authors committing as `root`), so only the
 * name-equals-login signal is used.
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

  const targetByLogin = new Map()

  for (const entry of merged.values()) {
    if (!entry.key?.startsWith('github:') || entry.isBot) {
      continue
    }

    const login = normalizeValue(entry.login)

    if (login) {
      indexEvidence(targetByLogin, [login], entry)
    }
  }

  for (const [key, entry] of merged) {
    if (!key.startsWith('email:') || entry.isBot) {
      continue
    }

    const target = lookupEvidence(targetByLogin, entry.names)

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
