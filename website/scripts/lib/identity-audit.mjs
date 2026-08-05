export function findUnresolvedIdentities(entries, { minCommits = 2 } = {}) {
  return entries.filter(entry =>
    !entry.isBot
    && entry.key.startsWith('email:')
    && entry.commits >= minCommits,
  )
}
