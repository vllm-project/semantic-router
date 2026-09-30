import { createHash } from 'node:crypto'
import { existsSync, readFileSync } from 'node:fs'
import { resolve } from 'node:path'

// Live GitHub statistics cannot be reproduced from repository files alone.
// This digest checks the local generator and inputs that produced a snapshot,
// independently of when the remote statistics were last refreshed.
export function generatedSourceDigest(repoRoot, sourcePaths) {
  const hash = createHash('sha256')
  for (const path of [...new Set(sourcePaths)].sort()) {
    hash.update(`${path}\0`)
    hash.update(readFileSync(resolve(repoRoot, path)))
    hash.update('\0')
  }
  return hash.digest('hex')
}

export function withGeneratedSource(content, digest) {
  const bodyDigest = createHash('sha256').update(content).digest('hex')
  return `// Source SHA-256: ${digest}\n// Body SHA-256: ${bodyDigest}\n${content}`
}

export function assertGeneratedSource(outputPath, digest, command) {
  const content = existsSync(outputPath) ? readFileSync(outputPath, 'utf8') : ''
  const provenance = content.match(/^\/\/ Source SHA-256: ([a-f0-9]{64})\n\/\/ Body SHA-256: ([a-f0-9]{64})\n/)
  if (provenance?.[1] !== digest) {
    throw new Error(`Generated snapshot source is out of date: ${outputPath}. Run \`${command}\` in website/.`)
  }
  const body = content.slice(provenance[0].length)
  const bodyDigest = createHash('sha256').update(body).digest('hex')
  if (provenance[2] !== bodyDigest) {
    throw new Error(`Generated snapshot content has changed: ${outputPath}. Run \`${command}\` in website/.`)
  }
}
