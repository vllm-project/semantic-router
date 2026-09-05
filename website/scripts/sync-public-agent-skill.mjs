import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const scriptDir = dirname(fileURLToPath(import.meta.url))
const repoRoot = resolve(scriptDir, '..', '..')
const sourcePath = resolve(repoRoot, 'tools', 'agent', 'skills', 'vllm-sr-install', 'SKILL.md')
const destinationPath = resolve(repoRoot, 'website', 'static', 'install', 'agent', 'vllm-sr', 'SKILL.md')

const sourceContent = readFileSync(sourcePath, 'utf8')
let destinationContent = ''

try {
  destinationContent = readFileSync(destinationPath, 'utf8')
} catch {
  destinationContent = ''
}

if (destinationContent !== sourceContent) {
  mkdirSync(dirname(destinationPath), { recursive: true })
  writeFileSync(destinationPath, sourceContent)
}
