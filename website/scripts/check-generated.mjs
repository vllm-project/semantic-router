import { spawnSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../..')
const localPython = resolve(repoRoot, '.venv-agent/bin/python')
const python = process.env.VLLM_SR_DOCS_PYTHON
  || (existsSync(localPython) ? localPython : 'python3')

// Check committed artifacts before any build-time generators can run.
// Keep using the authoritative generators, rather than a second JS compiler.
const commands = [
  [python, 'tools/catalog/generate_model_catalog.py', '--check'],
  [python, 'tools/docs/generate_cli_reference.py', '--check'],
  [python, 'tools/agent/scripts/sync_public_skill.py', '--check'],
  [process.execPath, 'website/scripts/generate-configuration-catalog.mjs', '--check'],
  [process.execPath, 'website/scripts/generate-contributor-rank.mjs', '--check-source'],
  [process.execPath, 'website/scripts/generate-committer-activity.mjs', '--check-source'],
]

for (const [command, ...args] of commands) {
  const result = spawnSync(command, args, { cwd: repoRoot, stdio: 'inherit' })
  if (result.error || result.status !== 0) {
    console.error(
      'Generated reference check failed. Install Python 3.10+ dependencies with '
      + '`python3 -m pip install -r website/requirements.txt`, then regenerate '
      + 'the reported artifact and commit it with its source.',
    )
    if (result.error) console.error(result.error.message)
    process.exit(result.status || 1)
  }
}
