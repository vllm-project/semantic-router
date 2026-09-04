import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync, existsSync } from 'node:fs'
import { execSync } from 'node:child_process'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const scriptDir = dirname(fileURLToPath(import.meta.url))
const repoRoot = resolve(scriptDir, '..', '..', '..')

const canonicalPath = resolve(repoRoot, 'tools', 'agent', 'skills', 'vllm-sr-install', 'SKILL.md')
const publicPath = resolve(repoRoot, 'website', 'static', 'install', 'agent', 'vllm-sr', 'SKILL.md')
const installationDocPath = resolve(repoRoot, 'website', 'docs', 'installation', 'installation.md')
const syncScript = resolve(repoRoot, 'website', 'scripts', 'sync-public-agent-skill.mjs')

const SKILL_URL = 'https://vllm-sr.ai/install/agent/vllm-sr/SKILL.md'

function readSkill(path) {
  return readFileSync(path, 'utf8')
}

function parseFrontmatter(content) {
  const match = content.match(/^---\n([\s\S]*?)\n---/)
  if (!match) return null
  const block = match[1]
  const fields = {}
  for (const line of block.split('\n')) {
    const kv = line.match(/^(\w+):\s*(.*)$/)
    if (kv) fields[kv[1]] = kv[2]
  }
  return fields
}

test('canonical SKILL.md exists and is non-empty', () => {
  const content = readSkill(canonicalPath)
  assert.ok(content.length > 0, 'SKILL.md should not be empty')
})

test('canonical SKILL.md has valid frontmatter', () => {
  const content = readSkill(canonicalPath)
  const fm = parseFrontmatter(content)
  assert.ok(fm, 'frontmatter block must exist')
  assert.equal(fm.name, 'vllm-sr-install')
  assert.ok(fm.description && fm.description.length > 20, 'description must be meaningful')
  assert.ok(['primary', 'support'].includes(fm.category), `category must be primary or support, got ${fm.category}`)
})

test('canonical SKILL.md has required sections', () => {
  const content = readSkill(canonicalPath)
  const requiredSections = [
    '## Scope',
    '## Safety Boundary',
    '## Environment Discovery',
    '## Supported Installation Paths',
    '## Plan Before Mutation',
    '## Workflow',
    '## Validation',
    '## Unsupported States',
    '## Recovery',
    '## Next Supported Step',
    '## Acceptance',
  ]
  for (const section of requiredSections) {
    assert.ok(content.includes(section), `missing required section: ${section}`)
  }
})

test('canonical SKILL.md references the public URL', () => {
  const content = readSkill(canonicalPath)
  assert.ok(content.includes(SKILL_URL), 'SKILL.md must reference its own public URL')
})

test('canonical SKILL.md uses maintained installer contract', () => {
  const content = readSkill(canonicalPath)
  assert.ok(content.includes('https://vllm-sr.ai/install.sh'), 'must reference the maintained installer')
  assert.ok(content.includes('--mode cli --runtime skip --no-launch'), 'must recommend agent-safe flags')
})

test('canonical SKILL.md does not reference unsupported runtime values', () => {
  const content = readSkill(canonicalPath)
  // --runtime podman is not yet in main; only auto|docker|skip are valid
  const runtimeTableMatch = content.match(/\| `--runtime` \| (.+?) \|/)
  if (runtimeTableMatch) {
    const cell = runtimeTableMatch[1]
    assert.ok(
      cell.includes('auto') && cell.includes('docker') && cell.includes('skip'),
      'runtime values must match current installer contract',
    )
    assert.ok(!cell.includes('podman'), 'must not list podman until the upstream PR lands')
  }
})

test('canonical SKILL.md separates runtime state from active deployment', () => {
  const content = readSkill(canonicalPath)
  assert.ok(
    content.includes('Existing local runtime state'),
    'must have a runtime state subsection distinct from active deployment',
  )
  assert.ok(
    /runtime\.env[^]+does not by itself prove/i.test(content),
    'must state that runtime.env alone does not prove an active deployment',
  )
  assert.ok(
    content.includes('### Active deployment'),
    'must have an active deployment subsection',
  )
})

test('canonical SKILL.md does not define a pip installation path', () => {
  const content = readSkill(canonicalPath)
  assert.ok(
    !/Alternative path.*pip/i.test(content),
    'must not define a pip alternative path; the public journey uses the maintained installer only',
  )
  assert.ok(
    !content.includes('python -m pip install'),
    'must not instruct pip install in the public skill body',
  )
})

test('canonical SKILL.md declares out-of-scope items', () => {
  const content = readSkill(canonicalPath)
  const outOfScope = ['Generating or applying Router configuration', 'evaluation', 'activation', 'rollback']
  for (const item of outOfScope) {
    assert.ok(content.toLowerCase().includes(item.toLowerCase()), `must declare out-of-scope: ${item}`)
  }
})

test('sync script produces a byte-for-byte identical public artifact', () => {
  // The static copy is gitignored (built at build time, same as install.sh).
  // Run the sync script to generate it, then verify parity.
  execSync(`node "${syncScript}"`, { cwd: repoRoot })
  assert.ok(existsSync(publicPath), 'sync script must produce the static artifact')
  const canonical = readSkill(canonicalPath)
  const artifact = readSkill(publicPath)
  assert.equal(artifact, canonical, 'public SKILL.md must be identical to canonical source after sync')
})

test('static artifact is gitignored (built, not committed)', () => {
  // Mirrors the install.sh pattern: canonical is committed, static copy is generated.
  const result = execSync('git check-ignore website/static/install/agent/vllm-sr/SKILL.md', {
    cwd: repoRoot,
    encoding: 'utf8',
  }).trim()
  assert.ok(result.includes('website/static/install/agent/'), 'static artifact must be gitignored')
})

test('installation page contains the public SKILL URL', () => {
  const doc = readSkill(installationDocPath)
  assert.ok(doc.includes(SKILL_URL), 'installation.md must link to the public skill URL')
})

test('installation page Agent prompt contains plan-first semantics', () => {
  const doc = readSkill(installationDocPath)
  assert.ok(doc.includes('Show me the plan first'), 'Agent prompt must contain plan-first language')
})

test('installation page Agent prompt contains approval boundary', () => {
  const doc = readSkill(installationDocPath)
  assert.ok(
    doc.includes('ask before changing an existing configuration or deployment'),
    'Agent prompt must contain the approval boundary',
  )
})
