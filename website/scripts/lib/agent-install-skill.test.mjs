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

test('Workflow stops on command -v, not only on --version success', () => {
  const content = readSkill(canonicalPath)
  // Extract the ## Workflow section body (up to the next ## heading).
  const workflowMatch = content.match(/## Workflow\n([\s\S]*?)\n## /)
  assert.ok(workflowMatch, '## Workflow section must exist')
  const workflow = workflowMatch[1]

  // The Workflow must gate "stop" on `command -v vllm-sr` finding the launcher,
  // not only on `vllm-sr --version` succeeding. This prevents a broken or
  // stale launcher from being silently overwritten.
  assert.ok(
    workflow.includes('`command -v vllm-sr`'),
    'Workflow must reference command -v vllm-sr as the existence gate',
  )
  assert.ok(
    /command -v vllm-sr[\s\S]*stop/i.test(workflow),
    'Workflow must stop when command -v finds the launcher',
  )
  assert.ok(
    /version[\s\S]*diagnostic/i.test(workflow),
    'Workflow must treat --version as diagnostics, not as the gate for continuing',
  )

  // The old regression: "If vllm-sr --version succeeds, report the version
  // and stop" implied that a broken launcher allows installation to proceed.
  // Ensure that exact phrasing is gone from Workflow.
  assert.ok(
    !/If `vllm-sr --version` succeeds[\s\S]*stop/.test(workflow),
    'Workflow must not gate stop on --version success alone (the old broken-gate phrasing)',
  )
})

test('Workflow checks default absolute paths, not only PATH', () => {
  const content = readSkill(canonicalPath)
  const workflowMatch = content.match(/## Workflow\n([\s\S]*?)\n## /)
  assert.ok(workflowMatch, '## Workflow section must exist')
  const workflow = workflowMatch[1]

  // Non-interactive agent shells often omit ~/.local/bin from PATH, so
  // command -v alone is insufficient. The Workflow must also check the
  // default absolute launcher and install root.
  assert.ok(
    workflow.includes('~/.local/bin/vllm-sr'),
    'Workflow must check the default absolute launcher ~/.local/bin/vllm-sr',
  )
  assert.ok(
    workflow.includes('~/.local/share/vllm-sr'),
    'Workflow must check the default install root ~/.local/share/vllm-sr',
  )
  assert.ok(
    /any signal/i.test(workflow),
    'Workflow must treat any of the signals as an existing installation',
  )
})

test('Workflow detects stale launchers and dangling symlinks, not only executable files', () => {
  const content = readSkill(canonicalPath)
  const workflowMatch = content.match(/## Workflow\n([\s\S]*?)\n## /)
  assert.ok(workflowMatch, '## Workflow section must exist')
  const workflow = workflowMatch[1]

  // test -x misses stale non-executable files and dangling symlinks that
  // still occupy the launcher path. The Workflow must use test -e (regular
  // file exists) and test -L (symlink exists regardless of target).
  assert.ok(
    /test -e .*vllm-sr/.test(workflow),
    'Workflow must use test -e to catch non-executable stale launcher files',
  )
  assert.ok(
    /test -L .*vllm-sr/.test(workflow),
    'Workflow must use test -L to catch dangling symlinks at the launcher path',
  )
  assert.ok(
    !/test -x .*vllm-sr/.test(workflow),
    'Workflow must not rely on test -x alone for launcher detection',
  )
})

test('Workflow detects installer override env vars', () => {
  const content = readSkill(canonicalPath)
  const workflowMatch = content.match(/## Workflow\n([\s\S]*?)\n## /)
  assert.ok(workflowMatch, '## Workflow section must exist')
  const workflow = workflowMatch[1]

  // install.sh reads VLLM_SR_INSTALL_ROOT, VLLM_SR_BIN_DIR, and
  // VLLM_SR_PIP_SPEC from the environment. When set, discovery must check
  // the override path too, not only the defaults.
  assert.ok(
    workflow.includes('VLLM_SR_BIN_DIR'),
    'Workflow must check the VLLM_SR_BIN_DIR override path when set',
  )
  assert.ok(
    workflow.includes('VLLM_SR_INSTALL_ROOT'),
    'Workflow must check the VLLM_SR_INSTALL_ROOT override path when set',
  )
  assert.ok(
    workflow.includes('VLLM_SR_PIP_SPEC'),
    'Workflow must handle the VLLM_SR_PIP_SPEC package override',
  )
})

test('VLLM_SR_PIP_SPEC presence stops the flow and its value is never printed', () => {
  const content = readSkill(canonicalPath)

  // The installer uses any non-empty VLLM_SR_PIP_SPEC as the pip package
  // spec, so its presence means an alternate package could be installed
  // instead of the official CLI. The workflow must treat presence as a
  // stop condition.
  const workflowMatch = content.match(/## Workflow\n([\s\S]*?)\n## /)
  assert.ok(workflowMatch, '## Workflow section must exist')
  const workflow = workflowMatch[1]
  assert.ok(
    /VLLM_SR_PIP_SPEC[\s\S]*stop/i.test(workflow),
    'Workflow must stop when VLLM_SR_PIP_SPEC is set',
  )
  assert.ok(
    /alternate package/i.test(workflow),
    'Workflow must explain that VLLM_SR_PIP_SPEC installs an alternate package',
  )

  // The value may embed credentials such as private index tokens, so the
  // skill must never instruct printing it — presence-only detection only.
  assert.ok(
    !content.includes('echo "${VLLM_SR_PIP_SPEC'),
    'Skill must not instruct echoing the VLLM_SR_PIP_SPEC value',
  )
  assert.ok(
    !content.includes('printf "${VLLM_SR_PIP_SPEC'),
    'Skill must not instruct printing the VLLM_SR_PIP_SPEC value',
  )
  assert.ok(
    /test -n "\$\{VLLM_SR_PIP_SPEC:-\}"/.test(content),
    'Detection must use a presence-only check for VLLM_SR_PIP_SPEC',
  )

  // Defense in depth: the install step must clear the installer's entire
  // documented VLLM_SR_* override surface so no inherited value can change
  // what gets installed or how. The unset must apply to the installer's
  // shell, not just to curl.
  const installBlock = workflow.match(/```bash\n([\s\S]*?)```/)
  assert.ok(installBlock, 'Install step must include the install bash block')
  const overrideSurface = [
    'VLLM_SR_INSTALL_MODE',
    'VLLM_SR_RUNTIME',
    'VLLM_SR_INSTALL_ROOT',
    'VLLM_SR_BIN_DIR',
    'VLLM_SR_INSTALL_CHANNEL',
    'VLLM_SR_PIP_SPEC',
    'VLLM_SR_PYTHON',
    'VLLM_SR_INSTALL_PLATFORM',
    'VLLM_SR_INSTALL_AUTO_LAUNCH',
  ]
  for (const overrideVar of overrideSurface) {
    assert.ok(
      installBlock[1].includes(overrideVar),
      `Install step must unset ${overrideVar}`,
    )
  }
  assert.ok(
    /unset /.test(installBlock[1]),
    'Install step must unset the overrides in the installer shell',
  )
  assert.ok(
    !workflow.includes('env -u VLLM_SR'),
    'Install step must not use env -u on the pipeline (it would leave the overrides set for the installer shell)',
  )
})

test('Plan discloses shell completion setup as a side effect', () => {
  const content = readSkill(canonicalPath)
  const planMatch = content.match(/## Plan Before Mutation\n([\s\S]*?)\n## /)
  assert.ok(planMatch, '## Plan Before Mutation section must exist')
  const plan = planMatch[1]

  // install.sh unconditionally runs `vllm-sr completion install`, which may
  // edit ~/.bashrc, ~/.zshrc, or equivalent shell rc files. The plan must
  // disclose this side effect so the user can approve it explicitly.
  assert.ok(
    /completion/i.test(plan),
    'Plan must disclose that the installer sets up shell completions',
  )
  assert.ok(
    /shell rc|bashrc|zshrc/i.test(plan),
    'Plan must mention shell rc files as the thing completion edits',
  )
})

test('Existing CLI installation section detects all default signals and override paths', () => {
  const content = readSkill(canonicalPath)
  const sectionMatch = content.match(/### Existing CLI installation\n([\s\S]*?)\n### /)
  assert.ok(sectionMatch, '### Existing CLI installation section must exist')
  const section = sectionMatch[1]

  assert.ok(
    section.includes('`command -v vllm-sr`'),
    'Existing CLI installation must gate on command -v vllm-sr',
  )
  assert.ok(
    section.includes('~/.local/bin/vllm-sr'),
    'Existing CLI installation must check the default absolute launcher',
  )
  assert.ok(
    section.includes('~/.local/share/vllm-sr'),
    'Existing CLI installation must check the default install root',
  )
  assert.ok(
    /test -e .*vllm-sr/.test(section),
    'Existing CLI installation must use test -e to catch stale launchers',
  )
  assert.ok(
    /test -L .*vllm-sr/.test(section),
    'Existing CLI installation must use test -L to catch dangling symlinks',
  )
  assert.ok(
    section.includes('VLLM_SR_BIN_DIR'),
    'Existing CLI installation must check the VLLM_SR_BIN_DIR override path',
  )
  assert.ok(
    section.includes('VLLM_SR_INSTALL_ROOT'),
    'Existing CLI installation must check the VLLM_SR_INSTALL_ROOT override path',
  )
  assert.ok(
    /all[\s\S]*stop/i.test(section),
    'Existing CLI installation must stop in all detection cases',
  )
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
