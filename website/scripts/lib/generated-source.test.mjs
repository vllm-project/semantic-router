import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { test } from 'node:test'
import { fileURLToPath } from 'node:url'
import { assertGeneratedSource, generatedSourceDigest, withGeneratedSource } from './generated-source.mjs'

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../../..')
const helperPath = 'website/scripts/lib/generated-source.mjs'
const snapshots = [
  {
    script: 'website/scripts/generate-contributor-rank.mjs',
    dependency: 'website/scripts/lib/identity-audit.mjs',
    output: 'website/src/data/contributorRank.generated.ts',
  },
  {
    script: 'website/scripts/generate-committer-activity.mjs',
    dependency: 'website/src/data/teamMembers.tsx',
    output: 'website/src/data/committerActivity.generated.ts',
  },
]

test('site lifecycle commands reuse committed community snapshots', () => {
  const packageJson = JSON.parse(readFileSync(join(repoRoot, 'website/package.json'), 'utf8'))
  const scripts = packageJson.scripts

  for (const name of ['start', 'start:zh', 'build', 'build:en', 'build:zh', 'deploy']) {
    assert.doesNotMatch(scripts[name], /npm run (?:contributors:rank|committers:activity)/)
  }
  assert.equal(scripts['contributors:rank'], 'node scripts/generate-contributor-rank.mjs')
  assert.equal(scripts['committers:activity'], 'node scripts/generate-committer-activity.mjs')
})

for (const snapshot of snapshots) {
  test(`${snapshot.script}: check source offline and reject generator/input drift`, (context) => {
    const root = mkdtempSync(join(tmpdir(), 'generated-source-'))
    context.after(() => rmSync(root, { recursive: true, force: true }))
    const sources = [snapshot.script, snapshot.dependency, helperPath]
    for (const source of sources) {
      mkdirSync(dirname(join(root, source)), { recursive: true })
      copyFileSync(join(repoRoot, source), join(root, source))
    }
    const output = join(root, snapshot.output)
    mkdirSync(dirname(output), { recursive: true })
    const digest = generatedSourceDigest(root, sources)
    assert.throws(() => assertGeneratedSource(output, digest, 'regenerate'), /out of date/)
    writeFileSync(output, '// Existing snapshot without provenance\n')
    assert.throws(() => assertGeneratedSource(output, digest, 'regenerate'), /out of date/)
    const content = withGeneratedSource('// Test snapshot\n', digest)
    writeFileSync(output, content)

    // No gh or git executable is available. --check-source must neither refresh
    // live data nor consult git; fallback can only reuse a source-current file.
    const run = (...args) => spawnSync(process.execPath, [join(root, snapshot.script), ...args], {
      cwd: root,
      env: {
        PATH: join(root, 'no-executables'),
        GH_API_MAX_ATTEMPTS: '1',
        GH_API_RETRY_BASE_MS: '0',
        GH_API_PAGE_DELAY_MS: '0',
      },
      encoding: 'utf8',
    })
    let result = run('--check-source')
    assert.equal(result.status, 0, result.stderr)
    result = run()
    assert.equal(result.status, 0, result.stderr)
    assert.match(result.stderr, /Skipping .* refresh/)
    assert.equal(readFileSync(output, 'utf8'), content)

    writeFileSync(output, `${content}// Hand-edited output\n`)
    result = run('--check-source')
    assert.notEqual(result.status, 0)
    assert.match(result.stderr, /Generated snapshot content has changed/)
    result = run()
    assert.notEqual(result.status, 0)
    assert.match(result.stderr, /Generated snapshot content has changed/)
    assert.equal(readFileSync(output, 'utf8'), `${content}// Hand-edited output\n`)
    writeFileSync(output, content)

    for (const source of sources) {
      const path = join(root, source)
      const original = readFileSync(path, 'utf8')
      writeFileSync(path, `${original}\n// Changed local source\n`)
      result = run('--check-source')
      assert.notEqual(result.status, 0)
      assert.match(result.stderr, /Generated snapshot source is out of date/)
      result = run()
      assert.notEqual(result.status, 0)
      assert.match(result.stderr, /Generated snapshot source is out of date/)
      assert.equal(readFileSync(output, 'utf8'), content)
      writeFileSync(path, original)
    }
  })
}
