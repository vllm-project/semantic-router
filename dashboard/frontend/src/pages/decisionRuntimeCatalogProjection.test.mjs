import { spawnSync } from 'node:child_process'
import { cpSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'

const root = fileURLToPath(new URL('../../../../', import.meta.url))
const generator = 'dashboard/frontend/scripts/generate-decision-runtime-catalog.py'
const projection = 'dashboard/frontend/src/pages/decisionRuntimeCatalog.generated.json'
const runtime = 'src/model-runtime/vllm_srun/'
const sources = [
  'registry/tables/common.py',
  'registry/tables/decision1.py',
  'registry/tables/decision2.py',
  'systemone.py',
  'families/decision1/questions.py',
  'families/decision1/family.py',
  'families/decision2/family.py',
].map((path) => runtime + path)

describe('Decision catalog generation boundary', () => {
  it('checks drift in sources and output without importing the ML runtime or rewriting files', () => {
    const fixture = mkdtempSync(join(tmpdir(), 'decision-catalog-projection-'))
    try {
      for (const file of [generator, projection, ...sources]) {
        mkdirSync(dirname(join(fixture, file)), { recursive: true })
        cpSync(join(root, file), join(fixture, file))
      }
      const check = () =>
        spawnSync('python3', [join(fixture, generator), '--check'], { encoding: 'utf8' })
      expect(check().status).toBe(0)
      const originalProjection = readFileSync(join(fixture, projection), 'utf8')
      for (const [file, from, to] of [
        [projection, '597103104-never-present', 'unused'],
        [
          runtime + 'registry/tables/decision2.py',
          'cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764',
          'a'.repeat(40),
        ],
        [runtime + 'registry/tables/common.py', '"vllm-sr"', '"new-provider"'],
        [
          runtime + 'systemone.py',
          '("choice", "noul", "score")',
          '("choice", "noul", "score", "span")',
        ],
      ]) {
        const path = join(fixture, file)
        const before = readFileSync(path, 'utf8')
        writeFileSync(path, file === projection ? '{"models":[]}' : before.replace(from, to))
        expect(check().status, file).not.toBe(0)
        if (file !== projection)
          expect(readFileSync(join(fixture, projection), 'utf8')).toBe(originalProjection)
        writeFileSync(path, before)
      }
      rmSync(join(fixture, projection))
      expect(check().status).not.toBe(0)
      expect(spawnSync('python3', [join(fixture, generator)], { encoding: 'utf8' }).status).toBe(0)
      expect(check().status).toBe(0)
    } finally {
      rmSync(fixture, { recursive: true, force: true })
    }
  })
})
