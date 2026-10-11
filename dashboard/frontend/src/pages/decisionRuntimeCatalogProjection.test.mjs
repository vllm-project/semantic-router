import { spawnSync } from 'node:child_process'
import { cpSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'

const root = fileURLToPath(new URL('../../../../', import.meta.url))
const generator = 'dashboard/frontend/scripts/generate-decision-runtime-catalog.py'
const backendProjection = 'src/semantic-router/pkg/modelservice/decision_catalog.generated.json'
const releaseProjection = 'src/model-runtime/vllm_srun/registry/releases.generated.json'
const projection = 'dashboard/frontend/src/pages/decisionRuntimeCatalog.generated.json'
const runtime = 'src/model-runtime/vllm_srun/'
const sources = [
  'registry/tables/common.py',
  'registry/tables/decision1.py',
  'registry/tables/decision2.py',
  'registry/tables/decision3.py',
  'registry/tables/vela2.py',
  'registry/tables/vela1.py',
  'registry/tables/omni.py',
  'families/vela2/request.py',
  'systemone.py',
  'families/decision1/questions.py',
  'families/decision1/family.py',
  'families/decision2/family.py',
  'families/decision3/family.py',
].map((path) => runtime + path)

describe('Decision catalog generation boundary', () => {
  it('checks drift in sources and output without importing the ML runtime or rewriting files', () => {
    const fixture = mkdtempSync(join(tmpdir(), 'decision-catalog-projection-'))
    try {
      for (const file of [
        generator,
        projection,
        backendProjection,
        releaseProjection,
        ...sources,
      ]) {
        mkdirSync(dirname(join(fixture, file)), { recursive: true })
        cpSync(join(root, file), join(fixture, file))
      }
      writeFileSync(
        join(fixture, runtime, '__init__.py'),
        'raise AssertionError("runtime import forbidden")\n',
      )
      const check = () =>
        spawnSync('python3', [join(fixture, generator), '--check'], { encoding: 'utf8' })
      expect(check().status).toBe(0)
      const originalProjection = readFileSync(join(fixture, projection), 'utf8')
      for (const [file, from, to] of [
        [projection, '597103104-never-present', 'unused'],
        [releaseProjection, 'c1e64d4f872cb38bc58502e6888340100bab9d55', 'b'.repeat(40)],
        [
          runtime + 'registry/tables/omni.py',
          '2ff2d66385dbdd661a560ec3e8bcb45a0527d92e',
          'c'.repeat(40),
        ],
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
