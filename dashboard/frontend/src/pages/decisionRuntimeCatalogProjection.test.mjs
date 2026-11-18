import { spawnSync } from 'node:child_process'
import { cpSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'

const root = fileURLToPath(new URL('../../../../', import.meta.url))
const generator = 'dashboard/frontend/scripts/generate-decision-runtime-catalog.py'
const backendProjection = 'src/semantic-router/pkg/config/decision_catalog.generated.json'
const releaseProjection = 'src/model-runtime/vllm_srun/registry/releases.generated.json'
const projection = 'dashboard/frontend/src/pages/decisionRuntimeCatalog.generated.json'
const editorial = 'tools/docs/supported_models.yaml'
const runtime = 'src/model-runtime/vllm_srun/'
const sources = [
  'registry/tables/common.py',
  'registry/tables/decision1.py',
  'registry/tables/decision2.py',
  'registry/tables/decision3.py',
  'registry/tables/vela2.py',
  'registry/tables/vela3.py',
  'registry/tables/vela3_d3.py',
  'registry/tables/vela3_pico.py',
  'registry/tables/vela1.py',
  'registry/tables/omni.py',
  'families/vela2/request.py',
  'families/vela2/family.py',
  'families/vela2/package.py',
  'families/vela3_shared.py',
  'systemone.py',
  'families/decision1/questions.py',
  'families/decision1/family.py',
  'families/decision2/family.py',
  'families/decision3/family.py',
  'families/vela3/family.py',
  'families/vela3_d3/family.py',
  'families/vela3_pico/family.py',
].map((path) => runtime + path)

function copyFixture() {
  const fixture = mkdtempSync(join(tmpdir(), 'decision-catalog-projection-'))
  for (const file of [
    generator,
    projection,
    backendProjection,
    releaseProjection,
    editorial,
    ...sources,
  ]) {
    mkdirSync(dirname(join(fixture, file)), { recursive: true })
    cpSync(join(root, file), join(fixture, file))
  }
  writeFileSync(
    join(fixture, runtime, '__init__.py'),
    'raise AssertionError("runtime import forbidden")\n',
  )
  return fixture
}

describe('Decision catalog generation boundary', () => {
  // Twelve python3 runs of the generator take longer than vitest's 5 s default on CI runners.
  it('checks drift in sources and output without importing the ML runtime or rewriting files', () => {
    const fixture = copyFixture()
    try {
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
        [runtime + 'registry/tables/vela3_d3.py', '4_792_108_035', '4_792_108_036'],
        [runtime + 'registry/tables/common.py', '"vllm-sr"', '"new-provider"'],
        [
          runtime + 'systemone.py',
          '("choice", "noul", "score")',
          '("choice", "noul", "score", "span")',
        ],
        [
          runtime + 'families/vela3_shared.py',
          'SPAN_HEADS = ("router",)',
          'SPAN_HEADS = ("broad",)',
        ],
        [editorial, 'The default decision model.', 'The decision model.'],
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
  }, 60_000)

  it('lists announced models as pending until a registry table pins the same repository', () => {
    const fixture = copyFixture()
    try {
      const generated = () => JSON.parse(readFileSync(join(fixture, projection), 'utf8')).models
      const pending = generated().filter(({ pending }) => pending)
      expect(pending.map(({ id, family, pending: link }) => [id, family, link])).toEqual([
        [
          'vllm-sr/vela3.1-fast',
          'vela3',
          'https://github.com/vllm-project/semantic-router/pull/6391',
        ],
        [
          'vllm-sr/vela3.1-pico-150m',
          'vela3_pico',
          'https://github.com/vllm-project/semantic-router/pull/6390',
        ],
        [
          'vllm-sr/vela3.1-pico-380m',
          'vela3_pico',
          'https://github.com/vllm-project/semantic-router/pull/6390',
        ],
      ])
      for (const entry of pending) expect(entry).toMatchObject({ revision: '', questionTypes: [] })
      expect(pending[1]).toMatchObject({
        parameters: 153_207_299,
        minMemoryGiB: 2,
        backbone: 'ultrabert',
      })
      expect(pending[0]).toMatchObject({ parameters: null, minMemoryGiB: null })
      const table = join(fixture, runtime, 'registry/tables/vela3.py')
      const release = `    _vela(
        "vela3.1-fast",
        "${'d'.repeat(40)}",
        "${'e'.repeat(64)}",
        380_961_027,
        "ultrabert",
        3,
        {},
    ),
)
MODELS = with_recorded(MODELS, "golden_answers_vela3.json"`
      writeFileSync(
        table,
        readFileSync(table, 'utf8').replace(
          ')\nMODELS = with_recorded(MODELS, "golden_answers_vela3.json"',
          release,
        ),
      )
      expect(spawnSync('python3', [join(fixture, generator)], { encoding: 'utf8' }).status).toBe(0)
      const models = generated()
      const listed = models.filter(({ id }) => id === 'vllm-sr/vela3.1-fast')
      expect(listed).toEqual([
        expect.objectContaining({
          name: 'vela3.1-fast',
          family: 'vela3',
          familyLabel: 'Vela 3.0',
          revision: 'd'.repeat(40),
          backbone: 'ultrabert',
          minMemoryGiB: 3,
          parameters: 380_961_027,
          questionTypes: ['choice', 'noul', 'span'],
          modalities: ['text'],
          pending: '',
        }),
      ])
      expect(models.filter(({ pending }) => pending).map(({ id }) => id)).toEqual([
        'vllm-sr/vela3.1-pico-150m',
        'vllm-sr/vela3.1-pico-380m',
      ])
      expect(models.find(({ id }) => id === 'vllm-sr/vela3-fast-vision')).toMatchObject({
        family: 'vela3',
        questionTypes: [],
      })
    } finally {
      rmSync(fixture, { recursive: true, force: true })
    }
  }, 30_000)
})
