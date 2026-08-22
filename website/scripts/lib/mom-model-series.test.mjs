import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { test } from 'node:test'

const repositoryRoot = resolve(
  dirname(fileURLToPath(import.meta.url)),
  '../../..',
)

test('MoM V1 blog model series follows the packaged CLI catalog', () => {
  const catalog = readFileSync(
    resolve(
      repositoryRoot,
      'src/vllm-sr/cli/model_assets/latest/catalog.yaml',
    ),
    'utf8',
  )
  const blog = readFileSync(
    resolve(
      repositoryRoot,
      'website/blog/2026-07-21-vllm-sr-new-chapter-mom.md',
    ),
    'utf8',
  )

  const cliModels = [...catalog.matchAll(
    /^  - id: (vllm-sr\/mom-v1-[a-z-]+)$/gm,
  )].map(match => match[1])
  const blogModels = [...blog.matchAll(
    /^\| `(vllm-sr\/mom-v1-[a-z-]+)` \|/gm,
  )].map(match => match[1])

  assert.deepEqual(blogModels, cliModels)
  assert.doesNotMatch(blog, /\bmom-v1-(?:light|halu|secu)\b/)
  assert.match(
    blog,
    /alt="The MoM V1 family exposes blend, lite, flash, ultra, and vault as individually versioned model identities"/,
  )
})
