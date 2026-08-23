import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { test } from 'node:test'

const repositoryRoot = resolve(
  dirname(fileURLToPath(import.meta.url)),
  '../../..',
)

test('MoM V1 blog model series follows the built-in Recipe family', () => {
  const recipeSource = readFileSync(
    resolve(
      repositoryRoot,
      'config/recipes/built-in/latest/mom-v1/config.yaml',
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

  const publicNameByRecipe = new Map([
    ['balance', 'vllm-sr/mom-v1-blend'],
    ['cost', 'vllm-sr/mom-v1-lite'],
    ['speed', 'vllm-sr/mom-v1-flash'],
    ['accuracy', 'vllm-sr/mom-v1-ultra'],
    ['vault', 'vllm-sr/mom-v1-vault'],
  ])
  const recipeModels = [...recipeSource.matchAll(/^- name: ([a-z-]+)$/gm)]
    .map(match => publicNameByRecipe.get(match[1]))
    .filter(Boolean)
    .sort()
  const blogModels = [...blog.matchAll(
    /^\| `(vllm-sr\/mom-v1-[a-z-]+)` \|/gm,
  )].map(match => match[1]).sort()

  assert.deepEqual(blogModels, recipeModels)
  assert.doesNotMatch(blog, /\bmom-v1-(?:light|halu|secu)\b/)
  assert.match(
    blog,
    /alt="The MoM V1 family exposes blend, lite, flash, ultra, and vault as individually versioned model identities"/,
  )
})
