import assert from 'node:assert/strict'
import test from 'node:test'
import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const promptSourcePath = path.resolve(__dirname, '../src/components/ChatComponentTypes.ts')

const promptExpectations = [
  'You are HireClaw, an elite recruiter and talent partner',
  'present a shortlist of 2-3 candidates by default',
  'wait for explicit user approval',
  'plausible English first names',
  'Worker A, Analyst Bot, Operator-1, Helper, Assistant',
  'respect it unless they explicitly ask for alternatives or a rename',
]

test('HireClaw prompt keeps the recruiter-style hiring contract', async () => {
  const source = await readFile(promptSourcePath, 'utf8')

  for (const expectation of promptExpectations) {
    assert.match(
      source,
      new RegExp(expectation.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')),
      `expected prompt source to include: ${expectation}`,
    )
  }
})
