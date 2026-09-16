import { readFileSync } from 'node:fs'

import { describe, expect, it } from 'vitest'

import { isEvaluationMethodDescriptor } from './evaluationMethodReportContract'

interface MethodConformanceCase {
  id: string
  expected_valid: boolean
  remove_fields: string[]
  overrides: Record<string, unknown>
}

interface MethodConformanceCorpus {
  schema_version: string
  method_contract_version: string
  base_descriptor: Record<string, unknown>
  cases: MethodConformanceCase[]
}

const corpus = JSON.parse(
  readFileSync(
    new URL(
      '../../../../src/vllm-sr/tests/fixtures/evaluation_method_contract_v2_conformance.v1.json',
      import.meta.url,
    ),
    'utf8',
  ),
) as MethodConformanceCorpus

function decodeJsonPointer(pointer: string): string[] {
  if (pointer.length === 0 || !pointer.startsWith('/')) {
    throw new Error(`JSON Pointer must identify a descriptor field: ${JSON.stringify(pointer)}`)
  }
  return pointer
    .slice(1)
    .split('/')
    .map((encodedToken) => {
      for (let index = 0; index < encodedToken.length; index += 1) {
        if (encodedToken[index] !== '~') continue
        if (
          index + 1 >= encodedToken.length ||
          !['0', '1'].includes(encodedToken[index + 1])
        ) {
          throw new Error(`JSON Pointer has an invalid escape: ${JSON.stringify(pointer)}`)
        }
        index += 1
      }
      return encodedToken.replace(/~1/g, '/').replace(/~0/g, '~')
    })
}

function jsonArrayIndex(token: string, length: number, pointer: string): number {
  if (!/^(0|[1-9][0-9]*)$/.test(token)) {
    throw new Error(`JSON Pointer has an invalid array index: ${JSON.stringify(pointer)}`)
  }
  const index = Number(token)
  if (!Number.isSafeInteger(index) || index >= length) {
    throw new Error(`JSON Pointer array index is out of bounds: ${JSON.stringify(pointer)}`)
  }
  return index
}

function isJsonObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function removeJsonPointer(document: Record<string, unknown>, pointer: string): void {
  const tokens = decodeJsonPointer(pointer)
  let parent: unknown = document
  for (const token of tokens.slice(0, -1)) {
    if (Array.isArray(parent)) {
      parent = parent[jsonArrayIndex(token, parent.length, pointer)]
      continue
    }
    if (!isJsonObject(parent) || !Object.prototype.hasOwnProperty.call(parent, token)) {
      throw new Error(`JSON Pointer field does not exist: ${JSON.stringify(pointer)}`)
    }
    parent = parent[token]
  }

  const finalToken = tokens[tokens.length - 1]
  if (Array.isArray(parent)) {
    parent.splice(jsonArrayIndex(finalToken, parent.length, pointer), 1)
    return
  }
  if (!isJsonObject(parent) || !Object.prototype.hasOwnProperty.call(parent, finalToken)) {
    throw new Error(`JSON Pointer field does not exist: ${JSON.stringify(pointer)}`)
  }
  delete parent[finalToken]
}

describe('evaluation method v2 admission contract', () => {
  it('matches the shared Python, Go, and TypeScript conformance corpus', () => {
    expect(Object.keys(corpus).sort()).toEqual(
      ['base_descriptor', 'cases', 'method_contract_version', 'schema_version'].sort(),
    )
    expect(corpus.schema_version).toBe('evaluation-method-conformance.v1')
    expect(corpus.method_contract_version).toBe('evaluation-method.v2')
    expect(corpus.cases.length).toBeGreaterThan(0)
    expect(new Set(corpus.cases.map((testCase) => testCase.id)).size).toBe(corpus.cases.length)

    for (const testCase of corpus.cases) {
      expect(Object.keys(testCase).sort(), testCase.id).toEqual(
        ['expected_valid', 'id', 'overrides', 'remove_fields'].sort(),
      )
      expect(typeof testCase.id, testCase.id).toBe('string')
      expect(testCase.id.length, testCase.id).toBeGreaterThan(0)
      expect(typeof testCase.expected_valid, testCase.id).toBe('boolean')
      expect(Array.isArray(testCase.remove_fields), testCase.id).toBe(true)
      expect(
        testCase.remove_fields.every((pointer) => typeof pointer === 'string'),
        testCase.id,
      ).toBe(true)
      expect(isJsonObject(testCase.overrides), testCase.id).toBe(true)

      const descriptor = JSON.parse(JSON.stringify(corpus.base_descriptor)) as Record<
        string,
        unknown
      >
      for (const pointer of testCase.remove_fields) removeJsonPointer(descriptor, pointer)
      Object.assign(descriptor, testCase.overrides)
      expect(isEvaluationMethodDescriptor(descriptor), testCase.id).toBe(testCase.expected_valid)
    }
  })
})
