import { test } from 'node:test'
import assert from 'node:assert/strict'
import { findUnresolvedIdentities } from './identity-audit.mjs'

test('flags an email-keyed identity with commits at or above the threshold', () => {
  const entries = [
    { key: 'email:someone@example.com', name: 'Someone', commits: 5 },
  ]

  const result = findUnresolvedIdentities(entries)

  assert.equal(result.length, 1)
  assert.equal(result[0].key, 'email:someone@example.com')
})

test('does not flag a github-keyed identity regardless of commit count', () => {
  const entries = [
    { key: 'github:someone', name: 'Someone', commits: 100 },
  ]

  assert.deepEqual(findUnresolvedIdentities(entries), [])
})

test('does not flag an email-keyed identity below the commit threshold', () => {
  const entries = [
    { key: 'email:someone@example.com', name: 'Someone', commits: 1 },
  ]

  assert.deepEqual(findUnresolvedIdentities(entries, { minCommits: 2 }), [])
})

test('respects a custom minCommits threshold', () => {
  const entries = [
    { key: 'email:someone@example.com', name: 'Someone', commits: 3 },
  ]

  assert.deepEqual(findUnresolvedIdentities(entries, { minCommits: 5 }), [])
  assert.equal(findUnresolvedIdentities(entries, { minCommits: 3 }).length, 1)
})

test('ignores bot-like keys that happen to lack a github: prefix', () => {
  const entries = [
    { key: 'dependabot[bot]', name: 'dependabot[bot]', commits: 50, isBot: true },
  ]

  assert.deepEqual(findUnresolvedIdentities(entries), [])
})
