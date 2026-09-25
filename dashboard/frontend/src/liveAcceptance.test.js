import { afterEach, describe, expect, it } from 'vitest'
import { chmodSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { acceptanceOrigin } from '../e2e/support/liveAcceptance'

const directories = []
afterEach(() =>
  directories.splice(0).forEach((directory) => rmSync(directory, { recursive: true, force: true })),
)

function publicPlan() {
  const directory = mkdtempSync(join(tmpdir(), 'sr-bench-auth-guard-'))
  directories.push(directory)
  const state = join(directory, 'state.json')
  writeFileSync(state, '{"cookies":[],"origins":[]}', { mode: 0o600 })
  return {
    base_url: 'https://dashboard.example',
    public_https_origin: 'https://dashboard.example',
    expected_deployment_sha: 'a'.repeat(40),
    auth_state_path: state,
  }
}

describe('explicit live acceptance connection', () => {
  it('preserves credential-free loopback and rejects arbitrary or credential-bearing origins', () => {
    expect(acceptanceOrigin({ base_url: 'http://127.0.0.1:18100' })).toBe('http://127.0.0.1:18100')
    for (const base_url of [
      'http://dashboard.example',
      'https://dashboard.example',
      'http://user:pass@localhost',
      'http://localhost/path',
    ])
      expect(() => acceptanceOrigin({ base_url })).toThrow()
  })

  it('requires matching explicit public origin and a secure state file', () => {
    const plan = publicPlan()
    expect(acceptanceOrigin(plan, plan.public_https_origin)).toBe(plan.base_url)
    expect(() => acceptanceOrigin(plan, 'https://other.example')).toThrow()
    expect(() =>
      acceptanceOrigin({ ...plan, expected_deployment_sha: '' }, plan.public_https_origin),
    ).toThrow()
    chmodSync(plan.auth_state_path, 0o644)
    expect(() => acceptanceOrigin(plan, plan.public_https_origin)).toThrow(/0600/)
  })

  it('rejects symlink state files instead of following another secret location', () => {
    const plan = publicPlan()
    const link = `${plan.auth_state_path}.link`
    symlinkSync(plan.auth_state_path, link)
    expect(() =>
      acceptanceOrigin({ ...plan, auth_state_path: link }, plan.public_https_origin),
    ).toThrow(/0600/)
  })
})
