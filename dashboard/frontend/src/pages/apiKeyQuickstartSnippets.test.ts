import { describe, expect, it } from 'vitest'
import { buildAPIKeyQuickstartSnippets } from './apiKeyQuickstartSnippets'

describe('API key quickstart snippets', () => {
  it('escapes user-authored routing names and revealed secrets for every language', () => {
    const model = `team/'$(command)'quoted"\\model`
    const secret = `key/'quoted"\\secret`
    const snippets = buildAPIKeyQuickstartSnippets('https://router.example/v1', model, secret)

    expect(snippets.python).toContain(`model=${JSON.stringify(model)}`)
    expect(snippets.python).toContain(`api_key=${JSON.stringify(secret)}`)
    expect(snippets.javascript).toContain(`model: ${JSON.stringify(model)}`)
    expect(snippets.javascript).toContain(`apiKey: ${JSON.stringify(secret)}`)
    expect(snippets.curl).toContain(`'"'"'quoted`)
    expect(snippets.curl).toContain(`'"'"'$(command)'"'"'`)
    expect(snippets.curl).not.toContain(`"model":"${model}"`)
    expect(snippets.curl).not.toContain('\n+')
    expect(snippets.curl).toMatch(/\n {2}-H/)
    expect(snippets.curl).toMatch(/\n {2}-d/)
  })

  it('keeps environment-based key examples executable without a revealed secret', () => {
    const snippets = buildAPIKeyQuickstartSnippets('https://router.example/v1', 'vllm-sr/blend', '')

    expect(snippets.python).toContain('os.environ["VLLM_SR_API_KEY"]')
    expect(snippets.javascript).toContain('process.env.VLLM_SR_API_KEY')
    expect(snippets.curl).toContain('"Authorization: Bearer $VLLM_SR_API_KEY"')
  })
})
