import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

describe('dashboard social metadata', () => {
  it('publishes the branded large-image preview', () => {
    const html = readFileSync(new URL('../../index.html', import.meta.url), 'utf8')
    const image = readFileSync(
      new URL('../../public/vllm-sr-logo.social.png', import.meta.url),
    )

    expect(html).toContain('property="og:site_name" content="vLLM Semantic Router"')
    expect(html).toMatch(
      /property="og:image"\s+content="https:\/\/play\.vllm-semantic-router\.com\/vllm-sr-logo\.social\.png"/,
    )
    expect(html).toContain('name="twitter:card" content="summary_large_image"')
    expect(html).toMatch(
      /name="twitter:image"\s+content="https:\/\/play\.vllm-semantic-router\.com\/vllm-sr-logo\.social\.png"/,
    )
    expect(image.subarray(1, 4).toString('ascii')).toBe('PNG')
    expect(image.byteLength).toBeGreaterThan(100_000)
  })
})
