import { describe, expect, it } from 'vitest'

import { getProviderIconAsset } from './modelProviderLogoSupport'

describe('provider icon descriptors', () => {
  it('renders control-plane descriptors without a provider lookup table', () => {
    expect(getProviderIconAsset({ source: 'lobe', value: 'example', color: true })).toBe(
      'https://unpkg.com/@lobehub/icons-static-svg@1.90.0/icons/example-color.svg',
    )
    expect(getProviderIconAsset({ source: 'asset', value: '/example.svg', color: true })).toBe(
      '/example.svg',
    )
    expect(
      getProviderIconAsset({
        source: 'url',
        value: 'https://icons.example.test/example.svg',
        color: true,
      }),
    ).toBe('https://icons.example.test/example.svg')
  })
})
