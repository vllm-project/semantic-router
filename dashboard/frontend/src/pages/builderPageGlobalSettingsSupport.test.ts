import { describe, expect, it } from 'vitest'

import type { DSLFieldObject } from '@/types/dsl'
import { getListeners, serializeListeners } from './builderPageGlobalSettingsSupport'

const tlsListener: DSLFieldObject = {
  name: 'https-8443',
  address: '0.0.0.0',
  port: 8443,
  timeout: '120s',
  api_keys: ['client-key-a', 'client-key-b'],
  tls: { cert_file: 'certs/server.crt', key_file: 'certs/server.key' },
  identity: { trust_headers: true, trusted_peers: ['192.0.2.0/24'] },
}

describe('Builder listener round trip', () => {
  it('carries tls, api_keys and identity through an edit unchanged', () => {
    const [listener] = getListeners({ listeners: [tlsListener] }, 'listeners')

    const [written] = serializeListeners([{ ...listener, port: 9443 }])

    expect(written).toEqual({ ...tlsListener, port: 9443 })
  })

  it('writes the edited fields over the stored ones', () => {
    const [listener] = getListeners({ listeners: [tlsListener] }, 'listeners')

    const [written] = serializeListeners([{ ...listener, name: 'edge', timeout: '30s' }])

    expect(written).toMatchObject({ name: 'edge', timeout: '30s', port: 8443 })
    expect(written.tls).toEqual(tlsListener.tls)
  })

  it('writes a new listener with only the fields the editor sets', () => {
    const [written] = serializeListeners([
      { name: 'http-8900', address: '0.0.0.0', port: 8900, timeout: '300s', source: {} },
    ])

    expect(written).toEqual({ name: 'http-8900', address: '0.0.0.0', port: 8900, timeout: '300s' })
  })
})
