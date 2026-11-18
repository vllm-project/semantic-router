import { expect, test } from 'claude-code/testing'

import { isCleartextRemote, remoteHostOf } from './lib/settings'

test('only loopback addresses count as this machine', () => {
  for (const url of ['http://127.0.0.1:8790/v1/decisions', 'http://127.1.2.3/x', 'http://localhost:8790', 'http://LOCALHOST/x', 'http://[::1]:8790/x', 'http://user@127.0.0.1/x']) {
    expect([url, remoteHostOf(url)]).toEqual([url, undefined])
  }
})

test('a host name that only starts like a loopback address is another machine', () => {
  expect(remoteHostOf('http://127.example.com/v1/decisions')).toBe('127.example.com')
  expect(remoteHostOf('http://127.0.0.1.nip.io/x')).toBe('127.0.0.1.nip.io')
  expect(remoteHostOf('http://localhost.example.com/x')).toBe('localhost.example.com')
  expect(remoteHostOf('http://127.0.0.256/x')).toBe('127.0.0.256')
  expect(isCleartextRemote('http://127.example.com/v1/decisions')).toBe(true)
})
