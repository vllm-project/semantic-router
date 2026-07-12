import { describe, expect, it } from 'vitest'

import {
  formatAccountRole,
  getAccountInitials,
  groupAccountPermissions,
} from './LayoutAccountControlSupport'

describe('account control presentation support', () => {
  it('derives compact initials from a name and falls back to email', () => {
    expect(getAccountInitials('Ada Lovelace', 'ada@example.com')).toBe('AL')
    expect(getAccountInitials('', 'router@example.com')).toBe('RO')
    expect(getAccountInitials(' ', ' ')).toBe('U')
  })

  it('formats machine roles for display', () => {
    expect(formatAccountRole('platform_admin')).toBe('Platform Admin')
    expect(formatAccountRole('read-only.user')).toBe('Read Only User')
    expect(formatAccountRole()).toBe('Unknown role')
  })

  it('deduplicates and groups permissions without changing their order', () => {
    expect(
      groupAccountPermissions([
        'config.read',
        ' config.write ',
        'users:manage',
        'config.read',
        'health',
        '',
      ]),
    ).toEqual([
      { key: 'config', label: 'Config', permissions: ['config.read', 'config.write'] },
      { key: 'users', label: 'Users', permissions: ['users:manage'] },
      { key: 'general', label: 'General', permissions: ['health'] },
    ])
  })
})
