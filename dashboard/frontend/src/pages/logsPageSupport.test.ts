import { describe, expect, it } from 'vitest'

import { describeLogsAvailability, filterLogs, getLogLevel, type LogEntry } from './logsPageSupport'

const logs: LogEntry[] = [
  { service: 'router', line: '{"level":"info","message":"ready"}' },
  { service: 'envoy', line: '[ERROR] upstream reset' },
  { service: 'dashboard', line: 'request complete' },
]

describe('logs page support', () => {
  it('normalizes common log levels', () => {
    expect(getLogLevel(logs[0].line)).toBe('info')
    expect(getLogLevel(logs[1].line)).toBe('error')
    expect(getLogLevel(logs[2].line)).toBe('other')
  })

  it('filters by level and searches both service and message', () => {
    expect(filterLogs(logs, '', 'error')).toEqual([logs[1]])
    expect(filterLogs(logs, 'dashboard', 'all')).toEqual([logs[2]])
    expect(filterLogs(logs, 'ready', 'info')).toEqual([logs[0]])
  })

  it('describes the explicit bounded-spool support contract', () => {
    expect(describeLogsAvailability(null)).toBe('Detecting')
    expect(describeLogsAvailability(true)).toBe('Available')
    expect(describeLogsAvailability(false)).toBe('Unavailable')
  })
})
