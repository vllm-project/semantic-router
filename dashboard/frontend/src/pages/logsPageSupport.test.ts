import { describe, expect, it } from 'vitest'

import {
  describeLogsAvailability,
  filterLogs,
  getLogLevel,
  logsDownloadName,
  matchRanges,
  parseStructuredLine,
  splitServicePrefix,
  type LogEntry,
} from './logsPageSupport'

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

  it('reads levels from logfmt, Envoy and leading-token formats', () => {
    expect(getLogLevel('ts=2026-11-17T20:00:00Z level=warning msg="slow upstream"')).toBe('warn')
    expect(getLogLevel('[2026-11-17 20:00:00.123][12][debug][main] loading config')).toBe('debug')
    expect(getLogLevel('2026-11-17T20:00:00Z FATAL could not bind')).toBe('error')
    expect(getLogLevel('[router] {"severity":"TRACE","msg":"tick"}')).toBe('debug')
  })

  it('filters by any selected level and searches both service and message', () => {
    expect(filterLogs(logs, '', ['error'])).toEqual([logs[1]])
    expect(filterLogs(logs, '', ['error', 'info'])).toEqual([logs[0], logs[1]])
    expect(filterLogs(logs, 'dashboard')).toEqual([logs[2]])
    expect(filterLogs(logs, 'ready', ['info'])).toEqual([logs[0]])
  })

  it('finds every case-insensitive match for highlighting', () => {
    expect(matchRanges('Ready, ready, READY', 'ready')).toEqual([
      { start: 0, end: 5 },
      { start: 7, end: 12 },
      { start: 14, end: 19 },
    ])
    expect(matchRanges('anything', '  ')).toEqual([])
  })

  it('separates the combined-stream service prefix and parses JSON fields', () => {
    expect(splitServicePrefix('[router] [INFO] request completed')).toEqual({
      service: 'router',
      rest: '[INFO] request completed',
    })
    expect(splitServicePrefix('Authorization: [REDACTED]').service).toBeNull()
    expect(parseStructuredLine('[router] {"level":"info","msg":"ready"}')).toEqual({
      level: 'info',
      msg: 'ready',
    })
    expect(parseStructuredLine('{not json}')).toBeNull()
    expect(parseStructuredLine('plain text')).toBeNull()
  })

  it('describes the explicit bounded-spool support contract', () => {
    expect(describeLogsAvailability(null)).toBe('Detecting')
    expect(describeLogsAvailability(true)).toBe('Available')
    expect(describeLogsAvailability(false)).toBe('Unavailable')
  })

  it('names downloads by component and UTC time', () => {
    expect(logsDownloadName('all', new Date('2026-11-17T20:01:02.345Z'))).toBe(
      'vllm-sr-all-20261117200102.log',
    )
  })
})
