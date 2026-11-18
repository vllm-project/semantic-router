import { describe, expect, it } from 'vitest'

import {
  componentOptions,
  countNewLines,
  describeEmptyLogs,
  filterLogs,
  formatLogTime,
  getLogLevel,
  logLineText,
  logSourceOf,
  logsDownloadName,
  logsRequestFailure,
  matchRanges,
  parseStructuredLine,
  shortPodName,
  type LogEntry,
  type LogsResponse,
} from './logsPageSupport'

const logs: LogEntry[] = [
  { service: 'router', pod: 'router-7d9f8c6b5-x2x4z', line: '{"level":"info","message":"ready"}' },
  { service: 'envoy', line: '[ERROR] upstream reset' },
  { service: 'dashboard', line: 'request complete' },
]

const answer = (overrides: Partial<LogsResponse>): LogsResponse => ({
  deployment_type: 'kubernetes',
  service: 'all',
  supported: true,
  components: ['router', 'dashboard'],
  logs: [],
  count: 0,
  ...overrides,
})

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
    expect(getLogLevel('{"severity":"TRACE","msg":"tick"}')).toBe('debug')
  })

  it('filters by any selected level and searches the line, component and pod', () => {
    expect(filterLogs(logs, '', ['error'])).toEqual([logs[1]])
    expect(filterLogs(logs, '', ['error', 'info'])).toEqual([logs[0], logs[1]])
    expect(filterLogs(logs, 'dashboard')).toEqual([logs[2]])
    expect(filterLogs(logs, 'x2x4z')).toEqual([logs[0]])
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

  it('parses the fields of a JSON line and nothing else', () => {
    expect(parseStructuredLine('{"level":"info","msg":"ready"}')).toEqual({
      level: 'info',
      msg: 'ready',
    })
    expect(parseStructuredLine('{not json}')).toBeNull()
    expect(parseStructuredLine('plain text')).toBeNull()
  })

  it('names the source and offers the components it reads', () => {
    expect(logSourceOf('local-spool')).toBe('spool')
    expect(logSourceOf('kubernetes')).toBe('kubernetes')
    expect(logSourceOf('unsupported')).toBe('none')
    expect(logSourceOf(undefined)).toBeNull()
    expect(componentOptions(['router', 'dashboard'])).toEqual([
      { value: 'all', label: 'All' },
      { value: 'router', label: 'Router' },
      { value: 'dashboard', label: 'Dashboard' },
    ])
  })

  it('keeps replicas apart without their ReplicaSet hash', () => {
    expect(shortPodName('router-7d9f8c6b5-x2x4z')).toBe('router-x2x4z')
    expect(shortPodName('semantic-router-dashboard-6b8f9d7c4-abcde')).toBe(
      'semantic-router-dashboard-abcde',
    )
    expect(shortPodName('router-0')).toBe('router-0')
  })

  it('counts the lines that arrived after the last one seen', () => {
    const first = { pod: 'a', timestamp: '2026-11-18T02:40:00Z', line: 'same' }
    const second = { pod: 'b', timestamp: '2026-11-18T02:40:00Z', line: 'same' }
    const third = { pod: 'a', timestamp: '2026-11-18T02:40:01Z', line: 'next' }
    expect(countNewLines([first, second], [first, second, third])).toBe(1)
    expect(countNewLines([first], [second, third])).toBe(2)
    expect(countNewLines([], [first])).toBe(0)
  })

  it('formats times locally and copies lines with their source in a mixed view', () => {
    const timestamp = '2026-11-18T02:40:05.123456789Z'
    const time = new Date(timestamp)
    const pad = (value: number) => String(value).padStart(2, '0')
    expect(formatLogTime(timestamp)).toBe(
      `${pad(time.getHours())}:${pad(time.getMinutes())}:${pad(time.getSeconds())}`,
    )
    expect(formatLogTime('not a time')).toBeUndefined()
    const entry = { service: 'router', pod: 'router-a', timestamp, line: 'ready' }
    expect(logLineText(entry, true)).toBe(`${timestamp} [router-a] ready`)
    expect(logLineText({ service: 'router', line: 'ready' }, false)).toBe('ready')
  })

  it('explains failed requests without leaking a raw status line', () => {
    expect(logsRequestFailure(403, null)).toBe('Your account cannot read runtime logs.')
    expect(logsRequestFailure(400, { message: 'Component must be router.' })).toBe(
      'Component must be router.',
    )
    expect(logsRequestFailure(503, 'temporarily unavailable')).toBe(
      'The Dashboard answered HTTP 503.',
    )
  })

  it('says why the view is empty and what to do', () => {
    expect(describeEmptyLogs({ response: null, requestError: null, following: true })).toBeNull()
    expect(
      describeEmptyLogs({
        response: answer({ logs: [{ line: 'x' }] }),
        requestError: null,
        following: true,
      }),
    ).toBeNull()

    const forbidden = describeEmptyLogs({
      response: answer({
        supported: false,
        message: 'The Dashboard\u2019s service account cannot list pods in namespace "vllm-sr".',
      }),
      requestError: null,
      following: true,
    })
    expect(forbidden).toMatchObject({
      kind: 'unavailable',
      title: 'The Dashboard cannot read pod logs',
    })
    expect(forbidden?.description).toContain('cannot list pods')

    expect(
      describeEmptyLogs({
        response: answer({
          deployment_type: 'unsupported',
          supported: false,
          message: 'No spool.',
        }),
        requestError: null,
        following: false,
      }),
    ).toMatchObject({ kind: 'unavailable', title: 'No log source', description: 'No spool.' })

    expect(
      describeEmptyLogs({ response: null, requestError: 'Offline.', following: true }),
    ).toMatchObject({ kind: 'error', description: 'Offline. Follow retries every 5 seconds.' })

    expect(
      describeEmptyLogs({
        response: answer({ message: 'No pods match app=router in namespace "vllm-sr".' }),
        requestError: null,
        following: true,
      }),
    ).toMatchObject({
      kind: 'empty',
      description: 'No pods match app=router in namespace "vllm-sr".',
    })
  })

  it('names downloads by component and UTC time', () => {
    expect(logsDownloadName('all', new Date('2026-11-17T20:01:02.345Z'))).toBe(
      'vllm-sr-all-20261117200102.log',
    )
  })
})
