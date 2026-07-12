export interface LogEntry {
  line: string
  service?: string
}

export type LogLevel = 'all' | 'error' | 'warn' | 'info' | 'debug' | 'other'

export function getLogLevel(line: string): Exclude<LogLevel, 'all'> {
  const normalized = line.toLocaleLowerCase()
  if (normalized.includes('"level":"error"') || normalized.includes('[error]')) return 'error'
  if (normalized.includes('"level":"warn"') || normalized.includes('[warn]')) return 'warn'
  if (normalized.includes('"level":"info"') || normalized.includes('[info]')) return 'info'
  if (normalized.includes('"level":"debug"') || normalized.includes('[debug]')) return 'debug'
  return 'other'
}

export function filterLogs(logs: LogEntry[], query: string, level: LogLevel): LogEntry[] {
  const normalizedQuery = query.trim().toLocaleLowerCase()
  return logs.filter((entry) => {
    if (level !== 'all' && getLogLevel(entry.line) !== level) return false
    if (!normalizedQuery) return true
    return `${entry.service ?? ''} ${entry.line}`.toLocaleLowerCase().includes(normalizedQuery)
  })
}
