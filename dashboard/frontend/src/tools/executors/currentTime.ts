import { createTool } from '../registry'
import type { CurrentTimeArgs, CurrentTimeResult, ToolExecutionContext } from '../types'

export type { CurrentTimeArgs, CurrentTimeResult }

const DEFAULT_LOCALE = 'en-US'

function resolveLocale(locale?: string) {
  if (locale && locale.trim()) {
    return locale.trim()
  }

  if (typeof navigator !== 'undefined' && navigator.language) {
    return navigator.language
  }

  return DEFAULT_LOCALE
}

function resolveTimezone(timezone?: string) {
  if (timezone && timezone.trim()) {
    const value = timezone.trim()
    try {
      new Intl.DateTimeFormat(DEFAULT_LOCALE, { timeZone: value }).format(new Date())
      return value
    } catch {
      throw new Error(`Invalid timezone: ${value}`)
    }
  }

  return Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC'
}

function formatOffset(now: Date, timezone: string) {
  try {
    const parts = new Intl.DateTimeFormat(DEFAULT_LOCALE, {
      timeZone: timezone,
      timeZoneName: 'shortOffset',
      hour: '2-digit',
    }).formatToParts(now)
    const raw = parts.find(part => part.type === 'timeZoneName')?.value
    return raw ? raw.replace('GMT', 'UTC') : undefined
  } catch {
    return undefined
  }
}

function buildCurrentTimeResult(args: CurrentTimeArgs): CurrentTimeResult {
  const now = new Date()
  const locale = resolveLocale(args.locale)
  const timezone = resolveTimezone(args.timezone)
  const dateFormatter = new Intl.DateTimeFormat(locale, {
    timeZone: timezone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  })
  const timeFormatter = new Intl.DateTimeFormat(locale, {
    timeZone: timezone,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  })
  const weekdayFormatter = new Intl.DateTimeFormat(locale, {
    timeZone: timezone,
    weekday: 'long',
  })

  const localDate = dateFormatter.format(now)
  const localTime = timeFormatter.format(now)
  const weekday = weekdayFormatter.format(now)

  return {
    timezone,
    locale,
    iso: now.toISOString(),
    local_date: localDate,
    local_time: localTime,
    local_datetime: `${weekday}, ${localDate} ${localTime}`,
    weekday,
    unix_ms: now.getTime(),
    utc_offset: formatOffset(now, timezone),
  }
}

function validateCurrentTimeArgs(args: unknown): CurrentTimeArgs {
  if (args === null || args === undefined) {
    return {}
  }

  if (typeof args !== 'object' || Array.isArray(args)) {
    throw new Error('Arguments must be an object')
  }

  const { timezone, locale } = args as Record<string, unknown>

  return {
    timezone: typeof timezone === 'string' ? timezone : undefined,
    locale: typeof locale === 'string' ? locale : undefined,
  }
}

async function executeCurrentTime(
  args: CurrentTimeArgs,
  context: ToolExecutionContext,
): Promise<CurrentTimeResult> {
  context.onProgress?.(100)
  return buildCurrentTimeResult(args)
}

function formatCurrentTimeResult(result: CurrentTimeResult): string {
  const offset = result.utc_offset ? ` (${result.utc_offset})` : ''
  return [
    `Current time in ${result.timezone}${offset}`,
    `${result.local_datetime}`,
    `ISO: ${result.iso}`,
  ].join('\n')
}

export const currentTimeTool = createTool<CurrentTimeArgs, CurrentTimeResult>({
  metadata: {
    id: 'current_time',
    displayName: 'Current Time',
    category: 'custom',
    icon: 'clock',
    enabled: true,
    version: '1.0.0',
  },
  definition: {
    type: 'function',
    function: {
      name: 'current_time',
      description: 'Get the current date and time, optionally formatted for a specific IANA timezone such as Asia/Shanghai or America/New_York.',
      parameters: {
        type: 'object',
        properties: {
          timezone: {
            type: 'string',
            description: 'Optional IANA timezone name to format the time in, for example Asia/Shanghai or America/New_York.',
          },
          locale: {
            type: 'string',
            description: 'Optional BCP 47 locale tag used for formatting, for example en-US or zh-CN.',
          },
        },
        required: [],
      },
    },
  },
  validateArgs: validateCurrentTimeArgs,
  executor: executeCurrentTime,
  formatResult: formatCurrentTimeResult,
})
