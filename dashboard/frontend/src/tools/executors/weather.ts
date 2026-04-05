import { createTool } from '../registry'
import type { ToolExecutionContext, WeatherArgs, WeatherResult } from '../types'

export type { WeatherArgs, WeatherResult }

function validateWeatherArgs(args: unknown): WeatherArgs {
  if (typeof args !== 'object' || args === null || Array.isArray(args)) {
    throw new Error('Arguments must be an object')
  }

  const { location, unit } = args as Record<string, unknown>
  if (typeof location !== 'string' || !location.trim()) {
    throw new Error('location is required and must be a non-empty string')
  }

  let parsedUnit: 'celsius' | 'fahrenheit' | undefined
  if (typeof unit === 'string' && unit.trim()) {
    const normalized = unit.trim().toLowerCase()
    if (normalized !== 'celsius' && normalized !== 'fahrenheit') {
      throw new Error("unit must be either 'celsius' or 'fahrenheit'")
    }
    parsedUnit = normalized
  }

  return {
    location: location.trim(),
    unit: parsedUnit,
  }
}

async function executeWeather(
  args: WeatherArgs,
  context: ToolExecutionContext,
): Promise<WeatherResult> {
  context.onProgress?.(10)

  const response = await fetch('/api/tools/weather', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...context.headers,
    },
    body: JSON.stringify(args),
    signal: context.signal,
  })

  if (!response.ok) {
    const errorText = await response.text().catch(() => response.statusText)
    throw new Error(`Weather lookup failed: ${response.status} - ${errorText}`)
  }

  context.onProgress?.(100)
  return await response.json() as WeatherResult
}

function formatWeatherResult(result: WeatherResult): string {
  const locationLabel = [result.location.name, result.location.admin1, result.location.country]
    .filter(Boolean)
    .join(', ')
  const temperature = `${result.current.temperature}${result.current.temperature_unit}`
  return `${locationLabel}: ${temperature}, ${result.current.condition}`
}

export const weatherTool = createTool<WeatherArgs, WeatherResult>({
  metadata: {
    id: 'get_weather',
    displayName: 'Weather',
    category: 'custom',
    icon: 'cloud',
    enabled: true,
    version: '1.0.0',
  },
  definition: {
    type: 'function',
    function: {
      name: 'get_weather',
      description: 'Get the current weather for a city, region, or place name, including temperature, apparent temperature, condition, wind, and precipitation.',
      parameters: {
        type: 'object',
        properties: {
          location: {
            type: 'string',
            description: 'The location to look up, for example San Francisco, CA or Chengdu.',
          },
          unit: {
            type: 'string',
            enum: ['celsius', 'fahrenheit'],
            description: "Temperature unit preference. Defaults to 'celsius'.",
            default: 'celsius',
          },
        },
        required: ['location'],
      },
    },
  },
  validateArgs: validateWeatherArgs,
  executor: executeWeather,
  formatResult: formatWeatherResult,
})
