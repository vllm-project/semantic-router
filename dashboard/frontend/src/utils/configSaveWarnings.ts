/** A finding the Router's validator reports about a configuration it accepts. */
export interface RouterConfigWarning {
  code: string
  field: string
  message: string
}

/** The Router's warnings in a configuration save response; none when it lists none. */
export function configSaveWarnings(body: unknown): RouterConfigWarning[] {
  const warnings =
    body && typeof body === 'object' ? (body as { warnings?: unknown }).warnings : undefined
  if (!Array.isArray(warnings)) return []
  return warnings.flatMap((item) => {
    if (!item || typeof item !== 'object') return []
    const { code, field, message } = item as Record<string, unknown>
    if (typeof message !== 'string' || !message) return []
    return [
      {
        code: typeof code === 'string' ? code : '',
        field: typeof field === 'string' ? field : '',
        message,
      },
    ]
  })
}

/** Reads a successful save response's warnings without consuming the response for other readers. */
export async function responseConfigWarnings(response: Response): Promise<RouterConfigWarning[]> {
  try {
    return configSaveWarnings(await response.clone().json())
  } catch {
    return []
  }
}
