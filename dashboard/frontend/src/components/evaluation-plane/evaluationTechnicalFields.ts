export interface TechnicalField {
  label: string
  value: number | string
  copyable?: boolean
  displayValue?: string
  mono?: boolean
}

export type CopyState = 'idle' | 'copying' | 'copied' | 'failed'

type ClipboardWriter = Pick<Clipboard, 'writeText'>

export function shortIdentity(value: string): string {
  if (value.length <= 28) return value
  return `${value.slice(0, 15)}…${value.slice(-8)}`
}

export function copyField(label: string, value: string): TechnicalField {
  return { label, value, copyable: true }
}

export function digestField(label: string, value: string): TechnicalField {
  return { ...copyField(label, value), displayValue: shortIdentity(value) }
}

export async function copyTextToClipboard(
  value: string,
  clipboard: ClipboardWriter | undefined,
): Promise<Extract<CopyState, 'copied' | 'failed'>> {
  try {
    if (!clipboard) throw new Error('Clipboard API is unavailable.')
    await clipboard.writeText(value)
    return 'copied'
  } catch {
    return 'failed'
  }
}

export function copyActionLabel(label: string, copyState: CopyState): string {
  switch (copyState) {
    case 'copying':
      return `Copying ${label}`
    case 'copied':
      return `Copied ${label}`
    case 'failed':
      return `Retry copy ${label}`
    default:
      return `Copy ${label}`
  }
}
