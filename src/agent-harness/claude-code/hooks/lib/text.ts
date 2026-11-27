import type { Risk } from '../../types'

export const oneLine = (text: string) => text.replace(/\s+/g, ' ').trim()
export const clip = (text: string, width: number) => (text.length > width ? `${text.slice(0, Math.max(0, width - 1))}…` : text)
export const fixed = (value: number) => value.toFixed(2)
export const capital = (word: string) => word.charAt(0).toUpperCase() + word.slice(1)
export const firstChars = (text: string, count: number) => Array.from(text).slice(0, count).join('')
export const plural = (count: number, one: string, many = `${one}s`) => `${count} ${count === 1 ? one : many}`

export const RISK_WORDS: Record<Risk, string> = {
  safe: 'looks routine',
  destructive: 'looks destructive',
  exfiltration: 'may send data out',
  remote_code: 'downloads and runs code',
}

export function modelName(model: string): string {
  const versioned = /^claude-(opus|sonnet|haiku|fable|mythos)-(\d+)(?:-(\d{1,2}))?(?:-\d{8})?$/.exec(model)
  if (versioned) return `${capital(versioned[1])} ${versioned[2]}${versioned[3] ? `.${versioned[3]}` : ''}`
  return /^(opus|sonnet|haiku|fable|mythos)$/.test(model) ? capital(model) : model
}

// The end of an answer as matched against its last drawn block: whitespace and markdown emphasis left out.
export const answerKey = (text: string) => oneLine(text.replace(/[*_`#>]/g, ''))
