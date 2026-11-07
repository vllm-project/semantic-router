import type { Tag } from '../../types'
import { isRecord, numberAt, type Reply } from './classify'

// A piece of text to hide, in UTF-16 offsets of the text it was found in; `probability` the classifier's, for a span it found.
export type Found = { from: number; to: number; tag: Tag; probability?: number }

export type Span = { label: string; start: number; end: number; text: string; probability: number }

export const PII_QUESTIONS = { pii: { preset: 'pii' } }

// DATE_TIME is left out: the runtime tags every date, commit dates included, and hiding them
// stopped Claude from lining events up in a live run.
const LABEL_TAGS: Record<string, Tag> = {
  PERSON: 'name',
  EMAIL_ADDRESS: 'email',
  PHONE_NUMBER: 'phone',
  CREDIT_CARD: 'card',
  IP_ADDRESS: 'ip',
  US_SSN: 'ssn',
  IBAN_CODE: 'iban',
  LOCATION: 'address',
  US_DRIVER_LICENSE: 'id',
  US_PASSPORT: 'id',
}

const isSpan = (value: unknown): value is Span =>
  isRecord(value) &&
  typeof value.label === 'string' &&
  typeof value.text === 'string' &&
  numberAt(value, 'start') !== undefined &&
  numberAt(value, 'end') !== undefined &&
  numberAt(value, 'probability') !== undefined

export function parsePii({ answers, spans }: Reply): Span[] | undefined {
  if (numberAt(answers.pii, 'noul') === undefined) return undefined
  return Array.isArray(spans.pii) ? spans.pii.filter(isSpan) : []
}

// A whole value, or the value after a key (`value` names its group; the groups before it are the key).
const SECRET_RULES: { pattern: RegExp; value?: number }[] = [
  { pattern: /-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----/g },
  { pattern: /((?:aws_secret_access_key|AWS_SECRET_ACCESS_KEY)["']?\s*[=:]\s*["']?)([A-Za-z0-9/+=]{16,})/g, value: 2 },
  { pattern: /AKIA[0-9A-Z]{16}/g },
  { pattern: /sk_(?:live|test)_[0-9A-Za-z]{16,}/g },
  { pattern: /ghp_[0-9A-Za-z]{36}/g },
  // Config-style assignments only: a bare `password:` also opens a Python block, and hiding what
  // followed it broke code Claude read in a live run.
  { pattern: /(^[ \t]*(?:export[ \t]+)?[A-Z0-9_]*(?:API_?KEY|SECRET|TOKEN|PASSWORD)[A-Z0-9_]*[ \t]*=[ \t]*["']?)([^\s"'`]{8,})/gm, value: 2 },
  { pattern: /(["'](?:api_?key|secret|token|password)["'][ \t]*:[ \t]*["'])([^"'\n]{8,})/gi, value: 2 },
]

export function secretsIn(text: string): Found[] {
  return SECRET_RULES.flatMap(({ pattern, value }) =>
    [...text.matchAll(pattern)].map(match => {
      const from = (match.index ?? 0) + (value === undefined ? 0 : match[value - 1].length)
      return { from, to: from + (value === undefined ? match[0] : match[value]).length, tag: 'secret' as const }
    }),
  )
}

const WORD = /[A-Za-z0-9]/

// The spans at or above `line`, of the labels that are personal data (a DOMAIN_NAME is not), in UTF-16 offsets of `text`; a span that stops inside a word runs to the word's end.
export function piiIn(text: string, spans: readonly Span[], line: number): Found[] {
  const offsets: number[] = []
  let at = 0
  for (const char of text) {
    offsets.push(at)
    at += char.length
  }
  offsets.push(at)
  return spans.flatMap(span => {
    const tag = LABEL_TAGS[span.label]
    if (tag === undefined || span.probability < line) return []
    let from = offsets[span.start]
    let to = offsets[span.end]
    if (from === undefined || to === undefined || text.slice(from, to) !== span.text) {
      from = text.indexOf(span.text)
      to = from + span.text.length
    }
    if (from < 0 || span.text === '') return []
    while (from > 0 && WORD.test(text[from - 1]) && WORD.test(text[from])) from--
    while (to < text.length && WORD.test(text[to]) && WORD.test(text[to - 1])) to++
    return [{ from, to, tag, probability: span.probability }]
  })
}

export type Hidden = { text: string; counts: Partial<Record<Tag, number>> }

// Replaces each found piece with its tag; of two that overlap, the one that starts first stays.
export function hide(text: string, found: readonly Found[]): Hidden {
  const counts: Partial<Record<Tag, number>> = {}
  let out = ''
  let at = 0
  for (const one of [...found].sort((a, b) => a.from - b.from || b.to - a.to)) {
    if (one.from < at) continue
    out += `${text.slice(at, one.from)}[${one.tag}]`
    counts[one.tag] = (counts[one.tag] ?? 0) + 1
    at = one.to
  }
  return { text: out + text.slice(at), counts }
}

export function addCounts(...all: Partial<Record<Tag, number>>[]): Partial<Record<Tag, number>> {
  const sum: Partial<Record<Tag, number>> = {}
  for (const counts of all) {
    for (const [tag, n] of Object.entries(counts) as [Tag, number][]) sum[tag] = (sum[tag] ?? 0) + n
  }
  return sum
}
