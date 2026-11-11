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

// A key that names a credential (`token`, `api_key`, `client_secret`, `db_password`, `access_token`), the keyword
// ending the key or followed by `_`, `-` or `.`: not `tokens`, `tokenizer` or `secretary`. `pass` only as a word of
// its own (`db_pass`), not in `bypass` or `passport`.
const KEY = String.raw`[A-Za-z0-9_.-]*?(?:api[_-]?key|apikey|secret|token|password|passwd|(?<![A-Za-z0-9])pass(?![A-Za-z0-9])|credentials?|private[_-]?key|access[_-]?key)(?:[_.-][A-Za-z0-9_.-]*)?`
// A value that names another value rather than holding one: `settings.API_KEY`, `process.env.TOKEN`, a call
// (`getpass.getpass()`, `get_token()`) or a shell variable (`$TOKEN`).
const IDENTIFIER = String.raw`[A-Za-z_][A-Za-z0-9_]*`
const REFERENCE = String.raw`(?:${IDENTIFIER}(?:\.${IDENTIFIER})*\([^\s]*\)|${IDENTIFIER}(?:\.${IDENTIFIER})+|\$\{?${IDENTIFIER}\}?)(?:[ \t;]|$)`
// The characters of a token or key: no spaces, quotes, brackets or calls.
const VALUE_CHAR = String.raw`[A-Za-z0-9._~+/=:@%!*^#-]`

// A whole value, or with `value` the capture group that is the value, the rest of the match (a key, a header) kept.
// Every rule with a `value` carries the `d` flag, so the group's own offsets are known.
const SECRET_RULES: { pattern: RegExp; value?: number }[] = [
  { pattern: /-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----/g },
  { pattern: /(?:aws_secret_access_key|AWS_SECRET_ACCESS_KEY)["']?\s*[=:]\s*["']?([A-Za-z0-9/+=]{16,})/dg, value: 1 },
  { pattern: /AKIA[0-9A-Z]{16}/g },
  { pattern: /sk_(?:live|test)_[0-9A-Za-z]{16,}/g },
  // GitHub (classic and fine-grained), Anthropic, OpenAI and Slack tokens, wherever they stand.
  { pattern: /(?<![A-Za-z0-9])gh[pousr]_[0-9A-Za-z]{36,}/g },
  { pattern: /(?<![A-Za-z0-9])github_pat_[0-9A-Za-z_]{22,}/g },
  { pattern: /(?<![A-Za-z0-9-])sk-(?:ant|proj|svcacct|admin)-[0-9A-Za-z_-]{20,}/g },
  { pattern: /(?<![A-Za-z0-9])xox[abposr]-[0-9A-Za-z-]{10,}/g },
  // An Authorization header's credentials, after its scheme or, a long one, with none.
  { pattern: /\bauthorization["']?[ \t]*[:=][ \t]*["']?(?:bearer|basic|token|bot|digest|apikey)[ \t]+([A-Za-z0-9._~+/=-]{8,})/dgi, value: 1 },
  { pattern: /\bauthorization["']?[ \t]*:[ \t]*["']?(?!(?:bearer|basic|token|bot|digest|apikey)\b)([A-Za-z0-9._~+/=-]{16,})/dgi, value: 1 },
  { pattern: /\b(?:x-)?api-key[ \t]*:[ \t]*["']?([A-Za-z0-9._~+/=-]{16,})/dgi, value: 1 },
  // A password in a connection URL: `postgres://user:hunter2@db`.
  { pattern: /\b[a-z][a-z0-9+.-]*:\/\/[^\s:/@]+:([^\s:/@]+)@/dgi, value: 1 },
  // Config-style assignments only: a bare `password:` also opens a Python block, and hiding what
  // followed it broke code Claude read in a live run. The value ends the word and is not code: not
  // `os.environ["SECRET_KEY"]`, `settings.API_KEY` or `$TOKEN` (a quoted value is the rules' below).
  {
    pattern: new RegExp(
      String.raw`^[ \t]*(?:export[ \t]+)?[A-Z0-9_]*(?:API_?KEY|SECRET|TOKEN|PASSWORD|PASSWD|(?<![A-Z0-9])PASS(?![A-Z0-9]))[A-Z0-9_]*[ \t]*=[ \t]*(?!${REFERENCE})([^\s"'\x60]{8,})(?=[ \t;]|$)`,
      'dgm',
    ),
    value: 1,
  },
  // The same in any case, shell style (no spaces around `=`), the value ending the word: `db_password=…`, not `password_hint=hint()`.
  { pattern: new RegExp(String.raw`^[ \t]*(?:export[ \t]+)?${KEY}=["']?(${VALUE_CHAR}{8,})(?=["']?(?:[ \t;]|$))`, 'dgim'), value: 1 },
  // INI and .cfg style (`api_key = …`), held to the YAML rule's bar below, so `api_key = settings.api_key` stays.
  {
    pattern: new RegExp(String.raw`^[ \t]*${KEY}[ \t]+=[ \t]+(?=[^\s]*[A-Za-z])(?=[^\s]*\d)(${VALUE_CHAR}{8,})[ \t]*$`, 'dgim'),
    value: 1,
  },
  // A quoted key with a quoted value: JSON, YAML, a Python dict.
  { pattern: new RegExp(String.raw`["']${KEY}["'][ \t]*:[ \t]*["']([^"'\n]{8,})`, 'dgi'), value: 1 },
  // A bare key with a quoted value of one word: YAML, TOML, an assignment in code.
  { pattern: new RegExp(String.raw`(?<![A-Za-z0-9_.$-])${KEY}[ \t]*[:=][ \t]*(["'])(${VALUE_CHAR}{8,})\1`, 'dgi'), value: 2 },
  // A bare YAML key with a bare value that ends the line and holds a letter and a digit, unlike a type or a count.
  {
    pattern: new RegExp(String.raw`^[ \t]*(?:-[ \t]+)?${KEY}[ \t]*:[ \t]+(?=[^\s]*[A-Za-z])(?=[^\s]*\d)(${VALUE_CHAR}{8,})[ \t]*$`, 'dgim'),
    value: 1,
  },
]

export function secretsIn(text: string): Found[] {
  return SECRET_RULES.flatMap(({ pattern, value }) =>
    [...text.matchAll(pattern)].flatMap(match => {
      const [from, to] = value === undefined ? [match.index ?? 0, (match.index ?? 0) + match[0].length] : (match.indices?.[value] ?? [])
      return from === undefined || to === undefined ? [] : [{ from, to, tag: 'secret' as const }]
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

// What stands where a value was hidden: `[vsr:email]`, a tag no code holds, so writing one back is always a lost value.
// The Router's masking plugin leaves text of this shape alone (placeholderPattern in pkg/masking), so keep it.
export const tagText = (tag: Tag) => `[vsr:${tag}]`

// Replaces each found piece with its tag; of two that overlap, the one that starts first stays.
export function hide(text: string, found: readonly Found[]): Hidden {
  const counts: Partial<Record<Tag, number>> = {}
  let out = ''
  let at = 0
  for (const one of [...found].sort((a, b) => a.from - b.from || b.to - a.to)) {
    if (one.from < at) continue
    out += `${text.slice(at, one.from)}${tagText(one.tag)}`
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
