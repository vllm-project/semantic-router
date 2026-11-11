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

// What kind of text the rules read, which decides what a value with no quotes is: in a config file a literal, in
// source code an expression (a name, a call), and in anything else (a command's output) judged by its shape.
export type Kind = 'config' | 'code' | 'unknown'

const CONFIG_NAMES = /^(?:\.env.*|.*\.env|credentials|\.netrc|\.npmrc|\.pypirc|\.htpasswd|\.pgpass|\.my\.cnf)$/i
const CONFIG_EXTENSIONS = new Set(['ini', 'cfg', 'conf', 'cnf', 'properties', 'toml', 'yaml', 'yml', 'tfvars', 'env'])
const CODE_EXTENSIONS = new Set([
  'py', 'pyi', 'ts', 'tsx', 'js', 'jsx', 'mjs', 'cjs', 'go', 'rs', 'java', 'kt', 'kts', 'scala', 'rb', 'php', 'c', 'h', 'cc', 'cpp', 'hpp',
  'cs', 'swift', 'm', 'mm', 'lua', 'pl', 'r', 'jl', 'dart', 'ex', 'exs', 'erl', 'hs', 'ml', 'clj', 'vue', 'svelte', 'zig', 'nim', 'groovy', 'tf', 'hcl',
])

// The kind of a file, by its name: a shell script, a Dockerfile or a Markdown page holds both, so it is `unknown`.
export function kindOf(path: string): Kind {
  const name = path.split(/[/\\]/).pop() ?? ''
  if (CONFIG_NAMES.test(name)) return 'config'
  const extension = name.includes('.') ? (name.split('.').pop() ?? '').toLowerCase() : ''
  return CONFIG_EXTENSIONS.has(extension) ? 'config' : CODE_EXTENSIONS.has(extension) ? 'code' : 'unknown'
}

// The kind of what a command prints, when every file it names is of one kind (`cat .env`, `grep token src/a.ts`).
export function kindOfCommand(command: string): Kind {
  const kinds = new Set((command.match(/[^\s"'`|;&<>()=]+/g) ?? []).filter(word => /[./]/.test(word) && !word.startsWith('-')).map(kindOf))
  kinds.delete('unknown')
  return kinds.size === 1 ? [...kinds][0] : 'unknown'
}

// ---- Keys: which names hold a credential.

// How much a key says its value is a credential: `strong` a password, `normal` a secret, a token or a named key,
// `weak` a name that often holds something else (`auth`, `OPENAI_KEY`), whose value must also look random.
type Tier = 'strong' | 'normal' | 'weak'

// The words of a key, lower-cased: `XGoogAPIKey` is x goog api key, `DB_PASSWORD` db password.
const keyWords = (key: string) =>
  key
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/([A-Z]+)([A-Z][a-z])/g, '$1 $2')
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter(word => word !== '')

const setOf = (words: string) => new Set(words.split(' '))

// The word before `key` that makes it a credential: `api_key`, `private_key`, `account_key`.
const KEY_OWNERS = setOf(
  'api private access secret signing sign encryption encrypt master account shared ssh license licence auth consumer subscription deploy jwt hmac aes webhook bot',
)
// The word before `KEY` in an environment variable that is not a credential: `CACHE_KEY`, `PRIMARY_KEY`.
const PLAIN_KEYS = setOf(
  'primary foreign sort partition cache public pub object row hash map dict json lookup index group bucket redis unique shard routing dedup idempotency config env feature flag metric label s3 i18n translation search query order column field table item entry state event topic queue message trace span request correlation',
)
// The word before `token` that makes it a piece of text or a cursor, not a credential: `eos_token`, `next_token`.
const TEXT_TOKENS = setOf(
  'eos bos pad unk sep cls mask special stop start end next prev previous first last current cur peek expected unexpected input output cached prompt completion per cancel cancellation page continuation max min num total each every word sub sentence lexer parser syntax',
)
// A word before the credential word that makes the key a switch or a pointer: `existingSecret`, `show_password`.
const HEAD_META = setOf('existing has is show hide use enable disable require requires allow skip no num max min')
// A word after the credential word that makes the value a description of the credential, not the credential:
// `password_file`, `token_url`, `api_key_env`, `secret_name`, `token_count`.
const TAIL_META = setOf(
  'url uri urls endpoint endpoints host file files filename filepath path paths dir directory name names env envvar var variable type types kind hint header headers field fields column param parameter arg flag option label title placeholder description desc help message msg error prompt ttl expiry expires expire expiration exp lifetime timeout duration age interval rotation length len size count limit limits budget usage cost price rate window bucket min max minimum maximum threshold pattern regex regexp format prefix suffix source provider issuer algorithm alg scope scopes policy enabled disabled required optional supported method methods mode strategy location loc position pos index idx offset ids ref reference selector mount volume version ver reporting architecture arch model class cls classification classifier embedding embeddings logits probs weights handler manager service client store storage cache validator generator factory builder parser lexer stream list map set dict schema config settings options status state valid invalid exists present missing check checked verified strength complexity attempts retries hasher hashers encoder backend input form view page screen modal dialog button link route component element style color icon visible hidden shown changed updated created at date time timestamp callback func fn getter lookup resolver loader fetcher filter filters code codes revision annotation annotations counter estimator level levels priority weight ratio penalty bias logprob logprobs',
)

const MAY_NAME_A_CREDENTIAL = /pass|pwd|secret|token|key|cred|auth|pat/i
const TIERS: readonly Tier[] = ['weak', 'normal', 'strong']

// Whether a key names a credential, and how surely; undefined for one that does not (`tokenizer`, `max_tokens`,
// `password_file`, `eos_token`, `CACHE_KEY`).
export function tierOf(key: string): Tier | undefined {
  if (!MAY_NAME_A_CREDENTIAL.test(key)) return undefined
  const words = keyWords(key)
  let tier: Tier | undefined
  let first = words.length
  let last = -1
  const found = (one: Tier, at: number) => {
    if (tier === undefined || TIERS.indexOf(one) > TIERS.indexOf(tier)) tier = one
    first = Math.min(first, at)
    last = Math.max(last, at)
  }
  words.forEach((word, at) => {
    const before = words[at - 1]
    if (/(?:password|passwd|passphrase)$/.test(word)) found('strong', at)
    // `pass` is also the opposite of fail, so its value needs a secret's length.
    else if (word === 'pass') found('normal', at)
    else if (word === 'pwd' && words.length > 1) found('strong', at)
    else if (/secret$/.test(word) || /apikey$/.test(word) || /^cred(?:ential)?s?$/.test(word)) found('normal', at)
    else if (/token$/.test(word) && !TEXT_TOKENS.has(before ?? '')) found('normal', at)
    else if (word === 'key' && KEY_OWNERS.has(before ?? '')) found('normal', at)
    else if (word === 'data' && before === 'key') found('normal', at)
    else if (word === 'auth' || word === 'pat') found('weak', at)
  })
  // An environment variable that ends in KEY: `OPENAI_KEY`, `SIGNING_KEY`, not `CACHE_KEY`.
  if (tier === undefined && /^[A-Z][A-Z0-9_]*_KEY$/.test(key) && !PLAIN_KEYS.has(words.at(-2) ?? '')) found('weak', words.length - 1)
  if (tier === undefined) return undefined
  if (words.slice(0, first).some(word => HEAD_META.has(word))) return undefined
  const tail = words.slice(last + 1)
  if (tail.some(word => TAIL_META.has(word))) return undefined
  // `token_id` and `access_key_id` name a credential; `secret_id` and `password_id` are one.
  if (tail.includes('id') && tier !== 'strong' && !/secret$/.test(words[last])) return undefined
  return tier
}

// ---- Values: which texts are a credential and not a name for one.

const MIN_CHARS: Record<Tier, number> = { strong: 6, normal: 8, weak: 16 }

// A value that stands for another: a shell or template variable, a format field, a placeholder.
const TEMPLATE = /\$\{|\$\(|\{\{|#\{|<%|%\(?\w*\)?[sd]|^\$[A-Za-z_]|\{[A-Za-z_][\w.]*\}|^<[^<>]*>$|^\[[A-Z_]+\]$|^\[vsr:\w+\]$|^`/
const WORDS_NOT_SECRETS = setOf(
  'true false null none nil undefined yes no on off string str bytes required optional include omit same-origin bearer basic digest oauth oauth2 password passwd secret token authorization redacted hidden masked changeme',
)
// A call, an index or a dotted reference (`settings.API_KEY`, `self.token`): what an unquoted value in code is. A
// dotted token (a JWT, `ya29.…`) is not one: its parts hold digits, and it does not start at a name code reads from.
const ROOTS = setOf('self this os process settings config cfg conf env var vars local module data secrets values ctx context args opts options props state request req params')
const isExpression = (value: string) => {
  if (/[([\]]/.test(value)) return true
  if (!/^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+$/.test(value)) return false
  return !/\d/.test(value) || ROOTS.has(value.split('.')[0])
}
const LOOKS_RANDOM = /^(?=.*\d|.*[a-z].*[A-Z]|.*[A-Z].*[a-z])[A-Za-z0-9+/_=.:~-]+$/

const squeezed = (text: string) => text.toLowerCase().replace(/[^a-z0-9]/g, '')

// A sentence about the credential rather than one: a form label, an error message.
function isProse(value: string, key: string) {
  const words = value.trim().split(/\s+/)
  if (words.length < 2 || !words.every(word => /^[\p{L}'’.,!?:;()-]+$/u.test(word))) return false
  const namesIt = keyWords(key).some(word => word.length > 3 && squeezed(value).includes(word)) || MAY_NAME_A_CREDENTIAL.test(value)
  return namesIt || (words.length > 2 && /^\p{Lu}/u.test(words[0]))
}

// Whether a value under a credential key is the credential: long enough, not a variable, a keyword, a key's name
// (`"password": "Password"`, `api_key_env: OPENAI_API_KEY`) or a sentence.
function isSecretValue(value: string, key: string, tier: Tier) {
  if (value.length < MIN_CHARS[tier] || TEMPLATE.test(value) || WORDS_NOT_SECRETS.has(value.toLowerCase())) return false
  if (/^(.)\1*$/.test(value) || /^x+$/i.test(value)) return false
  const echo = squeezed(value)
  if (echo.length >= 4 && squeezed(key).includes(echo)) return false
  // A key's name, written as keys are: `OPENAI_API_KEY`, `x-goog-api-key`, `access_token`.
  if (/^[A-Za-z]+(?:[_-][A-Za-z]+)+$/.test(value) && tierOf(value) !== undefined) return false
  if (isProse(value, key)) return false
  // Dotted lower-case words name a thing (`string.quote`), and so does code caught between two string literals
  // (`"#access_token=" + fragment + "&x="`).
  if (/^[a-z]+(?:\.[a-z]+)+$/.test(value) || /^\s*(?:\+|\.\.?|,|%)\s/.test(value)) return false
  // A token, a secret or a key is never one plain word: `"token": "alternative"` is a model's output. A password may be.
  // Nor words run together (`ModernBertForTokenClassification`).
  if (tier !== 'strong' && (/^\s?[A-Za-z]{1,19}$/.test(value) || /^(?:[A-Z]?[a-z]{2,})+$/.test(value))) return false
  return tier !== 'weak' || LOOKS_RANDOM.test(value)
}

// ---- Rules.

// A whole value, or with `value` the capture group that is the value, the rest of the match (a key, a header) kept.
// Every rule with a `value` carries the `d` flag, so the group's own offsets are known. `key` names the group that
// holds the value's key, for a rule that hides only under a credential key.
type Rule = { pattern: RegExp; value?: number; key?: number }

const NOT_IN_A_WORD = String.raw`(?<![A-Za-z0-9_-])`

// Tokens with a shape of their own, hidden wherever they stand.
const TOKEN_RULES: Rule[] = [
  { pattern: /-----BEGIN [A-Z ]*PRIVATE KEY(?: BLOCK)?-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY(?: BLOCK)?-----/g },
  { pattern: /(?:aws_secret_access_key|AWS_SECRET_ACCESS_KEY)["']?\s*[=:]\s*["']?([A-Za-z0-9/+=]{16,})/dg, value: 1 },
  { pattern: /(?:AKIA|ASIA)[0-9A-Z]{16}/g },
  { pattern: /(?:sk|rk)_(?:live|test)_[0-9A-Za-z]{16,}/g },
  { pattern: new RegExp(String.raw`${NOT_IN_A_WORD}whsec_[0-9A-Za-z]{24,}`, 'g') },
  // GitHub (classic and fine-grained), Anthropic, OpenAI and Slack tokens.
  { pattern: /(?<![A-Za-z0-9])gh[pousr]_[0-9A-Za-z]{36,}/g },
  { pattern: /(?<![A-Za-z0-9])github_pat_[0-9A-Za-z_]{22,}/g },
  { pattern: /(?<![A-Za-z0-9-])sk-(?:ant|proj|svcacct|admin)-[0-9A-Za-z_-]{20,}/g },
  { pattern: /(?<![A-Za-z0-9-])sk-[0-9A-Za-z]{32,}/g },
  { pattern: /(?<![A-Za-z0-9])xox[abeposr]-[0-9A-Za-z-]{10,}/g },
  { pattern: /https:\/\/hooks\.slack\.com\/services\/([0-9A-Za-z/]{20,})/dg, value: 1 },
  // Hugging Face, GitLab, npm, PyPI, Google, SendGrid, Docker, Vault, Databricks, Shopify, DigitalOcean, age.
  { pattern: new RegExp(String.raw`${NOT_IN_A_WORD}hf_[0-9A-Za-z]{30,}`, 'g') },
  { pattern: new RegExp(String.raw`${NOT_IN_A_WORD}glpat-[0-9A-Za-z_-]{20,}`, 'g') },
  { pattern: new RegExp(String.raw`${NOT_IN_A_WORD}npm_[0-9A-Za-z]{36,}`, 'g') },
  { pattern: new RegExp(String.raw`${NOT_IN_A_WORD}pypi-AgE[0-9A-Za-z_-]{20,}`, 'g') },
  { pattern: new RegExp(String.raw`${NOT_IN_A_WORD}AIza[0-9A-Za-z_-]{35}`, 'g') },
  { pattern: new RegExp(String.raw`${NOT_IN_A_WORD}ya29\.[0-9A-Za-z_-]{10,}`, 'g') },
  { pattern: new RegExp(String.raw`${NOT_IN_A_WORD}GOCSPX-[0-9A-Za-z_-]{10,}`, 'g') },
  { pattern: new RegExp(String.raw`${NOT_IN_A_WORD}SG\.[0-9A-Za-z_-]{16,}\.[0-9A-Za-z_-]{16,}`, 'g') },
  { pattern: new RegExp(String.raw`${NOT_IN_A_WORD}dckr_pat_[0-9A-Za-z_-]{20,}`, 'g') },
  { pattern: new RegExp(String.raw`${NOT_IN_A_WORD}hvs\.[0-9A-Za-z_-]{24,}`, 'g') },
  { pattern: new RegExp(String.raw`${NOT_IN_A_WORD}dapi[0-9a-f]{32}`, 'g') },
  { pattern: new RegExp(String.raw`${NOT_IN_A_WORD}shp(?:at|ca|pa|ss)_[0-9A-Fa-f]{32}`, 'g') },
  { pattern: new RegExp(String.raw`${NOT_IN_A_WORD}dop_v1_[0-9a-f]{64}`, 'g') },
  { pattern: /AGE-SECRET-KEY-1[0-9A-Z]{58}/g },
  // A JSON Web Token: three base64url parts, the first two JSON objects.
  { pattern: new RegExp(String.raw`${NOT_IN_A_WORD}eyJ[0-9A-Za-z_-]{8,}\.eyJ[0-9A-Za-z_-]{8,}\.[0-9A-Za-z_-]*`, 'g') },
  // A password hash, as /etc/shadow and .htpasswd keep it: crypt, bcrypt, sha-256, sha-512, apr1, yescrypt, argon2.
  { pattern: /\$(?:1|2[abxy]?|5|6|7|y|apr1|argon2(?:id|i|d))\$[A-Za-z0-9./$=,+-]{20,}/g },
  // An Authorization header's credentials, after its scheme or, a long one, with none.
  { pattern: /\bauthorization["']?[ \t]*[:=][ \t]*["']?(?:bearer|basic|token|bot|digest|apikey)[ \t]+([A-Za-z0-9._~+/=-]{8,})/dgi, value: 1 },
  { pattern: /\bauthorization["']?[ \t]*:[ \t]*["']?(?!(?:bearer|basic|token|bot|digest|apikey)\b)([A-Za-z0-9._~+/=-]{16,})/dgi, value: 1 },
  { pattern: /\b(?:x-)?api-key[ \t]*:[ \t]*["']?([A-Za-z0-9._~+/=-]{16,})/dgi, value: 1 },
]

// Values found by where they stand, each hidden unless it is a variable or a placeholder (`{password}`, `$PASS`).
const PLACED_RULES: Rule[] = [
  // A password in a connection URL: `postgres://user:hunter2@db`.
  { pattern: /\b[a-z][a-z0-9+.-]{0,19}:\/\/[^\s:/@]+:([^\s:/@]+)@/dgi, value: 1 },
  // A credential in a URL's query.
  { pattern: /[?&](?:api[_-]?key|access[_-]?token|auth[_-]?token|id[_-]?token|token|password|passwd|pwd|secret|client[_-]?secret|sig|signature)=([^\s&#"'<>]{8,})/dgi, value: 1 },
  // Passwords given on a command line: `sshpass -p`, `mysql -p` with the password attached, `docker login -p`, `curl -u user:pass`.
  { pattern: /\bsshpass[ \t]+-p[ \t]*["']?([^\s"']{4,})/dg, value: 1 },
  { pattern: /\b(?:mysql|mysqldump|mysqladmin|mariadb)\b[^\n|;&]*?[ \t]-p([^\s"'`-][^\s"'`]{3,})/dg, value: 1 },
  { pattern: /\b(?:docker|podman|helm|oras|skopeo|crane|npm|az|gh)\b[^\n|;&]*?\blogin\b[^\n|;&]*?[ \t]-p[ \t=]*["']?([^\s"']{4,})/dg, value: 1 },
  { pattern: /\b(?:curl|wget|http|https)\b[^\n|;&]*?[ \t](?:-u|--user|--proxy-user)[ \t=]*["']?[^\s:"']+:([^\s"']{4,})/dg, value: 1 },
  // A cookie header's cookies.
  { pattern: /\b(?:set-)?cookie["']?[ \t]*:[ \t]*["']?([^\s"'=;]+=[^\n"']{8,})/dgi, value: 1 },
  // A .netrc entry on one line.
  { pattern: /\blogin[ \t]+\S+[ \t]+password[ \t]+(\S{4,})/dg, value: 1 },
]

// A .netrc entry over several lines: `password` on its own line, in a text that names a machine and a login.
const NETRC_PASSWORD = /^[ \t]*password[ \t]+(\S{4,})[ \t]*$/dgm
const IS_NETRC = /\bmachine[ \t]+\S+[\s\S]*\blogin[ \t]+\S+|\blogin[ \t]+\S+[\s\S]*\bmachine[ \t]+\S+/

// Values hidden only under a key that names a credential, in the shapes that are not `key = value`.
const KEYED_RULES: Rule[] = [
  // A command-line flag: `--password hunter2`, `--token=abc`.
  { pattern: /(?<![A-Za-z0-9_-])--?([A-Za-z][A-Za-z0-9-]{3,})(?:=|[ \t]+)(?:"([^"\n]*)"|'([^'\n]*)'|([^\s"'`=-][^\s"'`]*))/dg, key: 1, value: 2 },
  // XML: `<password>hunter2</password>`, and `key="ApiKey" value="abc"`.
  { pattern: /<([A-Za-z_][\w.-]*)>([^<>\n]+)<\/\1>/dg, key: 1, value: 2 },
  { pattern: /\b(?:key|name)[ \t]*=[ \t]*["']([^"'\n]+)["'][ \t]+value[ \t]*=[ \t]*["']([^"'\n]+)["']/dgi, key: 1, value: 2 },
  // A call that sets a named value: `define('DB_PASSWORD', 'x')`, `setenv("API_KEY", "x")`, `headers.set("X-Api-Key", "x")`.
  { pattern: /\b(?:define|setenv|putenv|setdefault|setProperty|setRequestHeader|set|put|add|header|env)\(\s*["']([A-Za-z_][\w.-]*)["']\s*,\s*["']([^"'\n]+)["']/dg, key: 1, value: 2 },
  // An assignment to a named entry: `os.environ["API_KEY"] = "x"`, `$_ENV['DB_PASSWORD'] = 'x'`.
  { pattern: /\[["']([A-Za-z_][\w.-]*)["']\][ \t]*=[ \t]*["']([^"'\n]+)["']/dg, key: 1, value: 2 },
]

// `key = value`, `key: value`, `"key": "value"`, `key := value`, `key => value`, anywhere on a line: the key (group
// 2, in the quote of group 1), an optional type between them (`password: str = "x"`), the separator with the space
// around it (groups 3 to 5), and the value in double quotes, in single quotes, or bare (groups 6 to 8).
const ASSIGNMENT = new RegExp(
  String.raw`(?<![A-Za-z0-9_.-])(["']?)([A-Za-z_][A-Za-z0-9_.-]*)\1(?:[ \t]*:[ \t]*[A-Za-z_][\w.\[\]|, ]{0,40}?(?=[ \t]*=(?![=>~])))?` +
    String.raw`([ \t]*)(=>|:=|=(?![=>~])|:(?!:))([ \t]*)` +
    String.raw`(?:"((?:[^"\\\n]|\\.)*)"|'((?:[^'\\\n]|\\.)*)'|([^\s"'\x60]+))`,
  'dg',
)

// Targets before this one on its line make the value one of several (`email, password = "a", "b"`): not paired.
const SEVERAL_TARGETS = /^[ \t]*(?:(?:const|var|let|final)[ \t]+)?(?:[A-Za-z_]\w*[ \t]*,[ \t]*)+$/
// What may follow the last value on a line: nothing, punctuation that closes it, or a comment.
const ENDS_THE_LINE = /^[ \t]*[,;)\]}]*[ \t]*(?:(?:#|\/\/).*)?$/

function assignmentsIn(text: string, kind: Kind): Found[] {
  const found: Found[] = []
  ASSIGNMENT.lastIndex = 0
  for (let match = ASSIGNMENT.exec(text); match !== null; match = ASSIGNMENT.exec(text)) {
    const key = match[2]
    const [keyAt] = match.indices?.[2] ?? [match.index]
    const quoted = match[6] !== undefined ? 6 : match[7] !== undefined ? 7 : undefined
    const [valueAt, valueEnd] = match.indices?.[quoted ?? 8] ?? [0, 0]
    // The value is read again as text of its own: it may hold an assignment (`CMD="run --x" TOKEN=…`, a URL's query).
    const tier = tierOf(key)
    const span = tier === undefined ? undefined : spanOf(text, kind, { key, keyAt, tier, isQuoted: quoted !== undefined, valueAt, valueEnd, separator: match[4], isSpaced: match[3] !== '' || match[5] !== '' })
    if (span === undefined) ASSIGNMENT.lastIndex = Math.max(valueAt, keyAt + 1)
    else {
      found.push({ from: span[0], to: span[1], tag: 'secret' })
      ASSIGNMENT.lastIndex = Math.max(span[1], ASSIGNMENT.lastIndex)
    }
  }
  return found
}

type Assigned = { key: string; keyAt: number; tier: Tier; isQuoted: boolean; valueAt: number; valueEnd: number; separator: string; isSpaced: boolean }

// The part of an assignment's value to hide, or undefined when it is not a credential.
function spanOf(text: string, kind: Kind, { key, keyAt, tier, isQuoted, valueAt, valueEnd, separator, isSpaced }: Assigned): [number, number] | undefined {
  const lineAt = text.lastIndexOf('\n', keyAt - 1) + 1
  const before = text.slice(lineAt, keyAt).replace(/["']$/, '')
  // `${TOKEN:-default}` and `$TOKEN` read a variable; `$password = 'x'` (PHP) sets one.
  if (/\$\{?$/.test(before) && !(isQuoted && separator === '=' && !before.endsWith('{'))) return undefined
  if (separator !== ':' && SEVERAL_TARGETS.test(before)) return undefined
  if (isQuoted) return isSecretValue(text.slice(valueAt, valueEnd), key, tier) ? [valueAt, valueEnd] : undefined
  if (kind === 'code') return undefined
  const lineEnd = text.indexOf('\n', valueAt)
  const rest = text.slice(valueEnd, lineEnd < 0 ? text.length : lineEnd)
  if (kind === 'config') {
    // A config file's value runs to the end of its line, or to a comment.
    const line = text.slice(valueAt, lineEnd < 0 ? text.length : lineEnd).replace(/[ \t]+#.*$/, '').trimEnd()
    // Not a list, a block, a tag or an alias, not a call (a template's code), and not a sentence in a comment.
    if (/^[[{!|>&*]/.test(line) || /^[A-Za-z_][\w.:]*\(/.test(line) || (/^[ \t]*(?:#|\/\/|;)/.test(before) && /\s/.test(line))) return undefined
    if (!isSecretValue(line, key, tier)) return undefined
    return [valueAt, valueAt + line.length]
  }
  // In a URL's query the value ends at the next parameter.
  const inQuery = /[?&]$/.test(before)
  const value = text.slice(valueAt, valueEnd).replace(inQuery ? /[&#].*$/ : /[,;)\]}]+$/, '')
  if (/^[[{(]/.test(value) || isExpression(value) || !isSecretValue(value, key, tier)) return undefined
  // `KEY=value` as a shell or an env file writes it; a lower-case key with a plain name for a value is a keyword
  // argument (`token=access_token`).
  if (!isSpaced && separator === '=') return /^[A-Z0-9_]+$/.test(key) || !/^[A-Za-z_]+$/.test(value) ? [valueAt, valueAt + value.length] : undefined
  // Spaced or after a colon, a bare value must end its line and hold a letter and a digit, unlike a type or a count.
  // After a spaced `=`, a lower-case name with underscores is a variable (`access_token = oauth2_scheme`).
  if (separator !== ':' && /^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$/.test(value)) return undefined
  return ENDS_THE_LINE.test(rest) && /[A-Za-z]/.test(value) && /\d/.test(value) ? [valueAt, valueAt + value.length] : undefined
}

const spansOfRule = (text: string, { pattern, value }: Rule) =>
  [...text.matchAll(pattern)].flatMap(match => {
    const [from, to] = value === undefined ? [match.index ?? 0, (match.index ?? 0) + match[0].length] : (match.indices?.[value] ?? [])
    return from === undefined || to === undefined ? [] : [{ from, to, text: text.slice(from, to), match }]
  })

// A keyed rule's value: the first of the groups from `value` on that matched (a quoted or a bare one).
function keyedIn(text: string, { pattern, key, value }: Rule): Found[] {
  return [...text.matchAll(pattern)].flatMap(match => {
    const groups = match.indices ?? []
    const at = groups.findIndex((span, group) => group >= (value ?? 0) && span !== undefined)
    const tier = tierOf(match[key ?? 0] ?? '')
    if (at < 0 || tier === undefined) return []
    const [from, to] = groups[at]
    const hidden = text.slice(from, to).replace(/[,;)\]}]+$/, '')
    // A flag's `id=…,src=…` names a secret; it is not one.
    if (/^(?:id|src|source|target|type)=/.test(hidden) || isExpression(hidden) || !isSecretValue(hidden, match[key ?? 0], tier)) return []
    return [{ from, to: from + hidden.length, tag: 'secret' as const }]
  })
}

// The secrets the rules find in a text, with no model: `kind` says how a value with no quotes is read.
export function secretsIn(text: string, kind: Kind = 'unknown'): Found[] {
  const secret = ({ from, to }: { from: number; to: number }): Found => ({ from, to, tag: 'secret' })
  const placed = PLACED_RULES.flatMap(rule => spansOfRule(text, rule)).filter(one => !TEMPLATE.test(one.text))
  const netrc = IS_NETRC.test(text) ? spansOfRule(text, { pattern: NETRC_PASSWORD, value: 1 }) : []
  return [
    ...TOKEN_RULES.flatMap(rule => spansOfRule(text, rule)).map(secret),
    ...[...placed, ...netrc].map(secret),
    ...KEYED_RULES.flatMap(rule => keyedIn(text, rule)),
    ...assignmentsIn(text, kind),
  ]
}

const WORD = /[A-Za-z0-9]/

// An IPv4 address (with an optional /prefix or :port) or an IPv6 one. The model also tags version numbers such as
// the 2.32 in requests==2.32.3 as IP_ADDRESS, and hiding those breaks the files Claude reads.
const OCTET = '(?:25[0-5]|2[0-4]\\d|1\\d\\d|[1-9]?\\d)'
const IPV4 = new RegExp(`^${OCTET}(?:\\.${OCTET}){3}(?:/\\d{1,2})?(?::\\d{1,5})?$`)
const IPV6 = /^\[?[0-9A-Fa-f]{0,4}(?::[0-9A-Fa-f]{0,4}){2,7}(?:%\w+)?\]?(?::\d{1,5})?$/
export const looksLikeIp = (text: string) => IPV4.test(text) || (text.includes(':') && IPV6.test(text))

// The dotted number around [from, to), without dots at its ends. The model cuts such numbers short (it tags the 2.32
// of 2.32.3), so a span over part of an IPv4 address is checked, and hidden, as the whole address.
function dottedAround(text: string, from: number, to: number): [number, number] {
  while (from > 0 && /[0-9.]/.test(text[from - 1])) from--
  while (to < text.length && /[0-9.]/.test(text[to])) to++
  while (from < to && text[from] === '.') from++
  while (to > from && text[to - 1] === '.') to--
  return [from, to]
}

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
    if (tag === 'ip' && !looksLikeIp(text.slice(from, to))) {
      const [start, end] = dottedAround(text, from, to)
      if (!IPV4.test(text.slice(start, end))) return []
      from = Math.min(from, start)
      to = Math.max(to, end)
    }
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
