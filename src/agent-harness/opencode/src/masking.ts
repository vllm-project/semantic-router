// Secret rules and label policy retained from Claude revision 4021f944f.
import { isRecord } from "./settings.ts";
import type { Reply } from "./runtime.ts";
export const TAGS = [
  "name",
  "email",
  "phone",
  "card",
  "ip",
  "ssn",
  "iban",
  "address",
  "id",
  "secret",
] as const;
export type Tag = (typeof TAGS)[number];
// A tag in a placeholder: one of TAGS, or a listener rule's name, which the Router does not restrict beyond this.
export const isTagName = (value: string) => /^[^\]\s]{1,128}$/.test(value);
// A PII or secret tag, or the name of a listener rule that found the text.
export type Found = {
  from: number;
  to: number;
  tag: string;
  probability?: number;
};
export type Span = {
  label: string;
  start: number;
  end: number;
  text: string;
  probability: number;
};
export const PII_QUESTIONS = { pii: { preset: "pii" } };
// Dates and domain names remain useful source data; they are not masked by this policy.
export const LABEL_TAGS: Record<string, Tag> = {
  PERSON: "name",
  EMAIL_ADDRESS: "email",
  PHONE_NUMBER: "phone",
  CREDIT_CARD: "card",
  IP_ADDRESS: "ip",
  US_SSN: "ssn",
  IBAN_CODE: "iban",
  LOCATION: "address",
  US_DRIVER_LICENSE: "id",
  US_PASSPORT: "id",
};
export function probability(value: unknown): value is number {
  return (
    typeof value === "number" &&
    Number.isFinite(value) &&
    value >= 0 &&
    value <= 1
  );
}
// A malformed span makes the coverage unavailable, rather than silently disappearing.
export function parsePii(reply: Reply, text: string): Span[] | undefined {
  if (
    !isRecord(reply.answers.pii) ||
    !probability(reply.answers.pii.noul) ||
    !Array.isArray(reply.spans.pii)
  )
    return;
  const points = Array.from(text);
  const spans: Span[] = [];
  for (const span of reply.spans.pii) {
    if (
      !isRecord(span) ||
      typeof span.label !== "string" ||
      typeof span.text !== "string" ||
      !probability(span.probability) ||
      typeof span.start !== "number" ||
      typeof span.end !== "number" ||
      !Number.isInteger(span.start) ||
      !Number.isInteger(span.end) ||
      span.start < 0 ||
      span.end <= span.start ||
      span.end > points.length ||
      points.slice(span.start, span.end).join("") !== span.text
    )
      return;
    spans.push({
      label: span.label,
      text: span.text,
      probability: span.probability,
      start: span.start,
      end: span.end,
    });
  }
  return spans;
}
// A key that names a credential (`token`, `api_key`, `client_secret`, `db_password`, `access_token`), the keyword
// ending the key or followed by `_`, `-` or `.`: not `tokens`, `tokenizer` or `secretary`. `pass` only as a word of
// its own (`db_pass`), not in `bypass` or `passport`.
const KEY = String.raw`[A-Za-z0-9_.-]*?(?:api[_-]?key|apikey|secret|token|password|passwd|(?<![A-Za-z0-9])pass(?![A-Za-z0-9])|credentials?|private[_-]?key|access[_-]?key)(?:[_.-][A-Za-z0-9_.-]*)?`;
// A value that names another value rather than holding one: `settings.API_KEY`, `process.env.TOKEN`, a call
// (`getpass.getpass()`, `get_token()`) or a shell variable (`$TOKEN`).
const IDENTIFIER = String.raw`[A-Za-z_][A-Za-z0-9_]*`;
const REFERENCE = String.raw`(?:${IDENTIFIER}(?:\.${IDENTIFIER})*\([^\s]*\)|${IDENTIFIER}(?:\.${IDENTIFIER})+|\$\{?${IDENTIFIER}\}?)(?:[ \t;]|$)`;
// The characters of a token or key: no spaces, quotes, brackets or calls.
const VALUE_CHAR = String.raw`[A-Za-z0-9._~+/=:@%!*^#-]`;

// A whole value, or with `value` the capture group that is the value, the rest of the match (a key, a header) kept.
// Every rule with a `value` carries the `d` flag, so the group's own offsets are known.
const SECRET_RULES: { pattern: RegExp; value?: number }[] = [
  {
    pattern:
      /-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----/g,
  },
  {
    pattern:
      /(?:aws_secret_access_key|AWS_SECRET_ACCESS_KEY)["']?\s*[=:]\s*["']?([A-Za-z0-9/+=]{16,})/dg,
    value: 1,
  },
  { pattern: /AKIA[0-9A-Z]{16}/g },
  { pattern: /sk_(?:live|test)_[0-9A-Za-z]{16,}/g },
  // GitHub (classic and fine-grained), Anthropic, OpenAI and Slack tokens, wherever they stand.
  { pattern: /(?<![A-Za-z0-9])gh[pousr]_[0-9A-Za-z]{36,}/g },
  { pattern: /(?<![A-Za-z0-9])github_pat_[0-9A-Za-z_]{22,}/g },
  {
    pattern:
      /(?<![A-Za-z0-9-])sk-(?:ant|proj|svcacct|admin)-[0-9A-Za-z_-]{20,}/g,
  },
  { pattern: /(?<![A-Za-z0-9])xox[abposr]-[0-9A-Za-z-]{10,}/g },
  // An Authorization header's credentials, after its scheme or, a long one, with none.
  {
    pattern:
      /\bauthorization["']?[ \t]*[:=][ \t]*["']?(?:bearer|basic|token|bot|digest|apikey)[ \t]+([A-Za-z0-9._~+/=-]{8,})/dgi,
    value: 1,
  },
  {
    pattern:
      /\bauthorization["']?[ \t]*:[ \t]*["']?(?!(?:bearer|basic|token|bot|digest|apikey)\b)([A-Za-z0-9._~+/=-]{16,})/dgi,
    value: 1,
  },
  {
    pattern: /\b(?:x-)?api-key[ \t]*:[ \t]*["']?([A-Za-z0-9._~+/=-]{16,})/dgi,
    value: 1,
  },
  // A password in a connection URL: `postgres://user:hunter2@db`.
  { pattern: /\b[a-z][a-z0-9+.-]*:\/\/[^\s:/@]+:([^\s:/@]+)@/dgi, value: 1 },
  // Config-style assignments only: a bare `password:` also opens a Python block, and hiding what
  // followed it broke code Claude read in a live run. The value ends the word and is not code: not
  // `os.environ["SECRET_KEY"]`, `settings.API_KEY` or `$TOKEN` (a quoted value is the rules' below).
  {
    pattern: new RegExp(
      String.raw`^[ \t]*(?:export[ \t]+)?[A-Z0-9_]*(?:API_?KEY|SECRET|TOKEN|PASSWORD|PASSWD|(?<![A-Z0-9])PASS(?![A-Z0-9]))[A-Z0-9_]*[ \t]*=[ \t]*(?!${REFERENCE})([^\s"'\x60]{8,})(?=[ \t;]|$)`,
      "dgm",
    ),
    value: 1,
  },
  // The same in any case, shell style (no spaces around `=`), the value ending the word: `db_password=…`, not `password_hint=hint()`.
  {
    pattern: new RegExp(
      String.raw`^[ \t]*(?:export[ \t]+)?${KEY}=["']?(${VALUE_CHAR}{8,})(?=["']?(?:[ \t;]|$))`,
      "dgim",
    ),
    value: 1,
  },
  // INI and .cfg style (`api_key = …`), held to the YAML rule's bar below, so `api_key = settings.api_key` stays.
  {
    pattern: new RegExp(
      String.raw`^[ \t]*${KEY}[ \t]+=[ \t]+(?=[^\s]*[A-Za-z])(?=[^\s]*\d)(${VALUE_CHAR}{8,})[ \t]*$`,
      "dgim",
    ),
    value: 1,
  },
  // A quoted key with a quoted value: JSON, YAML, a Python dict.
  {
    pattern: new RegExp(
      String.raw`["']${KEY}["'][ \t]*:[ \t]*["']([^"'\n]{8,})`,
      "dgi",
    ),
    value: 1,
  },
  // A bare key with a quoted value of one word: YAML, TOML, an assignment in code.
  {
    pattern: new RegExp(
      String.raw`(?<![A-Za-z0-9_.$-])${KEY}[ \t]*[:=][ \t]*(["'])(${VALUE_CHAR}{8,})\1`,
      "dgi",
    ),
    value: 2,
  },
  // A bare YAML key with a bare value that ends the line and holds a letter and a digit, unlike a type or a count.
  {
    pattern: new RegExp(
      String.raw`^[ \t]*(?:-[ \t]+)?${KEY}[ \t]*:[ \t]+(?=[^\s]*[A-Za-z])(?=[^\s]*\d)(${VALUE_CHAR}{8,})[ \t]*$`,
      "dgim",
    ),
    value: 1,
  },
];

function rawSecrets(text: string): Found[] {
  return SECRET_RULES.flatMap(({ pattern, value }) =>
    [...text.matchAll(pattern)].flatMap((match) => {
      const [from, to] =
        value === undefined
          ? [match.index ?? 0, (match.index ?? 0) + match[0].length]
          : (match.indices?.[value] ?? []);
      return from === undefined || to === undefined
        ? []
        : [{ from, to, tag: "secret" as const }];
    }),
  );
}

export function secretsIn(text: string): Found[] {
  const found = rawSecrets(text);
  // Native read renders line numbers before source text. Anchored assignment rules
  // must still see the source line; findings retain offsets into the rendered output.
  for (const match of text.matchAll(/^(\s*\d+[:|\t][ \t]*)([^\n]*)/gm)) {
    const offset = match.index! + match[1]!.length;
    found.push(
      ...rawSecrets(match[2]!).map((item) => ({
        ...item,
        from: item.from + offset,
        to: item.to + offset,
      })),
    );
  }
  return found;
}

const WORD = /[A-Za-z0-9]/;

// An IPv4 address (with an optional /prefix or :port) or an IPv6 one. The model also tags version numbers such as
// the 2.32 in requests==2.32.3 as IP_ADDRESS, and hiding those breaks the files Claude reads.
const OCTET = "(?:25[0-5]|2[0-4]\\d|1\\d\\d|[1-9]?\\d)";
const IPV4 = new RegExp(
  `^${OCTET}(?:\\.${OCTET}){3}(?:/\\d{1,2})?(?::\\d{1,5})?$`,
);
const IPV6 =
  /^\[?[0-9A-Fa-f]{0,4}(?::[0-9A-Fa-f]{0,4}){2,7}(?:%\w+)?\]?(?::\d{1,5})?$/;
export const looksLikeIp = (text: string) =>
  IPV4.test(text) || (text.includes(":") && IPV6.test(text));

// The dotted number around [from, to), without dots at its ends. The model cuts such numbers short (it tags the 2.32
// of 2.32.3), so a span over part of an IPv4 address is checked, and hidden, as the whole address.
function dottedAround(
  text: string,
  from: number,
  to: number,
): [number, number] {
  while (from > 0 && /[0-9.]/.test(text[from - 1]!)) from--;
  while (to < text.length && /[0-9.]/.test(text[to]!)) to++;
  while (from < to && text[from] === ".") from++;
  while (to > from && text[to - 1] === ".") to--;
  return [from, to];
}

// The spans at or above `line`, of the labels that are personal data (a DOMAIN_NAME is not), in UTF-16 offsets of `text`; a span that stops inside a word runs to the word's end.
export function piiIn(
  text: string,
  spans: readonly Span[],
  line: number,
): Found[] {
  if (!spans.length) return [];
  const offsets: number[] = [];
  let at = 0;
  for (const char of text) {
    offsets.push(at);
    at += char.length;
  }
  offsets.push(at);
  return spans.flatMap((span) => {
    const tag = LABEL_TAGS[span.label];
    if (tag === undefined || span.probability < line) return [];
    let from = offsets[span.start];
    let to = offsets[span.end];
    if (
      from === undefined ||
      to === undefined ||
      text.slice(from, to) !== span.text ||
      span.text === ""
    )
      return [];
    while (from > 0 && WORD.test(text[from - 1]!) && WORD.test(text[from]!))
      from--;
    while (to < text.length && WORD.test(text[to]!) && WORD.test(text[to - 1]!))
      to++;
    if (tag === "ip" && !looksLikeIp(text.slice(from, to))) {
      const [start, end] = dottedAround(text, from, to);
      if (!IPV4.test(text.slice(start, end))) return [];
      from = Math.min(from, start);
      to = Math.max(to, end);
    }
    return [{ from, to, tag, probability: span.probability }];
  });
}

export type Counts = Partial<Record<string, number>>;
export type Hidden = { text: string; counts: Counts };

// What stands where a value was hidden: `[vsr:email]`, a tag no code holds, so writing one back is always a lost value.
// The Router's masking plugin leaves text of this shape alone (placeholderPattern in pkg/masking), so keep it.
export const tagText = (tag: string) => `[vsr:${tag}]`;

// Union overlapping intervals so an intersecting span cannot leave a sensitive suffix exposed.
export function hide(text: string, found: readonly Found[]): Hidden {
  const merged: Found[] = [];
  for (const item of [...found].sort(
    (a, b) => a.from - b.from || b.to - a.to,
  )) {
    if (
      !Number.isInteger(item.from) ||
      !Number.isInteger(item.to) ||
      item.from < 0 ||
      item.to <= item.from ||
      item.to > text.length
    )
      throw new Error("Invalid masking interval");
    const prior = merged.at(-1);
    if (prior && item.from < prior.to) {
      prior.to = Math.max(prior.to, item.to);
      if (item.tag === "secret") prior.tag = "secret";
    } else merged.push({ ...item });
  }
  const counts: Counts = {};
  let out = "",
    at = 0;
  for (const item of merged) {
    out += text.slice(at, item.from) + tagText(item.tag);
    counts[item.tag] = (counts[item.tag] ?? 0) + 1;
    at = item.to;
  }
  return { text: out + text.slice(at), counts };
}

export function addCounts(...all: Counts[]): Counts {
  const sum: Counts = {};
  for (const counts of all) {
    for (const [tag, n] of Object.entries(counts) as [string, number][])
      sum[tag] = (sum[tag] ?? 0) + n;
  }
  return sum;
}
