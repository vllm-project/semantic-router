import type { Risk } from '../../types'

// Bash read the way it runs, for the guard: a command split into the commands it runs (partsOf), and the fixed
// rules that hold one whatever the classifier says (alwaysHeld). Pure functions, no engine calls.

// `<<WORD`, `<<-WORD`, `<<'WORD'`, `<<"WORD"` or `<<\WORD`: a heredoc whose body runs to a line of WORD. Quoted,
// its body is plain text; unquoted, bash expands the `$(…)` and backticks in it.
const HEREDOC = /^<<(-?)[ \t]*(\\?)(['"]?)([A-Za-z0-9_.-]+)\3/

type Pending = { word: string; isTabbed: boolean; isQuoted: boolean; at: number }
// A heredoc at the top: where its `<<` stands, its body, and the index just past the line that ends it.
type Heredoc = { at: number; body: Span; past: number; isQuoted: boolean }

// Each heredoc's body, read from `at`, and the index just past the line that ends the last; the end of the command
// when one never ends (bash runs it so, the body running to the end).
function pastHeredocs(command: string, at: number, pending: readonly Pending[]) {
  const read: Heredoc[] = []
  for (const { word, isTabbed, isQuoted, at: marker } of pending) {
    const from = at
    let to = command.length
    while (at < command.length) {
      const end = command.indexOf('\n', at)
      const line = command.slice(at, end < 0 ? command.length : end)
      const start = at
      at = end < 0 ? command.length : end + 1
      if ((isTabbed ? line.replace(/^\t+/, '') : line) === word) {
        to = start
        break
      }
    }
    read.push({ at: marker, body: [from, to], past: at, isQuoted })
  }
  return { past: at, read }
}

// What the scanner stands inside: a quote (`'…'`, `$'…'`, `"…"`), a command substitution (`$(…)`, backticks), a
// subshell, a process substitution (`<(…)`, `>(…)`), a `${…}`, or arithmetic (`$((…))`, `((…))`).
type FrameKind = "'" | "$'" | '"' | '`' | '$(' | '(' | '<(' | '${' | '$((' | '(('
type Frame = { kind: FrameKind; from: number; body: number; parens: number }

// Where commands run, so a separator, a comment or a heredoc means what it does at the top.
const COMMAND_FRAMES: readonly FrameKind[] = ['`', '$(', '(', '<(']

type Span = [from: number, to: number]

type Scan = {
  // Separators that start a command of their own: `&&`, `||`, `;`, a lone `&`, a newline no pipe continues.
  cuts: Span[]
  // `|` and `|&` between the stages of a pipe.
  pipes: Span[]
  comments: Span[]
  // The quotes, substitutions and subshells at the top, each from its opener to just past its closer.
  regions: Span[]
  regionKinds: FrameKind[]
  // The text of each outermost substitution or subshell: commands that run too.
  bodies: string[]
  heredocs: Heredoc[]
  // False when a quote, substitution or subshell never closes: the split cannot be trusted.
  isSure: boolean
}

// The opener at `i`, if one starts there, and how wide it is. Inside double quotes only expansions open; at a
// command's level also subshells, process substitutions and `((`.
function openerAt(text: string, i: number, at: 'command' | 'double' | 'other'): [FrameKind, number] | undefined {
  if (text.startsWith('$((', i)) return ['$((', 3]
  if (text.startsWith('$(', i)) return ['$(', 2]
  if (text.startsWith('${', i)) return ['${', 2]
  if (text[i] === '`') return ['`', 1]
  if (at === 'double') return undefined
  if (text.startsWith("$'", i)) return ["$'", 2]
  if (text.startsWith('$"', i)) return ['"', 2]
  if (text[i] === "'" || text[i] === '"') return [text[i] as FrameKind, 1]
  if (at === 'other') return undefined
  if (text.startsWith('<(', i) || text.startsWith('>(', i)) return ['<(', 2]
  if (text.startsWith('((', i) && isWordStart(text, i)) return ['((', 2]
  if (text[i] === '(') return ['(', 1]
  return undefined
}

const isWordStart = (text: string, i: number) => i === 0 || /[\s;&|()]/.test(text[i - 1])

// Reads a shell line as bash does, far enough to know where its commands, stages, comments and quotes are.
function scan(command: string): Scan {
  const found: Scan = { cuts: [], pipes: [], comments: [], regions: [], regionKinds: [], bodies: [], heredocs: [], isSure: true }
  const stack: Frame[] = []
  const heredocs: Pending[] = []
  const open = (kind: FrameKind, i: number, width: number) => {
    if (stack.length === 0) found.regionKinds.push(kind)
    stack.push({ kind, from: i, body: i + width, parens: 0 })
    return i + width - 1
  }
  const close = (i: number, width = 1) => {
    const frame = stack.pop()!
    if ((COMMAND_FRAMES.includes(frame.kind) || frame.kind === '((') && !stack.some(one => COMMAND_FRAMES.includes(one.kind))) {
      found.bodies.push(command.slice(frame.body, i))
    }
    if (stack.length === 0) found.regions.push([frame.from, i + width])
    return i + width - 1
  }
  for (let i = 0; i < command.length; i++) {
    const c = command[i]
    const top = stack.at(-1)
    if (top?.kind === "'") {
      if (c === "'") close(i)
      continue
    }
    if (c === '\\') {
      i++
      continue
    }
    if (top?.kind === "$'") {
      if (c === "'") close(i)
      continue
    }
    if (top?.kind === '"' || top?.kind === '${') {
      const closer = top.kind === '"' ? '"' : '}'
      if (c === closer) close(i)
      else {
        const opener = openerAt(command, i, top.kind === '"' ? 'double' : 'other')
        if (opener) i = open(opener[0], i, opener[1])
      }
      continue
    }
    if (top?.kind === '$((' || top?.kind === '((') {
      if (c === ')' && top.parens === 0) i = close(i, command[i + 1] === ')' ? 2 : 1)
      else if (c === ')') top.parens--
      else if (c === '(') top.parens++
      else {
        const opener = openerAt(command, i, 'other')
        if (opener) i = open(opener[0], i, opener[1])
      }
      continue
    }
    // At a command's level: the top, or inside `$(…)`, `(…)`, `<(…)` or backticks.
    if (c === '`' && top?.kind === '`') {
      close(i)
      continue
    }
    if (c === ')' && (top?.kind === '$(' || top?.kind === '(' || top?.kind === '<(')) {
      close(i)
      continue
    }
    const opener = openerAt(command, i, 'command')
    if (opener) {
      i = open(opener[0], i, opener[1])
      continue
    }
    if (c === '#' && isWordStart(command, i)) {
      const end = command.indexOf('\n', i)
      if (stack.length === 0) found.comments.push([i, end < 0 ? command.length : end])
      i = (end < 0 ? command.length : end) - 1
    } else if (command.startsWith('<<<', i)) i += 2
    else if (command.startsWith('<<', i)) {
      const marker = HEREDOC.exec(command.slice(i))
      if (marker !== null) heredocs.push({ word: marker[4], isTabbed: marker[1] === '-', isQuoted: marker[2] !== '' || marker[3] !== '', at: i })
      i += (marker?.[0].length ?? 2) - 1
    } else if (c === '\n' && heredocs.length > 0) {
      const { past, read } = pastHeredocs(command, i + 1, heredocs.splice(0))
      if (stack.length === 0) found.heredocs.push(...read)
      if (stack.length === 0 && command[past - 1] === '\n') found.cuts.push([past - 1, past])
      i = past - 1
    } else if (stack.length === 0) {
      const width = separatorWidth(command, i)
      if (width > 0) found.cuts.push([i, i + width])
      else if (c === '|' && command[i - 1] !== '>') {
        const pipe = command[i + 1] === '&' ? 2 : 1
        found.pipes.push([i, i + pipe])
        i += pipe - 1
      }
      i += Math.max(0, width - 1)
    }
  }
  found.isSure = stack.length === 0
  while (stack.length > 0) close(command.length, 0)
  return found
}

// How wide the separator at `i` is, or 0: `&` beside a redirection (`2>&1`, `&>`) or after a pipe (`|&`) is not one,
// nor is a newline after a pipe that runs on to the next line.
function separatorWidth(command: string, i: number) {
  if (command.startsWith('&&', i) || command.startsWith('||', i)) return 2
  const c = command[i]
  const before = command[i - 1] ?? ''
  if (c === ';') return 1
  if (c === '&') return /[<>|]/.test(before) || command[i + 1] === '>' ? 0 : 1
  if (c !== '\n') return 0
  const line = command.slice(0, i).trimEnd()
  return line.endsWith('|') && !line.endsWith('||') ? 0 : 1
}

// The text between the cuts, each span trimmed, the empty ones left out.
function piecesOf(text: string, cuts: readonly Span[]): Span[] {
  const pieces: Span[] = []
  let from = 0
  for (const [cutFrom, cutTo] of [...cuts, [text.length, text.length] as const]) {
    const piece = text.slice(from, cutFrom)
    const start = from + (piece.length - piece.trimStart().length)
    const end = cutFrom - (piece.length - piece.trimEnd().length)
    if (end > start) pieces.push([start, end])
    from = cutTo
  }
  return pieces
}

// The commands a shell line runs, in order: a pipe stays whole, and a part that is only a comment is left out.
// A line the scanner cannot read through (a quote or a substitution that never closes) stays one part, whole.
export function partsOf(command: string): string[] {
  const { cuts, comments, isSure } = scan(command)
  if (!isSure) return command.trim() === '' ? [] : [command.trim()]
  const isComment = (at: number) => comments.some(([from, to]) => at >= from && at < to)
  return piecesOf(command, cuts)
    .filter(([from, to]) => [...command.slice(from, to)].some((c, at) => !/\s/.test(c) && !isComment(from + at)))
    .map(([from, to]) => command.slice(from, to))
}

const ANSI_ESCAPES: Record<string, string> = { a: '\x07', b: '\b', e: '\x1b', E: '\x1b', f: '\f', n: '\n', r: '\r', t: '\t', v: '\v' }

// What a `$'…'` string stands for: its escapes read as bash reads them.
const ansiText = (text: string) =>
  text.replace(/\\(x[0-9A-Fa-f]{1,2}|u[0-9A-Fa-f]{1,4}|U[0-9A-Fa-f]{1,8}|[0-7]{1,3}|c.|.)/gs, (_, code: string) => {
    if (/^[xuU]/.test(code)) return String.fromCodePoint(parseInt(code.slice(1), 16))
    if (/^[0-7]/.test(code)) return String.fromCharCode(parseInt(code, 8) & 0xff)
    if (code[0] === 'c' && code.length === 2) return String.fromCharCode(code.charCodeAt(1) & 0x1f)
    return ANSI_ESCAPES[code] ?? code
  })

// What a region at the top stands for in a word: a quote's text, unquoted; a substitution as written.
function regionText(text: string, [from, to]: Span, kind: FrameKind) {
  const isClosed = to > from && (kind === "'" || kind === "$'" || kind === '"' ? text[to - 1] === (kind === '"' ? '"' : "'") : true)
  const inner = (opener: number) => text.slice(from + opener, isClosed ? to - 1 : to)
  if (kind === "'") return inner(1)
  if (kind === "$'") return ansiText(inner(2))
  if (kind === '"') return inner(text[from] === '$' ? 2 : 1).replace(/\\([$`"\\\n])/g, (_, c: string) => (c === '\n' ? '' : c))
  return text.slice(from, to)
}

// The words of one command, unquoted as bash would pass them: a substitution kept as written, a subshell or an
// arithmetic `((…))` left out (its commands are read on their own), a redirection one word with its operator.
function wordsOf(stage: string): string[] {
  const { regions, regionKinds, comments } = scan(stage)
  const words: string[] = []
  let word: string | undefined
  const flush = () => {
    if (word !== undefined) words.push(word)
    word = undefined
  }
  let r = 0
  for (let i = 0; i < stage.length; i++) {
    if (regions[r]?.[0] === i) {
      const kind = regionKinds[r]
      if (kind === '(' || kind === '((') flush()
      else word = (word ?? '') + regionText(stage, regions[r], kind)
      i = regions[r][1] - 1
      r++
      continue
    }
    if (comments.some(([from]) => from === i)) break
    const c = stage[i]
    if (c === '\\') {
      if (stage[i + 1] !== '\n') word = (word ?? '') + (stage[i + 1] ?? '')
      i++
    } else if (/\s/.test(c) || c === ')') flush()
    else if (c === '<' || c === '>' || (c === '&' && stage[i + 1] === '>')) {
      if (!/^\d+$/.test(word ?? '')) flush()
      let end = i
      while (end < stage.length && '<>&|'.includes(stage[end])) end++
      word = (word ?? '') + stage.slice(i, end)
      i = end - 1
    } else word = (word ?? '') + c
  }
  flush()
  return words
}

// How a wrapper takes its options: the short ones (`takes`) and long ones that take the next word, the short ones
// after which it runs no command (`stops`), the one whose value is a command line (`script`), and how many words
// come before the command it runs (`timeout 10 cmd`).
type Wrapper = { takes?: string; long?: readonly string[]; stops?: string; script?: string; positionals?: number }

const WRAPPERS: Record<string, Wrapper> = {
  sudo: {
    takes: 'ugpCDhrtUTR',
    long: ['user', 'group', 'prompt', 'close-from', 'chdir', 'host', 'role', 'type', 'other-user', 'command-timeout', 'chroot'],
    stops: 'lvVKe',
  },
  doas: { takes: 'uC' },
  env: { takes: 'uCS', long: ['unset', 'chdir', 'split-string'], script: 'S' },
  nohup: {},
  command: { stops: 'vV' },
  builtin: {},
  exec: { takes: 'a' },
  time: { takes: 'fo', long: ['format', 'output'] },
  nice: { takes: 'n', long: ['adjustment'] },
  ionice: { takes: 'cnpPu', long: ['class', 'classdata', 'pid', 'pgid', 'uid'] },
  timeout: { takes: 'sk', long: ['signal', 'kill-after'], positionals: 1 },
  xargs: {
    takes: 'aEILnPsd',
    long: ['arg-file', 'eof', 'replace', 'max-lines', 'max-args', 'max-procs', 'max-chars', 'delimiter', 'process-slot-var'],
  },
  stdbuf: { takes: 'ioe', long: ['input', 'output', 'error'] },
  setsid: {},
}

// Words bash reads before a command without changing which command runs.
const KEYWORDS = new Set(['!', '{', '}', 'if', 'then', 'elif', 'else', 'while', 'until', 'do'])
const ASSIGNMENT = /^[A-Za-z_][A-Za-z0-9_]*(\[[^\]]*\])?\+?=/
// A redirection, its operator alone or with its target; not a process substitution, `<(…)`.
const REDIRECTION = /^\d*(?:>>?|<>?|&>>?|>&|<&|>\|)(?!\()/

// `stdin`: what a here-string (`<<< 'text'`) hands the command.
type Command = { name: string; args: string[]; scripts: string[]; stdin?: string }

const basename = (word: string) => word.split('/').pop() ?? ''

// The index of the word a wrapper runs, past its options (a `--` ends them), or undefined when it runs none.
function pastOptions(words: readonly string[], at: number, wrapper: Wrapper, scripts: string[]): number | undefined {
  while (at < words.length) {
    const word = words[at]
    if (word === '--') return at + 1 + (wrapper.positionals ?? 0)
    if (!word.startsWith('-') || word === '-') break
    if (word.startsWith('--')) {
      at += !word.includes('=') && wrapper.long?.includes(word.slice(2)) ? 2 : 1
      continue
    }
    const letters = word.slice(1)
    const taking = [...letters].findIndex(letter => wrapper.takes?.includes(letter))
    const flags = taking < 0 ? letters : letters.slice(0, taking + 1)
    if ([...flags].some(letter => wrapper.stops?.includes(letter))) return undefined
    const value = taking < 0 ? undefined : taking === letters.length - 1 ? words[at + 1] : letters.slice(taking + 1)
    if (value !== undefined && letters[taking] === wrapper.script) scripts.push(value)
    at += taking >= 0 && taking === letters.length - 1 ? 2 : 1
  }
  return at + (wrapper.positionals ?? 0)
}

// What starts a stage without being its command: a function's header (`f() {`, `function f {`), a `case … in`,
// and a case arm's pattern (`a|b)`), so the commands inside are read as the commands they are.
const FUNCTION_HEADER = /^\s*(?:function\s+[^\s(){}]+(?:\s*\(\s*\))?|[A-Za-z_][\w.:-]*\s*\(\s*\))\s*/
const CASE_HEAD = /^\s*case\s+\S+\s+in\b/
const CASE_ARM = /^\s*\(?\s*[^\s()|;&]+(?:\s*\|\s*[^\s()|;&]+)*\s*\)/

// The words a command gets, past its redirections, and what a here-string hands it. A process substitution after
// `<` stays (`bash < <(curl …)` runs what it reads).
function argumentsOf(words: readonly string[]) {
  const kept: string[] = []
  let stdin: string | undefined
  for (let at = 0; at < words.length; at++) {
    const word = words[at]
    const here = /^\d*<<<(.*)$/s.exec(word)
    if (here !== null) {
      stdin = here[1] !== '' ? here[1] : words[++at]
      continue
    }
    const isAlone = REDIRECTION.test(word) && (word.replace(REDIRECTION, '') === '' || /^\d*<<-?$/.test(word))
    if (REDIRECTION.test(word)) {
      if (isAlone && !/^[<>]\(/.test(words[at + 1] ?? '')) at++
      continue
    }
    kept.push(word)
  }
  return { kept, stdin }
}

// The command a stage runs, past keywords, assignments, redirections and wrappers (`sudo -u root`, `nice -n 10`,
// `timeout 5`); undefined when it runs none (a subshell, or `command -v`).
function commandOf(stage: string): Command | undefined {
  const { kept: words, stdin } = argumentsOf(wordsOf(stage.replace(FUNCTION_HEADER, '').replace(CASE_HEAD, '').replace(CASE_ARM, '')))
  const scripts: string[] = []
  let at = 0
  while (at < words.length) {
    const word = words[at]
    if (KEYWORDS.has(word) || ASSIGNMENT.test(word)) {
      at++
      continue
    }
    const wrapper = WRAPPERS[basename(word)]
    if (wrapper === undefined) break
    const past = pastOptions(words, at + 1, wrapper, scripts)
    if (past === undefined) return undefined
    at = past
  }
  if (at >= words.length) return scripts.length > 0 ? { name: '', args: [], scripts } : undefined
  const name = basename(words[at])
  const args = words.slice(at + 1)
  return { name, args, scripts: [...scripts, ...scriptsOf(name, args)], ...(stdin !== undefined ? { stdin } : {}) }
}

const SHELLS = new Set(['sh', 'bash', 'zsh', 'dash', 'ksh', 'mksh', 'ash', 'fish'])
// Interpreters that run a script they read from stdin: with no script named, or `-`.
const INTERPRETERS = /^(?:python[0-9.]*|perl|ruby|node|php)$/
const FETCHERS = /\b(?:curl|wget)\b/
const SHELL_OPTIONS: Wrapper = { takes: 'oO' }

// Whether a command runs what reaches its stdin as a script: a shell with no `-c` and no script file (or `-s`),
// an interpreter with no `-c`, `-e` or `-m` and no script file (or `-`).
function runsStdin({ name, args }: Command) {
  if (SHELLS.has(name)) {
    if (args.some(arg => /^-[A-Za-z]*c/.test(arg))) return false
    if (args.some(arg => /^-[A-Za-z]*s/.test(arg))) return true
    const at = pastOptions(args, 0, SHELL_OPTIONS, [])
    return at === undefined || args[at] === undefined || args[at] === '-'
  }
  if (!INTERPRETERS.test(name) || args.some(arg => /^-[A-Za-z]*[cem]/.test(arg))) return false
  const script = args.find(arg => !arg.startsWith('-') || arg === '-')
  return script === undefined || script === '-'
}

// The command lines a command runs from its arguments: `bash -c '…'`, `eval …`, `su -c '…'`.
function scriptsOf(name: string, args: readonly string[]): string[] {
  if (name === 'eval') return [args.join(' ')]
  if (SHELLS.has(name) && args.some(arg => /^-[A-Za-z]*c/.test(arg))) {
    const at = pastOptions(args, 0, SHELL_OPTIONS, [])
    return at === undefined || args[at] === undefined ? [] : [args[at]]
  }
  if (name === 'su') {
    const at = args.findIndex(arg => arg === '-c' || arg === '--command')
    return at < 0 || args[at + 1] === undefined ? [] : [args[at + 1]]
  }
  return []
}

const TOP_DIRECTORIES = ['bin', 'boot', 'dev', 'etc', 'home', 'lib', 'lib64', 'opt', 'root', 'sbin', 'srv', 'usr', 'var']
// `/`, the home directory, or a top-level system directory, with or without a trailing `/` or `/*`.
const TOP = new RegExp(`^["']?(/|~|\\$HOME|\\$\\{HOME\\}|/(${TOP_DIRECTORIES.join('|')}))/?\\*?["']?$`)
// Devices a `dd` may write to without harm.
const HARMLESS_DEVICE = /^of=\/dev\/(?:null|zero|full|stdout|stderr|tty|fd\/\d+)$/

const FORK_BOMB = /:\s*\(\s*\)\s*\{[^}]*:\s*\|\s*:\s*&[^}]*\}\s*;\s*:/
// How deep the rules read into substitutions, subshells and `bash -c` strings.
const MAX_DEPTH = 8

const hasFlag = (flags: readonly string[], letter: string, long: string) =>
  flags.some(flag => flag === long || (/^-[^-]/.test(flag) && flag.includes(letter)))

// The fixed rule one command breaks, or undefined.
function ruleOf({ name, args }: Command): string | undefined {
  const flags = args.filter(word => word.startsWith('-'))
  const targets = args.filter(word => !word.startsWith('-'))
  if (name === 'rm' && (hasFlag(flags, 'r', '--recursive') || hasFlag(flags, 'R', '--recursive'))) {
    if (targets.some(target => TOP.test(target))) return 'deletes a whole top-level directory'
  }
  if (name === 'mkfs' || name.startsWith('mkfs.')) return 'formats a disk'
  if (name === 'dd' && args.some(word => word.startsWith('of=/dev/') && !HARMLESS_DEVICE.test(word))) return 'writes straight to a device'
  if (['chmod', 'chown', 'chgrp'].includes(name) && hasFlag(flags, 'R', '--recursive') && targets.includes('/')) {
    return 'changes every file on the machine'
  }
  return undefined
}

const isFetched = (text: string) => /<\(|>\(|\$\(|`/.test(text) && FETCHERS.test(text)

// A pipe that feeds a download to a shell or an interpreter, or one handed a download through `<(…)`, `$(…)` or
// backticks, as an argument or as what it reads.
function runsDownload(stages: readonly string[], commands: readonly (Command | undefined)[], scripts: readonly string[]) {
  const fetched = stages.findIndex(stage => FETCHERS.test(stage))
  const isRunner = (command: Command) => SHELLS.has(command.name) || INTERPRETERS.test(command.name)
  if (fetched >= 0 && commands.slice(fetched + 1).some(command => command && (SHELLS.has(command.name) || runsStdin(command)))) return true
  if (scripts.some(isFetched)) return true
  return commands.some(
    command =>
      command !== undefined && (isRunner(command) || ['source', '.', 'eval'].includes(command.name)) && command.args.some(isFetched),
  )
}

// What a heredoc's unquoted body runs as bash expands it: the commands in its `$(…)` and backticks.
function expansionsIn(body: string): string[] {
  const found: string[] = []
  for (let i = 0; i < body.length; i++) {
    if (body[i] === '\\') i++
    else if (body.startsWith('$((', i)) i++
    else if (body.startsWith('$(', i) || body[i] === '`') {
      const { regions, bodies } = scan(body.slice(i))
      found.push(...bodies.slice(0, 1))
      i += (regions[0]?.[1] ?? body.length - i) - 1
    }
  }
  return found
}

// What `echo` or `printf` writes, its escapes read: what a shell after it in a pipe runs.
const printedBy = ({ name, args }: Command) =>
  (name === 'echo' ? args.filter((arg, at) => !(at === 0 && /^-[neE]+$/.test(arg))) : args).join(' ').replace(/\\n/g, '\n')

export const firstOf = <T, F>(items: readonly T[], find: (item: T) => F | undefined) => {
  for (const item of items) {
    const found = find(item)
    if (found !== undefined) return found
  }
  return undefined
}

// A fixed rule a command breaks, in words, and the risk it stands for.
export type Held = { rule: string; risk: Risk }

// What makes a command catastrophic whatever the classifier scores it (a live `rm -rf ~` scored 0.78, under the
// line), or undefined: such commands are always held, also while the classifier is offline. Every command of the
// line counts: each stage of a pipe, past wrappers like `sudo` or `timeout`, and those inside substitutions,
// subshells and `bash -c` strings.
export function alwaysHeld(text: string, depth = 0): Held | undefined {
  if (depth > MAX_DEPTH) return undefined
  if (FORK_BOMB.test(text)) return { rule: 'starts a fork bomb', risk: 'destructive' }
  const { cuts, pipes, bodies, heredocs } = scan(text)
  const inner = (line: string) => alwaysHeld(line, depth + 1)
  if (cuts.length > 0) return firstOf(piecesOf(text, cuts).map(([from, to]) => text.slice(from, to)), inner)
  // The stages read with each heredoc's body blanked out, so its lines are not taken for the command's words.
  const bodyOf = (heredoc: Heredoc) => text.slice(...heredoc.body)
  const plain = heredocs.reduce((line, { body: [from], past }) => line.slice(0, from) + ' '.repeat(past - from) + line.slice(past), text)
  const spans = piecesOf(plain, pipes)
  const stages = spans.map(([from, to]) => plain.slice(from, to))
  const commands = stages.map(commandOf)
  // What reaches a stage's stdin: its own heredoc or here-string, or what `echo`, `printf` or a bare `cat` before
  // it in the pipe writes.
  const stdinOf = (at: number): string | undefined => {
    const [from, to] = spans[at]
    const own = heredocs.filter(heredoc => heredoc.at >= from && heredoc.at < to).at(-1)
    if (own !== undefined) return bodyOf(own)
    const given = commands[at]?.stdin
    if (given !== undefined) return given
    const before = at > 0 ? commands[at - 1] : undefined
    if (before?.name === 'echo' || before?.name === 'printf') return printedBy(before)
    if (before?.name === 'cat' && before.args.every(arg => arg === '-')) return stdinOf(at - 1)
    return undefined
  }
  const fed = commands.flatMap((command, at) => (command !== undefined && runsStdin(command) ? [stdinOf(at)] : [])).filter(script => script !== undefined)
  if (runsDownload(stages, commands, fed)) return { rule: 'runs a script straight from the internet', risk: 'remote_code' }
  const present = commands.filter(command => command !== undefined)
  const rule = firstOf(present, ruleOf)
  if (rule !== undefined) return { rule, risk: 'destructive' }
  const expanded = heredocs.filter(heredoc => !heredoc.isQuoted).flatMap(heredoc => expansionsIn(bodyOf(heredoc)))
  return firstOf([...fed, ...bodies, ...expanded, ...present.flatMap(command => command.scripts)], inner)
}
