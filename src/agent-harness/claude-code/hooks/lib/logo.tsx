import type { EngineInterface } from 'claude-code'

import type { Elements } from './draw'
import { type ArtSet, type Cell, HUES, MARK_ART } from './mark-art'
import type { LogoMode } from './settings'

export const LOGO_NAME = 'vLLM Semantic Router'

// How many rows the mark takes: 2 or 3 beside a card's lines, 4 at the top of the pane.
export type MarkRows = keyof (typeof MARK_ART)['blocks']

const WORDMARK_ROWS: MarkRows = 4

// `image`: the mark as base64 PNG; `cells`: drawn in characters (MARK_ART), smooth where the terminal draws
// the Symbols for Legacy Computing itself, else in block elements, which every terminal draws.
export type Logo = { kind: 'image'; mark: string } | { kind: 'cells'; set: ArtSet } | { kind: 'off' }

export const BLOCKS_LOGO: Logo = { kind: 'cells', set: 'blocks' }
export const SMOOTH_LOGO: Logo = { kind: 'cells', set: 'smooth' }
export const NO_LOGO: Logo = { kind: 'off' }

// A PNG's signature and the IHDR chunk's length and type, base64 (the last character also holds two bits of
// the width): the engine refuses a whole tree whose Image does not start so, which would take the card with it.
const PNG_START = /^iVBORw0KGgoAAAANSUhEU[g-v]/

// The terminals the engine draws pictures in: kitty, Ghostty, iTerm2 3.5 or later. Inside tmux only with
// `allow-passthrough`, which the environment does not tell, so tmux gets the text.
async function canDrawPictures($: EngineInterface) {
  const [tmux, program = '', version = '', term = '', kitty] = await Promise.all([
    $.env.get('TMUX'),
    $.env.get('TERM_PROGRAM'),
    $.env.get('TERM_PROGRAM_VERSION'),
    $.env.get('TERM'),
    $.env.get('KITTY_WINDOW_ID'),
  ])
  if (tmux !== undefined) return false
  if (kitty !== undefined || term === 'xterm-kitty' || program === 'ghostty' || term === 'xterm-ghostty') return true
  if (program !== 'iTerm.app') return false
  const [major = 0, minor = 0] = version.split('.').map(Number)
  return major > 3 || (major === 3 && minor >= 5)
}

const isAtLeast = (version: string, least: readonly number[]) => {
  const parts = version.split('.').map(Number)
  for (const [i, want] of least.entries()) {
    const have = parts[i] ?? 0
    if (Number.isNaN(have) || have !== want) return have > want
  }
  return true
}

// The terminals that draw the sextants and diagonal wedges themselves, whatever the font (from their sources and
// release notes): VS Code 1.108, VTE 0.60 (GNOME Terminal, Ptyxis), Konsole 22.04, foot, kitty and Ghostty.
// Windows Terminal does from 1.25 but tells no version, and inside tmux the variables are rewritten or stale.
async function drawsLegacyComputing($: EngineInterface) {
  const [tmux, program = '', version = '', term = '', kitty, vte = '', konsole = ''] = await Promise.all([
    $.env.get('TMUX'),
    $.env.get('TERM_PROGRAM'),
    $.env.get('TERM_PROGRAM_VERSION'),
    $.env.get('TERM'),
    $.env.get('KITTY_WINDOW_ID'),
    $.env.get('VTE_VERSION'),
    $.env.get('KONSOLE_VERSION'),
  ])
  if (tmux !== undefined) return false
  if (program === 'vscode') return isAtLeast(version, [1, 108])
  if (kitty !== undefined || term === 'xterm-kitty' || program === 'ghostty' || term.startsWith('foot')) return true
  return Number(vte) >= 6000 || Number(konsole) >= 220400
}

async function cellsFor($: EngineInterface) {
  return (await drawsLegacyComputing($)) ? SMOOTH_LOGO : BLOCKS_LOGO
}

// What the cards and the pane draw for `mode`: the picture where it can be drawn and read, else characters the
// terminal draws. Never throws.
export async function loadLogo($: EngineInterface, mode: LogoMode): Promise<Logo> {
  if (mode === 'off') return NO_LOGO
  if (mode === 'smooth') return SMOOTH_LOGO
  if (mode === 'blocks') return BLOCKS_LOGO
  try {
    if (mode === 'auto' && !(await canDrawPictures($))) return await cellsFor($)
    const { base64 } = await $.fs.read(`${$.plugin.root}/assets/mark.png`, { as: 'bytes' })
    return PNG_START.test(base64) ? { kind: 'image', mark: base64 } : await cellsFor($)
  } catch {
    return BLOCKS_LOGO
  }
}

// The columns the mark takes at `rows`: none with the logo off. Every set, and the picture, take the same box.
export const markColumns = (logo: Logo, rows: MarkRows) => (logo.kind === 'off' ? 0 : MARK_ART.blocks[rows][0].length)

const isSameStyle = (a: Cell, b: Cell) => a[1] === b[1] && a[2] === b[2]

// One row of blocks, runs of one style drawn together.
function artRow({ Text }: Elements, cells: readonly Cell[]) {
  const runs: Cell[] = []
  for (const cell of cells) {
    const last = runs.at(-1)
    if (last !== undefined && isSameStyle(last, cell)) runs[runs.length - 1] = [last[0] + cell[0], last[1], last[2]]
    else runs.push(cell)
  }
  return (
    <Text>
      {runs.map(([chars, fg, bg]) =>
        fg === null ? chars : <Text color={HUES[fg]} {...(bg === null ? {} : { backgroundColor: HUES[bg] })}>{chars}</Text>,
      )}
    </Text>
  )
}

// The mark, `rows` high: the picture where the surface draws one, else characters.
export function mark(elements: Elements, logo: Logo, rows: MarkRows) {
  if (logo.kind === 'off') return null
  if (logo.kind === 'image' && 'Image' in elements) {
    const { Image } = elements
    return <Image source={{ png: logo.mark }} columns={markColumns(logo, rows)} rows={rows} alt="vLLM" />
  }
  const { Box } = elements
  // A picture the surface cannot draw falls back to the characters every terminal draws.
  const set = logo.kind === 'cells' ? logo.set : 'blocks'
  return <Box flexDirection="column">{MARK_ART[set][rows].map(cells => artRow(elements, cells))}</Box>
}

// The top of the pane: the mark, and the name beside it in the terminal's own text color.
// The short name as a capsule: v white, S in the mark's yellow, R in its blue, on a dark ground every theme shows.
// BRAND is its text, for measuring; square ends, since rounded ones need a patched font.
export const BRAND = ' vSR '
const BRAND_GROUND = '#2d333f'
const BRAND_LETTERS = [
  ['v', '#ffffff'],
  ['S', HUES.Y],
  ['R', HUES.B],
] as const

export function brand({ Text }: Elements) {
  return (
    <Text backgroundColor={BRAND_GROUND} weight="bold">
      <Text backgroundColor={BRAND_GROUND}> </Text>
      {BRAND_LETTERS.map(([letter, color]) => (
        <Text backgroundColor={BRAND_GROUND} color={color}>
          {letter}
        </Text>
      ))}
      <Text backgroundColor={BRAND_GROUND}> </Text>
    </Text>
  )
}

export function wordmark(elements: Elements, logo: Logo) {
  if (logo.kind === 'off') return null
  const { Box, Text } = elements
  return (
    <Box flexDirection="row" gap={2} alignItems="center" marginBottom={1}>
      {mark(elements, logo, WORDMARK_ROWS)}
      <Text weight="bold">{LOGO_NAME}</Text>
    </Box>
  )
}
