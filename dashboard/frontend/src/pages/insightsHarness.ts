import ampLogo from '@lobehub/icons-static-svg/icons/amp-color.svg'
import claudeCodeLogo from '@lobehub/icons-static-svg/icons/claudecode-color.svg'
import clineLogo from '@lobehub/icons-static-svg/icons/cline.svg'
import codexLogo from '@lobehub/icons-static-svg/icons/codex-color.svg'
import copilotLogo from '@lobehub/icons-static-svg/icons/copilot-color.svg'
import cursorLogo from '@lobehub/icons-static-svg/icons/cursor.svg'
import deepseekLogo from '@lobehub/icons-static-svg/icons/deepseek-color.svg'
import geminiCliLogo from '@lobehub/icons-static-svg/icons/geminicli-color.svg'
import githubCopilotLogo from '@lobehub/icons-static-svg/icons/githubcopilot.svg'
import gooseLogo from '@lobehub/icons-static-svg/icons/goose.svg'
import kimiLogo from '@lobehub/icons-static-svg/icons/kimi-color.svg'
import openClawLogo from '@lobehub/icons-static-svg/icons/openclaw-color.svg'
import openCodeLogo from '@lobehub/icons-static-svg/icons/opencode.svg'
import rooCodeLogo from '@lobehub/icons-static-svg/icons/roocode.svg'

/**
 * The agent harnesses the Router names in `agent.harness` of a replay record
 * and `harness` of a run (routerreplay/store/agent.go), and the ones it may
 * name later. A record without a harness came from a plain API client.
 */
export interface HarnessInfo {
  /** The canonical id, as the Router writes it. */
  id: string
  label: string
  /** The logo asset; none for a harness this registry does not know. */
  logo?: string
  /** A logo drawn in currentColor, which an <img> renders black: invert it in dark mode. */
  monochrome: boolean
  known: boolean
}

interface HarnessEntry {
  label: string
  logo: string
  monochrome?: boolean
  aliases?: string[]
}

const HARNESSES: Record<string, HarnessEntry> = {
  claude_code: { label: 'Claude Code', logo: claudeCodeLogo, aliases: ['claudecode'] },
  codex: { label: 'Codex', logo: codexLogo, aliases: ['codex_cli', 'openai_codex'] },
  opencode: { label: 'OpenCode', logo: openCodeLogo, monochrome: true, aliases: ['open_code'] },
  cursor: {
    label: 'Cursor',
    logo: cursorLogo,
    monochrome: true,
    aliases: ['cursor_agent', 'cursor_app'],
  },
  openclaw: {
    label: 'OpenClaw',
    logo: openClawLogo,
    aliases: ['open_claw', 'clawdbot', 'moltbot'],
  },
  deepseek_harness: {
    label: 'DeepSeek Harness',
    logo: deepseekLogo,
    aliases: ['dsh', 'deepseek', 'deepseekharness'],
  },
  gemini_cli: { label: 'Gemini CLI', logo: geminiCliLogo, aliases: ['geminicli', 'gemini'] },
  cline: { label: 'Cline', logo: clineLogo, monochrome: true },
  roo_code: { label: 'Roo Code', logo: rooCodeLogo, monochrome: true, aliases: ['roocode', 'roo'] },
  kimi_cli: { label: 'Kimi CLI', logo: kimiLogo, aliases: ['kimi', 'kimi_code'] },
  copilot: { label: 'Copilot', logo: copilotLogo },
  github_copilot: {
    label: 'GitHub Copilot',
    logo: githubCopilotLogo,
    monochrome: true,
    aliases: ['githubcopilot', 'copilot_cli'],
  },
  amp: { label: 'Amp', logo: ampLogo },
  goose: { label: 'Goose', logo: gooseLogo, monochrome: true },
}

/** The harnesses the Router identifies today, first in every list and filter. */
export const PRIMARY_HARNESSES = [
  'claude_code',
  'codex',
  'opencode',
  'cursor',
  'openclaw',
  'deepseek_harness',
] as const

/** The agent signals a harness can give the Router, as the Requests page's signals table names them. */
export const HARNESS_SIGNALS = [
  'session',
  'subagents',
  'subagent_type',
  'request_class',
  'compaction',
  'housekeeping',
  'images',
] as const

export type HarnessSignal = (typeof HARNESS_SIGNALS)[number]

/**
 * The signals a harness's requests cannot carry, as captured from it:
 * Cursor and DeepSeek Harness name no agent or request class, and an
 * OpenClaw sub-agent's requests name no parent. A signal not listed may
 * simply not be seen yet.
 */
export const HARNESS_NEVER_SENDS: Record<string, readonly HarnessSignal[]> = {
  cursor: ['subagents', 'subagent_type', 'request_class'],
  deepseek_harness: ['subagents', 'subagent_type', 'request_class'],
  openclaw: ['subagents', 'subagent_type'],
}

const ALIASES = new Map<string, string>(
  Object.entries(HARNESSES).flatMap(([id, entry]) => [
    [id, id] as const,
    ...(entry.aliases ?? []).map((alias) => [alias, id] as const),
  ]),
)

/** A harness value as an id: lower case, with spaces and dashes as underscores. */
export function normalizeHarnessId(raw?: string | null): string {
  const value = (raw ?? '')
    .trim()
    .toLowerCase()
    .replace(/[\s-]+/g, '_')
  return ALIASES.get(value) ?? value
}

function titleCase(id: string): string {
  return id
    .split('_')
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}

/** The harness a value names, or undefined for none: a plain API client. */
export function harnessInfo(raw?: string | null): HarnessInfo | undefined {
  const id = normalizeHarnessId(raw)
  if (!id) return undefined
  const entry = HARNESSES[id]
  if (!entry) return { id, label: titleCase(id), monochrome: false, known: false }
  return {
    id,
    label: entry.label,
    logo: entry.logo,
    monochrome: Boolean(entry.monochrome),
    known: true,
  }
}

/** What a client column reads for a record or run with no harness. */
export const API_CLIENT_LABEL = 'API client'

/** The canonical harness of a replay record: its agent identity's harness. */
export function recordHarnessId(record: { agent?: { harness?: string } | null }): string {
  return normalizeHarnessId(record.agent?.harness)
}

/** Harness ids in display order: the identified harnesses first, then by label. */
export function sortHarnessIds(ids: Iterable<string>): string[] {
  const rank = (id: string) => {
    const index = (PRIMARY_HARNESSES as readonly string[]).indexOf(id)
    return index < 0 ? PRIMARY_HARNESSES.length : index
  }
  return [...new Set(ids)].sort(
    (left, right) =>
      rank(left) - rank(right) ||
      (harnessInfo(left)?.label ?? left).localeCompare(harnessInfo(right)?.label ?? right),
  )
}
