import type { EngineInterface } from 'claude-code'

export async function stamp($: EngineInterface) {
  return new Date(await $.clock.now()).toISOString()
}

const trimmed = (dir: string) => dir.replace(/[/\\]+$/, '')

// The log file's path, or undefined where it cannot be placed. `~/.claude/` is the Claude config directory, so a
// session run with CLAUDE_CONFIG_DIR logs into it; any other `~/` is the home directory. A relative path is refused:
// it would land in each project's working directory, with every project's commands in it.
export async function fileOf($: EngineInterface, path: string): Promise<string | undefined> {
  if (path.startsWith('/') || /^[A-Za-z]:[/\\]/.test(path)) return path
  if (!path.startsWith('~/')) return undefined
  const config = await $.env.get('CLAUDE_CONFIG_DIR')
  if (config && path.startsWith('~/.claude/')) return `${trimmed(config)}/${path.slice('~/.claude/'.length)}`
  const home = (await $.env.get('HOME')) || (await $.env.get('USERPROFILE'))
  return home ? `${trimmed(home)}${path.slice(1)}` : undefined
}

// The labels file beside the log, where /vsr:calibrate reads the answers; none when the log file itself is named so.
export function labelsFileOf(logFile: string): string | undefined {
  const file = logFile.replace(/[^/\\]*$/, 'labels.jsonl')
  return file === logFile ? undefined : file
}
