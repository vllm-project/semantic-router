import type { LucideIcon } from 'lucide-react'
import { Blend, Boxes, Puzzle, Radar, Split } from 'lucide-react'
import type { Tone } from '../../ui'
import type { ASTProgram, DiagLevel, Diagnostic } from '@/types/dsl'
import type { EntityKind, Selection } from '../builderPageTypes'

/** Outline groups; the three projection kinds share one group. */
export type EditorGroup = 'model' | 'signal' | 'projection' | 'route' | 'plugin'

export const GROUP_ORDER: EditorGroup[] = ['model', 'signal', 'projection', 'route', 'plugin']

export const groupMeta: Record<
  EditorGroup,
  { label: string; one: string; icon: LucideIcon; tone: Tone; dsl: string }
> = {
  model: { label: 'Models', one: 'Model', icon: Boxes, tone: 'green', dsl: 'MODEL' },
  signal: { label: 'Signals', one: 'Signal', icon: Radar, tone: 'gold', dsl: 'SIGNAL' },
  projection: {
    label: 'Projections',
    one: 'Projection',
    icon: Blend,
    tone: 'teal',
    dsl: 'PROJECTION',
  },
  route: { label: 'Decisions', one: 'Decision', icon: Split, tone: 'violet', dsl: 'ROUTE' },
  plugin: { label: 'Plugins', one: 'Plugin', icon: Puzzle, tone: 'rose', dsl: 'PLUGIN' },
}

export function groupOf(kind: EntityKind): EditorGroup {
  return kind.startsWith('projection-') ? 'projection' : (kind as EditorGroup)
}

export const kindLabel: Record<EntityKind, string> = {
  model: 'Model',
  signal: 'Signal',
  'projection-partition': 'Projection partition',
  'projection-score': 'Projection score',
  'projection-mapping': 'Projection mapping',
  route: 'Decision',
  plugin: 'Plugin',
}

export interface EditorEntity {
  key: string
  kind: EntityKind
  name: string
  /** Signal type or plugin type. */
  subType?: string
  /** Short right-aligned fact in the outline, such as a priority or a projection kind. */
  meta?: string
  line: number
  endLine: number
}

export function selectionKey(selection: Pick<Selection, 'kind' | 'name' | 'signalType'>): string {
  return `${selection.kind}:${selection.signalType ?? ''}:${selection.name}`
}

/** The last line of the block that opens on `line`, found by matching braces outside strings. */
export function blockEndLine(lines: readonly string[], line: number): number {
  let depth = 0
  let opened = false
  for (let index = Math.max(0, line - 1); index < lines.length; index += 1) {
    let inString = false
    const text = lines[index]
    for (let column = 0; column < text.length; column += 1) {
      const char = text[column]
      if (inString) {
        if (char === '\\') column += 1
        else if (char === '"') inString = false
        continue
      }
      if (char === '#') break
      if (char === '"') inString = true
      else if (char === '{') {
        depth += 1
        opened = true
      } else if (char === '}') depth -= 1
    }
    if (opened && depth <= 0) return index + 1
    // A one-line declaration without a body ends where it starts.
    if (!opened && index > line - 1 && /^\s*(SIGNAL|ROUTE|MODEL|PLUGIN|PROJECTION)\b/.test(text))
      return index
  }
  return opened ? lines.length : line
}

/** Every entity of a program in outline order, with the source lines it spans. */
export function listEditorEntities(ast: ASTProgram | null, source: string): EditorEntity[] {
  if (!ast) return []
  const lines = source.split('\n')
  const at = (line: number | undefined) => {
    const start = Math.max(1, line ?? 1)
    return { line: start, endLine: blockEndLine(lines, start) }
  }
  const entity = (
    kind: EntityKind,
    name: string,
    line: number | undefined,
    extra: Partial<EditorEntity> = {},
  ): EditorEntity => ({
    key: selectionKey({ kind, name, signalType: kind === 'signal' ? extra.subType : undefined }),
    kind,
    name,
    ...at(line),
    ...extra,
  })
  return [
    ...(ast.models ?? []).map((model) => entity('model', model.name, model.pos?.Line)),
    ...(ast.signals ?? []).map((signal) =>
      entity('signal', signal.name, signal.pos?.Line, {
        subType: signal.signalType,
        meta: signal.signalType,
      }),
    ),
    ...(ast.projectionPartitions ?? []).map((partition) =>
      entity('projection-partition', partition.name, partition.pos?.Line, { meta: 'partition' }),
    ),
    ...(ast.projectionScores ?? []).map((score) =>
      entity('projection-score', score.name, score.pos?.Line, { meta: 'score' }),
    ),
    ...(ast.projectionMappings ?? []).map((mapping) =>
      entity('projection-mapping', mapping.name, mapping.pos?.Line, { meta: 'mapping' }),
    ),
    ...[...(ast.routes ?? [])]
      .sort((left, right) => right.priority - left.priority)
      .map((route) => entity('route', route.name, route.pos?.Line, { meta: `P${route.priority}` })),
    ...(ast.plugins ?? []).map((plugin) =>
      entity('plugin', plugin.name, plugin.pos?.Line, {
        subType: plugin.pluginType,
        meta: plugin.pluginType,
      }),
    ),
  ]
}

export function entityAtLine(entities: readonly EditorEntity[], line: number): EditorEntity | null {
  let best: EditorEntity | null = null
  for (const item of entities) {
    if (line < item.line || line > item.endLine) continue
    if (!best || item.endLine - item.line < best.endLine - best.line) best = item
  }
  return best
}

const SEVERITY_RANK: Record<DiagLevel, number> = { error: 3, warning: 2, constraint: 1 }

/** The most severe diagnostic inside each entity's lines. */
export function worstDiagnosticByEntity(
  entities: readonly EditorEntity[],
  diagnostics: readonly Diagnostic[],
): Map<string, DiagLevel> {
  const worst = new Map<string, DiagLevel>()
  for (const diagnostic of diagnostics) {
    const owner = entityAtLine(entities, diagnostic.line)
    if (!owner) continue
    const current = worst.get(owner.key)
    if (!current || SEVERITY_RANK[diagnostic.level] > SEVERITY_RANK[current])
      worst.set(owner.key, diagnostic.level)
  }
  return worst
}

export function selectionFromEntity(entity: EditorEntity): Selection {
  return {
    kind: entity.kind,
    name: entity.name,
    ...(entity.kind === 'signal' ? { signalType: entity.subType } : {}),
  }
}

export function countDiagnostics(diagnostics: readonly Diagnostic[], level: DiagLevel): number {
  return diagnostics.filter((diagnostic) => diagnostic.level === level).length
}

export function countLabel(count: number, one: string, many = `${one}s`): string {
  return `${count} ${count === 1 ? one : many}`
}
