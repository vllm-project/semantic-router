import { getSignalFieldSchema } from '../lib/dslSchemas'
import { requiredFieldErrors } from '../lib/schemaFieldValidation'
import { cloneConfigData } from './configPageCanonicalization'
import { findSignalCatalogEntry, signalCatalogByType } from './configPageSignalCatalog'
import type { ConfigData, ConfigSignals } from './configPageSupport'
import { boundsSummary } from './ConfigPageSchemaFieldsEditorSupport'

export type SignalRecord = Record<string, unknown> & { name: string }

export interface SignalRef {
  type: string
  name: string
}

/** A signal being edited: its canonical type, its name and every other field as written. */
export interface SignalDraft {
  type: string
  name: string
  fields: Record<string, unknown>
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

export function signalRecords(
  signals: ConfigSignals | undefined,
  collection: string,
): SignalRecord[] {
  const value = (signals as Record<string, unknown> | undefined)?.[collection]
  return Array.isArray(value)
    ? value.filter(
        (entry): entry is SignalRecord => isRecord(entry) && typeof entry.name === 'string',
      )
    : []
}

export function signalDraftFromRecord(type: string, record?: Record<string, unknown>): SignalDraft {
  const fields: Record<string, unknown> = record ? JSON.parse(JSON.stringify(record)) : {}
  const name = typeof fields.name === 'string' ? fields.name : ''
  delete fields.name
  return { type, name, fields }
}

export function signalRecordForSave(draft: SignalDraft): SignalRecord {
  return { name: draft.name.trim(), ...draft.fields }
}

/** Problems that keep a draft from saving: name, required fields and duplicates. */
export function signalDraftErrors(
  config: ConfigData | null,
  draft: SignalDraft,
  original?: SignalRef,
): string[] {
  const errors: string[] = []
  const name = draft.name.trim()
  const entry = findSignalCatalogEntry(draft.type)
  if (!entry) return [`Unsupported signal type: ${draft.type}`]
  if (!name) errors.push('Name is required.')
  errors.push(...requiredFieldErrors(getSignalFieldSchema(draft.type), draft.fields))
  const sameEntry = original && original.type === draft.type && original.name === name
  if (
    name &&
    !sameEntry &&
    signalRecords(config?.signals, entry.collection).some((signal) => signal.name === name)
  ) {
    errors.push(`${entry.label} signal “${name}” already exists.`)
  }
  return errors
}

/**
 * Writes a draft into a copy of the routing profile. An edited signal keeps
 * its position; a new one, or one that changed type, is added last.
 */
export function applySignalDraft(
  config: ConfigData,
  draft: SignalDraft,
  original?: SignalRef,
): ConfigData {
  const errors = signalDraftErrors(config, draft, original)
  if (errors.length > 0) throw new Error(errors[0])
  const next = cloneConfigData(config)
  const signals = (next.signals ||= {}) as Record<string, unknown>
  const target = signalCatalogByType(draft.type).collection
  const record = signalRecordForSave(draft)
  if (original && original.type === draft.type) {
    const entries = signalRecords(next.signals, target)
    const index = entries.findIndex((signal) => signal.name === original.name)
    if (index >= 0) {
      signals[target] = entries.map((signal, position) => (position === index ? record : signal))
      return next
    }
  }
  if (original) {
    const source = signalCatalogByType(original.type).collection
    signals[source] = signalRecords(next.signals, source).filter(
      (signal) => signal.name !== original.name,
    )
  }
  signals[target] = [...signalRecords(next.signals, target), record]
  return next
}

export function removeSignalsFromConfig(
  config: ConfigData,
  targets: readonly SignalRef[],
): ConfigData {
  const next = cloneConfigData(config)
  const signals = (next.signals ||= {}) as Record<string, unknown>
  for (const target of targets) {
    const collection = signalCatalogByType(target.type).collection
    signals[collection] = signalRecords(next.signals, collection).filter(
      (signal) => signal.name !== target.name,
    )
  }
  return next
}

/** A unique name for a copy: name_copy, name_copy_2, … */
export function copyName(existing: readonly string[], name: string): string {
  const base = `${name}_copy`
  if (!existing.includes(base)) return base
  let index = 2
  while (existing.includes(`${base}_${index}`)) index += 1
  return `${base}_${index}`
}

const SUMMARY_KEYS = [
  'operator',
  'method',
  'modality',
  'role',
  'harness',
  'key',
  'stage',
  'semantics',
] as const

/** One line describing a rule: its description, else its main settings. */
export function signalSummary(record: Record<string, unknown>): string {
  if (typeof record.description === 'string' && record.description.trim())
    return record.description.trim()
  const parts: string[] = []
  for (const [key, value] of Object.entries(record)) {
    if (key === 'name' || key === 'description' || value === undefined) continue
    if (Array.isArray(value)) {
      if (value.length) parts.push(`${value.length} ${key.replace(/_/g, ' ')}`)
    } else if (
      (SUMMARY_KEYS as readonly string[]).includes(key) &&
      typeof value === 'string' &&
      value
    ) {
      parts.push(value)
    } else if (key === 'predicate' && isRecord(value)) {
      const summary = boundsSummary(value)
      if (summary) parts.push(`score ${summary}`)
    } else if (isRecord(value) && key === 'question' && typeof value.type === 'string') {
      parts.push(`${value.type} question`)
    } else if (key === 'min_tokens' || key === 'max_tokens') {
      parts.push(`${key === 'min_tokens' ? '≥' : '≤'} ${String(value)} tokens`)
    } else if (key === 'feature' && isRecord(value) && typeof value.type === 'string') {
      const source =
        isRecord(value.source) && typeof value.source.type === 'string'
          ? ` ${value.source.type}`
          : ''
      parts.push(`${value.type}${source}`)
    }
    if (parts.length >= 3) break
  }
  return parts.join(' · ') || 'Uses default behavior'
}
