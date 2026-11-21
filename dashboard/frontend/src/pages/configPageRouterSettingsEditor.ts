import * as React from 'react'
import {
  changedLeafPaths,
  deepEqual,
  getAt,
  type JsonRecord,
  type SettingPath,
} from './configPageRouterSettingsDraft'
import type { RouterStructuredSchema } from './configPageRouterStructuredSchema'

// The state every router settings field reads: the draft of the global
// override, what config.yaml saved, and the router's effective values.

export type SettingsShow = 'all' | 'set' | 'edited'

export interface RouterSettingsEditor {
  draft: JsonRecord
  saved: JsonRecord
  effective: JsonRecord | null
  readOnly: boolean
  errors: ReadonlyMap<string, string>
  query: string
  show: SettingsShow
  set: (path: SettingPath, value: unknown) => void
  unset: (path: SettingPath) => void
  restore: (path: SettingPath) => void
  /** Applies an RFC 7396 merge patch of the global block to the draft. */
  applyPatch: (patch: unknown) => void
  reportInvalid: (key: string, message: string | null) => void
}

export const RouterSettingsEditorContext = React.createContext<RouterSettingsEditor | null>(null)

export function useRouterSettingsEditor(): RouterSettingsEditor {
  const editor = React.useContext(RouterSettingsEditorContext)
  if (!editor) throw new Error('Router settings fields render inside the router settings editor.')
  return editor
}

const SCALAR_KINDS = new Set([
  'string',
  'password',
  'number',
  'boolean',
  'select',
  'string-list',
  'number-list',
])

export function isScalarSchema(schema: RouterStructuredSchema): boolean {
  return SCALAR_KINDS.has(schema.kind)
}

export function selectOptions(schema: RouterStructuredSchema): string[] {
  return (schema.options ?? []).filter((option) => option !== '')
}

/** The kind the form renders: text with contract options becomes a select. */
export function effectiveKind(schema: RouterStructuredSchema): RouterStructuredSchema['kind'] {
  if (schema.kind === 'string' && selectOptions(schema).length > 0) return 'select'
  return schema.kind
}

function searchText(schema: RouterStructuredSchema, key: string, path: SettingPath): string {
  return `${schema.label} ${key} ${path.join('.')} ${schema.description ?? ''} ${(schema.options ?? []).join(' ')}`.toLowerCase()
}

/** Whether a field or anything below it matches the search. */
export function schemaMatches(
  schema: RouterStructuredSchema,
  key: string,
  path: SettingPath,
  query: string,
): boolean {
  if (!query) return true
  if (searchText(schema, key, path).includes(query)) return true
  if (schema.kind === 'object') {
    return Object.entries(schema.fields ?? {}).some(([childKey, child]) =>
      schemaMatches(child, childKey, [...path, childKey], query),
    )
  }
  if ((schema.kind === 'object-list' || schema.kind === 'object-map') && schema.item) {
    return schemaMatches(schema.item, key, path, query)
  }
  return false
}

/** Whether a field passes the search and the All · Set · Edited filter. */
export function fieldVisible(
  editor: RouterSettingsEditor,
  schema: RouterStructuredSchema,
  key: string,
  path: SettingPath,
): boolean {
  if (!schemaMatches(schema, key, path, editor.query)) return false
  if (editor.show === 'set') return getAt(editor.draft, path) !== undefined
  if (editor.show === 'edited')
    return !deepEqual(getAt(editor.draft, path), getAt(editor.saved, path))
  return true
}

export function changedCount(editor: RouterSettingsEditor, path: SettingPath): number {
  return changedLeafPaths(getAt(editor.saved, path), getAt(editor.draft, path)).length
}

export function displayValue(value: unknown): string {
  if (value === undefined || value === null) return ''
  if (typeof value === 'boolean') return value ? 'On' : 'Off'
  if (Array.isArray(value)) return value.map(displayValue).join(', ')
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}
