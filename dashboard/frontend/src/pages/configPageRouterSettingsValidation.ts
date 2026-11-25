import {
  deepEqual,
  getAt,
  isRecord,
  pathKey,
  type JsonRecord,
} from './configPageRouterSettingsDraft'
import {
  curatedStructuredDefinition,
  settingLabel,
  type RouterSettingsSection,
} from './configPageRouterSettingsModel'
import {
  normalizeRouterStructuredValue,
  type RouterStructuredSchema,
} from './configPageRouterStructuredSchema'

// Checks the edited part of the draft before it is sent. The contract types
// and bounds come from the generated schema; curated structured fields also
// run the shared normalizer the legacy editor used. The router validates the
// whole document again when it saves.

function options(schema: RouterStructuredSchema): string[] {
  return (schema.options ?? []).filter((option) => option !== '')
}

function missing(value: unknown): boolean {
  return (
    value === undefined ||
    value === null ||
    value === '' ||
    (Array.isArray(value) && value.length === 0)
  )
}

function checkNode(
  schema: RouterStructuredSchema,
  value: unknown,
  path: string[],
  label: string,
  errors: Map<string, string>,
  report: string[] = path,
) {
  const at = pathKey(report)
  const fail = (message: string) => {
    if (!errors.has(at)) errors.set(at, message)
  }
  const kind = schema.kind === 'string' && options(schema).length ? 'select' : schema.kind
  switch (kind) {
    case 'number': {
      if (typeof value !== 'number' || !Number.isFinite(value)) {
        fail(
          report === path
            ? 'Enter a number, or clear the field to use the default.'
            : `${label} must be a number.`,
        )
        return
      }
      if (schema.min !== undefined && value < schema.min)
        fail(`${label} must be at least ${schema.min}.`)
      else if (schema.max !== undefined && value > schema.max)
        fail(`${label} must be at most ${schema.max}.`)
      return
    }
    case 'boolean':
      if (typeof value !== 'boolean') fail(`${label} must be on or off.`)
      return
    case 'number-list':
      if (
        !Array.isArray(value) ||
        value.some((item) => typeof item !== 'number' || !Number.isFinite(item))
      ) {
        fail(`${label} must be a list of numbers.`)
      }
      return
    case 'string-list': {
      if (!Array.isArray(value)) {
        fail(`${label} must be a list.`)
        return
      }
      const seen = new Set(value.map((item) => String(item).trim().toLowerCase()))
      if (seen.size !== value.length) fail(`${label} lists a value twice.`)
      const unknown = schema.options
        ? value.find((item) => !schema.options?.includes(String(item)))
        : undefined
      if (unknown !== undefined) fail(`${label} does not support ${String(unknown)}.`)
      return
    }
    case 'object': {
      if (!isRecord(value)) return
      for (const [key, child] of Object.entries(schema.fields ?? {})) {
        const childLabel = settingLabel(child.label || key)
        if (child.required && missing(value[key]) && report !== path) {
          fail(`${label}: ${childLabel} is required.`)
          continue
        }
        if (value[key] !== undefined) {
          checkNode(
            child,
            value[key],
            [...path, key],
            childLabel,
            errors,
            report === path ? [...path, key] : report,
          )
        }
      }
      return
    }
    case 'object-list': {
      if (!Array.isArray(value) || !schema.item) return
      value.forEach((item, index) => {
        const name =
          (isRecord(item) &&
            typeof item[schema.itemLabelKey ?? 'name'] === 'string' &&
            (item[schema.itemLabelKey ?? 'name'] as string)) ||
          `${settingLabel(schema.item?.label ?? label)} ${index + 1}`
        checkRequired(schema.item as RouterStructuredSchema, item, name, fail)
        checkNode(
          schema.item as RouterStructuredSchema,
          item,
          [...path, String(index)],
          name,
          errors,
          report,
        )
      })
      return
    }
    case 'object-map': {
      if (!isRecord(value) || !schema.item) return
      for (const [name, entry] of Object.entries(value)) {
        checkRequired(schema.item, entry, name, fail)
        checkNode(schema.item, entry, [...path, name], name, errors, report)
      }
      return
    }
    default:
      return
  }
}

function checkRequired(
  schema: RouterStructuredSchema,
  value: unknown,
  owner: string,
  fail: (message: string) => void,
) {
  if (!isRecord(value)) return
  for (const [key, child] of Object.entries(schema.fields ?? {})) {
    if (child.required && missing(value[key]))
      fail(`${owner}: ${settingLabel(child.label || key)} is required.`)
  }
}

/** Errors by dotted path for every edited field of the draft. */
export function validateRouterSettings(
  sections: readonly RouterSettingsSection[],
  draft: JsonRecord,
  saved: JsonRecord,
): Map<string, string> {
  const errors = new Map<string, string>()
  for (const section of sections) {
    for (const field of section.master ? [section.master, ...section.fields] : section.fields) {
      const value = getAt(draft, field.path)
      if (value === undefined || deepEqual(value, getAt(saved, field.path))) continue
      const label = settingLabel(field.schema.label || field.key)
      checkNode(field.schema, value, field.path, label, errors)
      const curated = curatedStructuredDefinition(field.path)
      if (curated && !errors.has(pathKey(field.path))) {
        try {
          normalizeRouterStructuredValue(curated.schema, value, curated.label)
        } catch (error) {
          errors.set(pathKey(field.path), error instanceof Error ? error.message : String(error))
        }
      }
    }
  }
  return errors
}
