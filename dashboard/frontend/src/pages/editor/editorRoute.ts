import { formatDslName, quoteDSLString, serializeFields } from '@/lib/dslMutations'
import type { RouteAlgoInput, RouteModelInput, RoutePluginInput } from '@/lib/dslMutations'
import type { ASTAlgoSpec, ASTModelRef, ASTPluginRef, DSLFieldObject } from '@/types/dsl'
import { algorithmFieldSchema, fieldProblems } from './editorSchemas'

export interface RouteIssue {
  level: 'error' | 'warning' | 'constraint'
  message: string
}

export function astModelToInput(model: ASTModelRef): RouteModelInput {
  return {
    model: model.model,
    reasoning: model.reasoning,
    effort: model.effort,
    mode: model.mode,
    lora: model.lora,
    paramSize: model.paramSize,
    weight: model.weight,
    reasoningFamily: model.reasoningFamily,
  }
}

export function astAlgoToInput(algorithm?: ASTAlgoSpec): RouteAlgoInput | undefined {
  return algorithm ? { algoType: algorithm.algoType, fields: { ...algorithm.fields } } : undefined
}

export function astPluginRefToInput(plugin: ASTPluginRef): RoutePluginInput {
  return { name: plugin.name, fields: plugin.fields ? { ...plugin.fields } : undefined }
}

function filled(fields: DSLFieldObject): DSLFieldObject {
  return Object.fromEntries(
    Object.entries(fields).filter(
      ([, value]) => value !== undefined && value !== null && value !== '',
    ),
  ) as DSLFieldObject
}

/** The ROUTE block the form writes; settings outside the form are kept by the mutation, not shown here. */
export function routeDslPreview(
  name: string,
  description: string,
  priority: number,
  when: string,
  models: RouteModelInput[],
  algorithm: RouteAlgoInput | undefined,
  plugins: RoutePluginInput[],
): string {
  const options = description.trim() ? ` (description = ${quoteDSLString(description.trim())})` : ''
  const lines = [`ROUTE ${formatDslName(name)}${options} {`, `  PRIORITY ${priority}`]
  if (when.trim()) lines.push('', `  WHEN ${when.trim().replace(/\s+/g, ' ')}`)
  const refs = models
    .filter((model) => model.model.trim())
    .map((model) => {
      const attrs: string[] = []
      if (model.reasoning) attrs.push('reasoning = true')
      if (model.effort) attrs.push(`effort = ${quoteDSLString(model.effort)}`)
      if (model.paramSize) attrs.push(`param_size = ${quoteDSLString(model.paramSize)}`)
      if (model.weight !== undefined) attrs.push(`weight = ${model.weight}`)
      return `${quoteDSLString(model.model)}${attrs.length ? ` (${attrs.join(', ')})` : ''}`
    })
  if (refs.length) lines.push('', `  MODEL ${refs.join(',\n        ')}`)
  if (algorithm?.algoType) {
    const body = serializeFields(filled(algorithm.fields), '    ')
    lines.push('')
    if (body.trim()) lines.push(`  ALGORITHM ${algorithm.algoType} {`, body, '  }')
    else lines.push(`  ALGORITHM ${algorithm.algoType}`)
  }
  if (plugins.length) {
    lines.push('')
    for (const plugin of plugins) {
      const body = plugin.fields ? serializeFields(filled(plugin.fields), '    ') : ''
      if (body.trim()) lines.push(`  PLUGIN ${formatDslName(plugin.name)} {`, body, '  }')
      else lines.push(`  PLUGIN ${formatDslName(plugin.name)}`)
    }
  }
  lines.push('}')
  return lines.join('\n')
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value))
}

function plannerModel(fields: Record<string, unknown>): string {
  const dotted = fields['planner.model']
  if (typeof dotted === 'string') return dotted.trim()
  const planner = fields.planner
  return isRecord(planner) && typeof planner.model === 'string' ? planner.model.trim() : ''
}

function negative(fields: Record<string, unknown>, key: string): RouteIssue | null {
  const value = fields[key]
  return typeof value === 'number' && value < 0
    ? { level: 'error', message: `${key} cannot be negative (got ${value})` }
    : null
}

function workflowRoleIssues(
  fields: Record<string, unknown>,
  models: RouteModelInput[],
): RouteIssue[] {
  const roles = fields.roles
  if (!Array.isArray(roles) || roles.length === 0)
    return [{ level: 'error', message: 'workflows mode=static requires roles' }]
  const issues: RouteIssue[] = []
  const allowed = new Set(models.map((model) => model.model.trim()).filter(Boolean))
  roles.forEach((role, index) => {
    if (!isRecord(role)) {
      issues.push({ level: 'error', message: `workflows roles[${index}] must be an object` })
      return
    }
    const roleName = typeof role.name === 'string' ? role.name.trim() : ''
    if (!roleName)
      issues.push({ level: 'error', message: `workflows roles[${index}].name is required` })
    const roleModels = Array.isArray(role.models) ? role.models : []
    if (roleModels.length === 0)
      issues.push({ level: 'error', message: `workflows role "${roleName || index}" needs models` })
    roleModels.forEach((model, modelIndex) => {
      const modelName = typeof model === 'string' ? model.trim() : ''
      if (!modelName)
        issues.push({
          level: 'error',
          message: `workflows roles[${index}].models[${modelIndex}] is empty`,
        })
      else if (!allowed.has(modelName))
        issues.push({
          level: 'error',
          message: `workflows role "${roleName || index}" uses model outside route: ${modelName}`,
        })
    })
  })
  const final = fields.final
  if (isRecord(final) && typeof final.model === 'string' && final.model.trim()) {
    if (!allowed.has(final.model.trim()))
      issues.push({
        level: 'error',
        message: `workflows final model is outside route: ${final.model}`,
      })
  }
  return issues
}

/**
 * Problems the form can see before the compiler runs. Required algorithm fields
 * use the shared field validator; the rest are cross-field rules of a route.
 */
export function routeIssues(
  name: string,
  models: RouteModelInput[],
  algorithm: RouteAlgoInput | undefined,
): RouteIssue[] {
  const issues: RouteIssue[] = []
  if (!name.trim()) issues.push({ level: 'error', message: 'Name is required.' })
  const named = models.filter((model) => model.model.trim())
  if (named.length === 0)
    issues.push({ level: 'warning', message: 'No model yet: a decision needs at least one MODEL.' })
  if (!algorithm?.algoType) return issues

  const fields = algorithm.fields as Record<string, unknown>
  for (const message of fieldProblems(algorithmFieldSchema(algorithm.algoType), fields))
    issues.push({ level: 'error', message: `Algorithm: ${message}` })
  const type = algorithm.algoType
  if (type === 'remom' || type === 'fusion' || type === 'ratings') {
    for (const key of ['max_concurrent', 'temperature']) {
      const issue = negative(fields, key)
      if (issue) issues.push(issue)
    }
  }
  if (type === 'fusion') {
    const issue = negative(fields, 'max_completion_tokens')
    if (issue) issues.push(issue)
  }
  if (type === 'workflows') {
    const mode = typeof fields.mode === 'string' ? fields.mode : ''
    if (mode === 'dynamic') {
      if (!plannerModel(fields))
        issues.push({ level: 'error', message: 'workflows mode=dynamic requires planner.model' })
      if (Array.isArray(fields.roles) && fields.roles.length > 0)
        issues.push({
          level: 'error',
          message: 'workflows mode=dynamic cannot include static roles',
        })
      if (isRecord(fields.final))
        issues.push({
          level: 'error',
          message: 'workflows mode=dynamic cannot include static final',
        })
    } else {
      issues.push(...workflowRoleIssues(fields, models))
    }
    for (const key of ['max_steps', 'max_parallel', 'max_completion_tokens', 'temperature']) {
      const issue = negative(fields, key)
      if (issue) issues.push(issue)
    }
  }
  const panel = Array.isArray(fields.analysis_models)
    ? fields.analysis_models.filter((model) => String(model).trim()).length
    : 0
  if (
    named.length < 2 &&
    (type !== 'fusion' || panel < 2) &&
    ['escalate', 'ratings', 'fusion', 'workflows'].includes(type)
  ) {
    issues.push({ level: 'constraint', message: `${type} works best with two or more models.` })
  }
  return issues
}
