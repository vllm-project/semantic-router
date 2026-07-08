import React, { useMemo } from 'react'

import { getAlgorithmFieldSchema, serializeFields } from '@/lib/dslMutations'
import type { ASTAlgoSpec, ASTModelRef, ASTPluginRef, DSLFieldObject } from '@/types/dsl'
import type { RouteAlgoInput, RouteModelInput, RoutePluginInput } from '@/lib/dslMutations'

import styles from './BuilderPage.module.css'

// ===================================================================
// Route DSL Preview generator (shared by RouteEditorForm + AddRouteForm)
// ===================================================================

function generateRouteDslPreview(
  routeName: string,
  description: string,
  priority: number,
  whenExpr: string,
  models: RouteModelInput[],
  algorithm: RouteAlgoInput | undefined,
  plugins: RoutePluginInput[],
): string {
  const descPart = description.trim() ? ` (description = "${description.trim()}")` : ''
  const lines: string[] = [`ROUTE ${routeName}${descPart} {`]
  lines.push(`  PRIORITY ${priority}`)
  if (whenExpr.trim()) {
    lines.push('')
    lines.push(`  WHEN ${whenExpr.trim().replace(/\s+/g, ' ')}`)
  }
  if (models.length > 0) {
    lines.push('')
    const modelStrs = models
      .filter((m) => m.model.trim())
      .map((m) => {
        const attrs: string[] = []
        if (m.reasoning) attrs.push(`reasoning = true`)
        if (m.effort) attrs.push(`effort = "${m.effort}"`)
        if (m.paramSize) attrs.push(`param_size = "${m.paramSize}"`)
        if (m.weight !== undefined) attrs.push(`weight = ${m.weight}`)
        const attrStr = attrs.length > 0 ? ` (${attrs.join(', ')})` : ''
        return `"${m.model}"${attrStr}`
      })
    if (modelStrs.length === 1) {
      lines.push(`  MODEL ${modelStrs[0]}`)
    } else if (modelStrs.length > 1) {
      lines.push(`  MODEL ${modelStrs.join(',\n        ')}`)
    }
  }
  if (algorithm?.algoType) {
    lines.push('')
    const aFields = filterPreviewFields(algorithm.fields)
    const algoFields = serializeFields(aFields, '    ')
    if (algoFields.trim()) {
      lines.push(`  ALGORITHM ${algorithm.algoType} {`)
      lines.push(algoFields)
      lines.push(`  }`)
    } else {
      lines.push(`  ALGORITHM ${algorithm.algoType}`)
    }
  }
  if (plugins.length > 0) {
    lines.push('')
    plugins.forEach((p) => {
      if (p.fields && Object.keys(p.fields).length > 0) {
        const pluginFields = serializeFields(filterPreviewFields(p.fields), '    ')
        lines.push(`  PLUGIN ${p.name} {`)
        if (pluginFields.trim()) lines.push(pluginFields)
        lines.push(`  }`)
      } else {
        lines.push(`  PLUGIN ${p.name}`)
      }
    })
  }
  lines.push('}')
  return lines.join('\n')
}

function filterPreviewFields(fields: DSLFieldObject): DSLFieldObject {
  return Object.fromEntries(
    Object.entries(fields).filter(([, value]) => value !== undefined && value !== null && value !== ''),
  ) as DSLFieldObject
}

interface ValidationIssue {
  level: 'error' | 'warning' | 'constraint'
  message: string
}

function validateRouteInput(
  routeName: string,
  models: RouteModelInput[],
  algorithm: RouteAlgoInput | undefined,
  _plugins: RoutePluginInput[],
): ValidationIssue[] {
  const issues: ValidationIssue[] = []

  // Route name
  if (!routeName.trim()) {
    issues.push({ level: 'error', message: 'Route name is required' })
  }

  // Models
  const validModels = models.filter((m) => m.model.trim())
  if (validModels.length === 0) {
    issues.push({
      level: 'warning',
      message: 'No model specified — route needs at least one MODEL',
    })
  }

  // Algorithm field validation
  if (algorithm?.algoType) {
    const schema = getAlgorithmFieldSchema(algorithm.algoType)

    // Check required fields
    schema
      .filter((f) => f.required)
      .forEach((f) => {
        const v = algorithm.fields[f.key]
        if (v === undefined || v === '' || v === null || (Array.isArray(v) && v.length === 0)) {
          issues.push({
            level: 'error',
            message: `Algorithm field "${f.label}" is required`,
          })
        }
      })

    // Number range checks
    const fields = algorithm.fields
    if (algorithm.algoType === 'confidence') {
      const t = fields['threshold']
      if (t !== undefined && t !== '' && typeof t === 'number' && (t < -100 || t > 0)) {
        issues.push({
          level: 'warning',
          message: `Threshold ${t} — typically negative log-prob (e.g. -1.0)`,
        })
      }
    }
    if (algorithm.algoType === 'remom' || algorithm.algoType === 'fusion') {
      const mc = fields['max_concurrent']
      if (mc !== undefined && mc !== '' && typeof mc === 'number' && mc < 0) {
        issues.push({
          level: 'error',
          message: `max_concurrent cannot be negative (got ${mc})`,
        })
      }
      const temp = fields['temperature']
      if (temp !== undefined && temp !== '' && typeof temp === 'number' && temp < 0) {
        issues.push({
          level: 'error',
          message: `temperature cannot be negative (got ${temp})`,
        })
      }
    }
    if (algorithm.algoType === 'workflows') {
      const mode = typeof fields['mode'] === 'string' ? fields['mode'] : ''
      if (mode === 'dynamic' && !extractWorkflowPlannerModel(fields)) {
        issues.push({
          level: 'error',
          message: 'workflows mode=dynamic requires planner.model',
        })
      }
      if (mode === 'dynamic' && Array.isArray(fields['roles']) && fields['roles'].length > 0) {
        issues.push({
          level: 'error',
          message: 'workflows mode=dynamic cannot include static roles',
        })
      }
      if (mode === 'dynamic' && isObjectRecord(fields['final'])) {
        issues.push({
          level: 'error',
          message: 'workflows mode=dynamic cannot include static final',
        })
      }
      if (mode !== 'dynamic') {
        issues.push(...validateWorkflowStaticRoles(fields, models))
      }
      for (const key of ['max_steps', 'max_parallel', 'max_completion_tokens']) {
        const v = fields[key]
        if (v !== undefined && v !== '' && typeof v === 'number' && v < 0) {
          issues.push({
            level: 'error',
            message: `${key} cannot be negative (got ${v})`,
          })
        }
      }
      const temp = fields['temperature']
      if (temp !== undefined && temp !== '' && typeof temp === 'number' && temp < 0) {
        issues.push({
          level: 'error',
          message: `temperature cannot be negative (got ${temp})`,
        })
      }
    }
    if (algorithm.algoType === 'fusion') {
      const maxTokens = fields['max_completion_tokens']
      if (
        maxTokens !== undefined &&
        maxTokens !== '' &&
        typeof maxTokens === 'number' &&
        maxTokens < 0
      ) {
        issues.push({
          level: 'error',
          message: `max_completion_tokens cannot be negative (got ${maxTokens})`,
        })
      }
    }
    if (algorithm.algoType === 'latency_aware') {
      for (const key of ['tpot_percentile', 'ttft_percentile']) {
        const v = fields[key]
        if (v !== undefined && v !== '' && typeof v === 'number' && (v < 1 || v > 100)) {
          issues.push({
            level: 'error',
            message: `${key} must be 1-100 (got ${v})`,
          })
        }
      }
    }
    if (algorithm.algoType === 'ratings') {
      const mc = fields['max_concurrent']
      if (mc !== undefined && mc !== '' && typeof mc === 'number' && mc < 0) {
        issues.push({
          level: 'error',
          message: `max_concurrent cannot be negative (got ${mc})`,
        })
      }
    }

    // Multi-model recommendation
    const fusionPanelModels = Array.isArray(algorithm.fields['analysis_models'])
      ? algorithm.fields['analysis_models'].filter((model) => String(model).trim()).length
      : 0
    if (
      validModels.length < 2 &&
      (algorithm.algoType !== 'fusion' || fusionPanelModels < 2) &&
      ['confidence', 'ratings', 'fusion', 'workflows', 'hybrid', 'automix'].includes(algorithm.algoType)
    ) {
      issues.push({
        level: 'constraint',
        message: `Algorithm "${algorithm.algoType}" works best with multiple models`,
      })
    }
  }

  return issues
}

function extractWorkflowPlannerModel(fields: Record<string, unknown>): string {
  const dotted = fields['planner.model']
  if (typeof dotted === 'string') return dotted.trim()
  const planner = fields['planner']
  if (planner && typeof planner === 'object' && !Array.isArray(planner)) {
    const model = (planner as Record<string, unknown>)['model']
    if (typeof model === 'string') return model.trim()
  }
  return ''
}

function validateWorkflowStaticRoles(
  fields: Record<string, unknown>,
  models: RouteModelInput[],
): ValidationIssue[] {
  const roles = fields['roles']
  if (!Array.isArray(roles) || roles.length === 0) {
    return [{ level: 'error', message: 'workflows mode=static requires roles' }]
  }

  const issues: ValidationIssue[] = []
  const allowed = new Set(models.map((m) => m.model).filter((m) => m.trim()))
  roles.forEach((role, roleIndex) => {
    if (!isObjectRecord(role)) {
      issues.push({ level: 'error', message: `workflows roles[${roleIndex}] must be an object` })
      return
    }
    const roleName = typeof role.name === 'string' ? role.name.trim() : ''
    if (!roleName) {
      issues.push({ level: 'error', message: `workflows roles[${roleIndex}].name is required` })
    }
    const roleModels = Array.isArray(role.models) ? role.models : []
    if (roleModels.length === 0) {
      issues.push({ level: 'error', message: `workflows role "${roleName || roleIndex}" needs models` })
    }
    roleModels.forEach((model, modelIndex) => {
      const modelName = typeof model === 'string' ? model.trim() : ''
      if (!modelName) {
        issues.push({ level: 'error', message: `workflows roles[${roleIndex}].models[${modelIndex}] is empty` })
      } else if (!allowed.has(modelName)) {
        issues.push({ level: 'error', message: `workflows role "${roleName || roleIndex}" uses model outside route: ${modelName}` })
      }
    })
  })

  const final = fields['final']
  if (isObjectRecord(final) && typeof final.model === 'string' && final.model.trim() && !allowed.has(final.model.trim())) {
    issues.push({ level: 'error', message: `workflows final model is outside route: ${final.model}` })
  }
  return issues
}

function isObjectRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value))
}

// ===================================================================
// Route DSL Preview with validation badges
// ===================================================================

const ISSUE_ICONS: Record<string, string> = {
  error: '✕',
  warning: '⚠',
  constraint: 'ℹ',
}

const ISSUE_COLORS: Record<string, string> = {
  error: '#ff5555',
  warning: '#f1c40f',
  constraint: '#5dade2',
}

const RouteDslPreviewPanel: React.FC<{
  dslText: string
  issues: ValidationIssue[]
  /** Diagnostics from WASM for this route (line-matched) */
  wasmDiagnostics?: { level: string; message: string }[]
}> = ({ dslText, issues, wasmDiagnostics = [] }) => {
  const allIssues = useMemo(() => {
    const merged: ValidationIssue[] = [...issues]
    wasmDiagnostics.forEach((d) => {
      merged.push({
        level: d.level as ValidationIssue['level'],
        message: d.message,
      })
    })
    return merged
  }, [issues, wasmDiagnostics])

  const errorCount = allIssues.filter((i) => i.level === 'error').length
  const warnCount = allIssues.filter((i) => i.level === 'warning').length
  const constraintCount = allIssues.filter((i) => i.level === 'constraint').length

  return (
    <div className={styles.dslPreview}>
      <div className={styles.dslPreviewHeader}>
        <span className={styles.dslPreviewTitle}>
          DSL Preview
          {allIssues.length > 0 && (
            <span
              style={{
                marginLeft: '0.5rem',
                fontSize: '0.625rem',
                fontWeight: 400,
              }}
            >
              {errorCount > 0 && (
                <span style={{ color: ISSUE_COLORS.error, marginRight: '0.5rem' }}>
                  {errorCount} error{errorCount > 1 ? 's' : ''}
                </span>
              )}
              {warnCount > 0 && (
                <span style={{ color: ISSUE_COLORS.warning, marginRight: '0.5rem' }}>
                  {warnCount} warning{warnCount > 1 ? 's' : ''}
                </span>
              )}
              {constraintCount > 0 && (
                <span style={{ color: ISSUE_COLORS.constraint }}>
                  {constraintCount} hint{constraintCount > 1 ? 's' : ''}
                </span>
              )}
            </span>
          )}
        </span>
      </div>
      <pre className={styles.dslPreviewCode}>{dslText}</pre>
      {allIssues.length > 0 && (
        <div
          style={{
            padding: '0.5rem var(--spacing-md)',
            borderTop: '1px solid var(--color-border)',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.25rem',
          }}
        >
          {allIssues.map((issue, i) => (
            <div
              key={i}
              style={{
                display: 'flex',
                alignItems: 'flex-start',
                gap: '0.5rem',
                fontSize: '0.6875rem',
                lineHeight: 1.4,
              }}
            >
              <span
                style={{
                  color: ISSUE_COLORS[issue.level],
                  fontWeight: 700,
                  flexShrink: 0,
                  width: '1rem',
                  textAlign: 'center',
                }}
              >
                {ISSUE_ICONS[issue.level]}
              </span>
              <span style={{ color: ISSUE_COLORS[issue.level] }}>{issue.message}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ===================================================================
// Route Editor Form (editable)
// ===================================================================

function astModelToInput(m: ASTModelRef): RouteModelInput {
  return {
    model: m.model,
    reasoning: m.reasoning,
    effort: m.effort,
    lora: m.lora,
    paramSize: m.paramSize,
    weight: m.weight,
    reasoningFamily: m.reasoningFamily,
  }
}

function astAlgoToInput(a?: ASTAlgoSpec): RouteAlgoInput | undefined {
  if (!a) return undefined
  return { algoType: a.algoType, fields: { ...a.fields } }
}

function astPluginRefToInput(p: ASTPluginRef): RoutePluginInput {
  return { name: p.name, fields: p.fields ? { ...p.fields } : undefined }
}

export {
  RouteDslPreviewPanel,
  astAlgoToInput,
  astModelToInput,
  astPluginRefToInput,
  generateRouteDslPreview,
  validateRouteInput,
}
