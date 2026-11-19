import * as React from 'react'

import { Badge, ProviderStack, Tooltip } from '../ui'
import { SignalChip } from './configPageRoutingScopeViews'
import { DECISION_KIND_HINT, modelDisplayName, modelLogo } from './configPageDecisionPresentation'
import { ruleConnective, ruleLeaves } from './configPageDecisionRuleSupport'
import type { DecisionConfig, NormalizedModel } from './configPageSupport'

export function DecisionConditionChips({
  decision,
  max = 2,
}: {
  decision: DecisionConfig
  max?: number
}) {
  const leaves = ruleLeaves(decision.rules)
  if (leaves.length === 0) {
    return (
      <Tooltip content={DECISION_KIND_HINT.fallback}>
        <span className="text-xs text-subtle-foreground">Always</span>
      </Tooltip>
    )
  }
  const connective = ruleConnective(decision.rules)
  return (
    <span className="flex min-w-0 items-center gap-1">
      {leaves.slice(0, max).map((leaf, index) => (
        <React.Fragment key={index}>
          {index > 0 ? (
            <span className="font-mono text-[10px] font-semibold text-chart-3">{connective}</span>
          ) : null}
          <SignalChip
            type={String(leaf.node.type ?? '')}
            name={String(leaf.node.name ?? '')}
            label={typeof leaf.node.label === 'string' ? leaf.node.label : undefined}
            negated={leaf.negated}
            className="max-w-[170px]"
          />
        </React.Fragment>
      ))}
      {leaves.length > max ? (
        <span className="text-xs text-subtle-foreground">+{leaves.length - max}</span>
      ) : null}
    </span>
  )
}

export function DecisionModelsCell({
  decision,
  models,
}: {
  decision: DecisionConfig
  models: readonly NormalizedModel[]
}) {
  const refs = decision.modelRefs ?? []
  if (refs.length === 0) {
    const fixed = (decision.plugins ?? []).some((plugin) => plugin.type === 'fast_response')
    return (
      <Badge tone={fixed ? 'rose' : 'outline'} size="sm">
        {fixed ? 'Fixed response' : 'No models'}
      </Badge>
    )
  }
  const resolved = refs.map((ref) => models.find((model) => model.name === ref.model))
  return (
    <Tooltip
      content={refs.map((ref, index) => modelDisplayName(resolved[index], ref.model)).join(', ')}
    >
      <span className="inline-flex items-center gap-2">
        <ProviderStack providers={resolved.map((model) => modelLogo(model))} size={16} />
        <span className="text-xs tabular text-muted-foreground">{refs.length}</span>
      </span>
    </Tooltip>
  )
}
