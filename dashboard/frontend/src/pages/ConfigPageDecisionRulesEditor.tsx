import * as React from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { CircleSlash, Gauge, ListTree, Plus, Tag, Trash2, TriangleAlert, X } from 'lucide-react'

import { DECISION_SIGNAL_TYPES } from '../generated/routerConfigContract'
import {
  DECISION_RULE_ON_ERROR_OPTIONS,
  DECISION_RULE_ON_UNKNOWN_OPTIONS,
} from '../lib/decisionSchemas'
import {
  Badge,
  Button,
  Popover,
  PopoverContent,
  PopoverTrigger,
  Segmented,
  Tooltip,
  cn,
} from '../ui'
import { BoundsInput, OptionSelect, SuggestInput } from './ConfigPageSchemaFieldsEditorControls'
import { boundsSummary, useSchemaFormHints } from './ConfigPageSchemaFieldsEditorSupport'
import { isSingleConditionRules } from './configPageDecisionFormSupport'
import {
  childrenOf,
  isEmptyNode,
  isRecord,
  operatorOf,
  ruleExpression,
  type GroupOperator,
  type RuleNode,
} from './configPageDecisionRuleSupport'
import { findSignalCatalogEntry, signalTypeIcon, signalTypeLabel } from './configPageSignalCatalog'
import type { DecisionRuleSet } from './configPageSupport'

const GROUP_LABEL: Record<GroupOperator, string> = { AND: 'All of', OR: 'Any of', NOT: 'Not' }
const MAX_NESTING = 4
const ease = [0.16, 1, 0.3, 1] as const

function signalTypeOption(type: string) {
  const Icon = signalTypeIcon(type)
  return (
    <span className="flex items-center gap-2">
      <Icon className="size-3.5 text-chart-2" />
      <span>{signalTypeLabel(type)}</span>
      <span className="font-mono text-[11px] text-subtle-foreground in-data-[slot=select-trigger]:hidden">
        {type}
      </span>
    </span>
  )
}

function PredicateButton({
  value,
  onChange,
  readOnly,
  idPrefix,
}: {
  value: unknown
  onChange: (value: RuleNode | undefined) => void
  readOnly: boolean
  idPrefix: string
}) {
  const summary = boundsSummary(value)
  if (readOnly)
    return summary ? (
      <Badge tone="outline" className="font-mono">
        score {summary}
      </Badge>
    ) : null
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          type="button"
          variant={summary ? 'outline' : 'ghost'}
          size="xs"
          className={cn('font-mono', !summary && 'text-subtle-foreground')}
          aria-label={summary ? `Score ${summary}, edit threshold` : 'Add a score threshold'}
        >
          <Gauge />
          {summary ? `score ${summary}` : 'Threshold'}
        </Button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-80">
        <div className="flex flex-col gap-3">
          <div className="flex flex-col gap-0.5">
            <span className="text-[13px] font-medium">Score threshold</span>
            <span className="text-xs text-muted-foreground">
              Overrides the signal&apos;s own predicate for this condition only.
            </span>
          </div>
          <BoundsInput
            idPrefix={idPrefix}
            value={isRecord(value) ? value : {}}
            onChange={onChange}
          />
          {summary ? (
            <Button
              type="button"
              variant="ghost"
              size="xs"
              className="w-fit"
              onClick={() => onChange(undefined)}
            >
              <X />
              Remove threshold
            </Button>
          ) : null}
        </div>
      </PopoverContent>
    </Popover>
  )
}

function ConditionRow({
  node,
  negated,
  onChange,
  onToggleNot,
  onRemove,
  readOnly,
  signalTypes,
  idPrefix,
}: {
  node: RuleNode
  negated: boolean
  onChange: (node: RuleNode) => void
  onToggleNot?: () => void
  onRemove?: () => void
  readOnly: boolean
  signalTypes: readonly string[]
  idPrefix: string
}) {
  const hints = useSchemaFormHints()
  const [showLabel, setShowLabel] = React.useState(false)
  const type = typeof node.type === 'string' ? node.type : ''
  const name = typeof node.name === 'string' ? node.name : ''
  const declared = type ? (hints.signalNames?.(type) ?? []) : []
  const labelled = Boolean(findSignalCatalogEntry(type)?.labelled) || node.label !== undefined
  const set = (key: string, value: unknown) => {
    const next = { ...node }
    if (value === undefined || value === '') delete next[key]
    else next[key] = value
    onChange(next)
  }
  const undeclared = Boolean(type && name && hints.signalNames && !declared.includes(name))
  const Icon = signalTypeIcon(type)
  return (
    <motion.div
      layout="position"
      initial={{ opacity: 0, y: -4 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, x: 8 }}
      transition={{ duration: 0.22, ease }}
      className={cn(
        'group/row flex flex-wrap items-center gap-1.5 rounded-lg border bg-surface px-2 py-1.5 shadow-xs',
        negated ? 'border-destructive/30' : 'border-border',
      )}
    >
      {onToggleNot ? (
        <Tooltip
          content={negated ? 'Matches when this condition is false' : 'Negate this condition'}
        >
          <button
            type="button"
            disabled={readOnly}
            aria-pressed={negated}
            aria-label={negated ? 'Remove NOT' : 'Negate condition'}
            onClick={onToggleNot}
            className={cn(
              'h-7 shrink-0 rounded-md px-2 font-mono text-[11px] font-semibold transition-colors focus-visible:ring-[3px] focus-visible:ring-ring/40 focus-visible:outline-none',
              negated
                ? 'bg-destructive-soft text-destructive-foreground'
                : 'bg-secondary text-subtle-foreground hover:text-foreground',
              readOnly && 'pointer-events-none',
            )}
          >
            {negated ? 'NOT' : 'IS'}
          </button>
        </Tooltip>
      ) : null}
      {readOnly ? (
        <span className="flex h-7 items-center gap-1.5 px-1 text-[13px]">
          <Icon className="size-3.5 text-chart-2" />
          {signalTypeLabel(type)}
        </span>
      ) : (
        <OptionSelect
          ariaLabel="Signal type"
          value={type}
          options={signalTypes}
          required
          className="h-8 w-[190px] shrink-0 font-sans"
          renderOption={signalTypeOption}
          onChange={(nextType) => {
            const next: RuleNode = { ...node, type: nextType }
            if (nextType !== type) {
              delete next.name
              if (!findSignalCatalogEntry(nextType ?? '')?.labelled) delete next.label
            }
            if (!nextType) delete next.type
            onChange(next)
          }}
        />
      )}
      <SuggestInput
        ariaLabel="Signal name"
        value={name}
        mono
        placeholder={type ? `${type} name` : 'Pick a type first'}
        suggestions={declared}
        disabled={readOnly || !type}
        invalid={!name}
        onChange={(next) => set('name', next)}
        className="min-w-[150px] flex-1"
      />
      {labelled && (node.label !== undefined || showLabel) ? (
        <SuggestInput
          ariaLabel="Label"
          value={typeof node.label === 'string' ? node.label : ''}
          mono
          placeholder="label"
          suggestions={hints.signalLabels?.(type, name) ?? []}
          disabled={readOnly}
          onChange={(next) => set('label', next)}
          className="w-[132px]"
        />
      ) : labelled && !readOnly ? (
        <Tooltip content="Match one label this signal reports, such as hard or a question choice">
          <Button
            type="button"
            variant="ghost"
            size="xs"
            className="text-subtle-foreground"
            onClick={() => setShowLabel(true)}
          >
            <Tag />
            Label
          </Button>
        </Tooltip>
      ) : null}
      <PredicateButton
        value={node.predicate}
        readOnly={readOnly}
        idPrefix={`${idPrefix}-predicate`}
        onChange={(next) => set('predicate', next)}
      />
      {type === 'systemone' || node.on_error !== undefined ? (
        readOnly ? (
          node.on_error ? (
            <Badge tone="outline">on error: {String(node.on_error)}</Badge>
          ) : null
        ) : (
          <Tooltip content="What this condition does when its evaluator fails; empty uses the rule's on_unknown">
            <span>
              <OptionSelect
                ariaLabel="On error"
                value={node.on_error}
                options={['', ...DECISION_RULE_ON_ERROR_OPTIONS]}
                onChange={(next) => set('on_error', next)}
              />
            </span>
          </Tooltip>
        )
      ) : null}
      {undeclared ? (
        <Tooltip
          content={`No ${signalTypeLabel(type).toLowerCase()} rule named “${name}” is declared in this recipe`}
        >
          <span className="inline-flex h-6 items-center gap-1 rounded-md bg-warning-soft px-1.5 text-[11px] font-medium text-warning-foreground">
            <TriangleAlert className="size-3" />
            Not declared
          </span>
        </Tooltip>
      ) : null}
      {onRemove && !readOnly ? (
        <Button
          type="button"
          variant="ghost"
          size="icon-xs"
          onClick={onRemove}
          aria-label="Remove condition"
          className="ml-auto opacity-60 group-hover/row:opacity-100"
        >
          <Trash2 />
        </Button>
      ) : null}
    </motion.div>
  )
}

interface NodeEditorProps {
  node: RuleNode
  onChange: (node: RuleNode) => void
  onRemove?: () => void
  readOnly: boolean
  depth: number
  signalTypes: readonly string[]
  idPrefix: string
}

function RuleNodeEditor(props: NodeEditorProps) {
  const { node, onChange, onRemove, readOnly, signalTypes, idPrefix } = props
  const operator = operatorOf(node)
  const children = childrenOf(node)
  if (!operator) {
    return (
      <ConditionRow
        node={node}
        negated={false}
        onChange={onChange}
        onToggleNot={() => onChange({ operator: 'NOT', conditions: [node] })}
        onRemove={onRemove}
        readOnly={readOnly}
        signalTypes={signalTypes}
        idPrefix={idPrefix}
      />
    )
  }
  if (operator === 'NOT' && children.length === 1 && !operatorOf(children[0])) {
    return (
      <ConditionRow
        node={children[0]}
        negated
        onChange={(child) => onChange({ ...node, conditions: [child] })}
        onToggleNot={() => onChange(children[0])}
        onRemove={onRemove}
        readOnly={readOnly}
        signalTypes={signalTypes}
        idPrefix={idPrefix}
      />
    )
  }
  return <GroupEditor {...props} operator={operator} childNodes={children} />
}

function GroupEditor({
  node,
  onChange,
  onRemove,
  readOnly,
  depth,
  signalTypes,
  idPrefix,
  operator,
  childNodes,
}: NodeEditorProps & { operator: GroupOperator; childNodes: RuleNode[] }) {
  const setChildren = (conditions: RuleNode[]) => onChange({ ...node, operator, conditions })
  const setOperator = (next: GroupOperator) => {
    if (next === operator) return
    if (next === 'NOT' && childNodes.length !== 1) {
      onChange({ ...node, operator: 'NOT', conditions: [{ operator, conditions: childNodes }] })
      return
    }
    onChange({ ...node, operator: next, conditions: childNodes })
  }
  const connective = operator === 'OR' ? 'OR' : 'AND'
  return (
    <div
      className={cn(
        'relative flex flex-col gap-2 rounded-xl border p-3',
        depth === 0
          ? 'border-border bg-surface-subtle/60'
          : 'border-dashed border-chart-3/40 bg-chart-3/[0.04]',
      )}
    >
      <div className="flex flex-wrap items-center gap-2">
        {readOnly ? (
          <Badge tone="violet">{GROUP_LABEL[operator]}</Badge>
        ) : (
          <Segmented
            size="sm"
            aria-label="Group logic"
            value={operator}
            onValueChange={setOperator}
            options={(['AND', 'OR', 'NOT'] as const).map((value) => ({
              value,
              label: GROUP_LABEL[value],
            }))}
          />
        )}
        <span className="text-xs text-subtle-foreground">
          {operator === 'NOT'
            ? 'matches when the group below does not'
            : operator === 'OR'
              ? `at least one of ${childNodes.length} must match`
              : `${childNodes.length} ${childNodes.length === 1 ? 'condition' : 'conditions'} must all match`}
        </span>
        {onRemove && !readOnly ? (
          <Button
            type="button"
            variant="ghost"
            size="icon-xs"
            className="ml-auto"
            onClick={onRemove}
            aria-label="Remove group"
          >
            <Trash2 />
          </Button>
        ) : null}
      </div>
      <div className="relative flex flex-col gap-2 pl-5">
        <span aria-hidden className="absolute bottom-3 left-2 top-3 w-px bg-border-strong" />
        <AnimatePresence initial={false}>
          {childNodes.map((child, index) => (
            <motion.div
              key={index}
              layout="position"
              className="relative"
              transition={{ duration: 0.22, ease }}
            >
              <span aria-hidden className="absolute -left-3 top-[18px] h-px w-3 bg-border-strong" />
              {index > 0 && operator !== 'NOT' ? (
                <span className="absolute -left-[23px] -top-[13px] z-10 rounded bg-background px-1 font-mono text-[10px] font-semibold text-chart-3">
                  {connective}
                </span>
              ) : null}
              <RuleNodeEditor
                node={child}
                depth={depth + 1}
                readOnly={readOnly}
                signalTypes={signalTypes}
                idPrefix={`${idPrefix}-${index}`}
                onChange={(next) =>
                  setChildren(
                    childNodes.map((current, position) => (position === index ? next : current)),
                  )
                }
                onRemove={
                  operator === 'NOT' && childNodes.length <= 1
                    ? undefined
                    : () => setChildren(childNodes.filter((_, position) => position !== index))
                }
              />
            </motion.div>
          ))}
        </AnimatePresence>
        {childNodes.length === 0 ? (
          <p className="rounded-lg border border-dashed border-border-strong px-3 py-2 text-xs text-muted-foreground">
            Empty group. Add a condition; an empty AND group matches every request.
          </p>
        ) : null}
      </div>
      {readOnly || (operator === 'NOT' && childNodes.length >= 1) ? null : (
        <div className="flex items-center gap-1 pl-5">
          <Button
            type="button"
            variant="ghost"
            size="xs"
            onClick={() => setChildren([...childNodes, {}])}
          >
            <Plus />
            Condition
          </Button>
          {depth < MAX_NESTING ? (
            <Button
              type="button"
              variant="ghost"
              size="xs"
              onClick={() =>
                setChildren([
                  ...childNodes,
                  { operator: operator === 'AND' ? 'OR' : 'AND', conditions: [{}] },
                ])
              }
            >
              <ListTree />
              Nested group
            </Button>
          ) : null}
        </div>
      )}
    </div>
  )
}

export interface RuleTreeEditorProps {
  value: RuleNode
  onChange: (value: RuleNode) => void
  readOnly?: boolean
  /** Signal types a condition can name; decisions default to the referenceable ones. */
  signalTypes?: readonly string[]
  /** An empty tree is a valid "always" rule (decisions) rather than a condition to fill in. */
  allowAlways?: boolean
}

/** Structured All/Any/NOT editor over the canonical rule tree. Unknown keys on nodes are kept. */
export function RuleTreeEditor({
  value,
  onChange,
  readOnly = false,
  signalTypes = DECISION_SIGNAL_TYPES,
  allowAlways = true,
}: RuleTreeEditorProps) {
  const idPrefix = React.useId().replace(/:/g, '')
  const root = isRecord(value) ? value : {}
  const policy = root.on_unknown !== undefined ? { on_unknown: root.on_unknown } : {}
  if (isEmptyNode(root) && allowAlways) {
    return (
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-dashed border-border-strong px-4 py-3">
        <div className="flex items-center gap-2.5">
          <span className="grid size-7 place-items-center rounded-md bg-secondary text-muted-foreground">
            <CircleSlash className="size-4" />
          </span>
          <div className="flex flex-col">
            <span className="text-[13px] font-medium">Always matches</span>
            <span className="text-xs text-muted-foreground">
              No conditions: every request that reaches this decision matches it.
            </span>
          </div>
        </div>
        {readOnly ? null : (
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => onChange({ ...policy, operator: 'AND', conditions: [{}] })}
          >
            <Plus />
            Add condition
          </Button>
        )}
      </div>
    )
  }
  if (!operatorOf(root)) {
    const leaf = { ...root }
    delete leaf.on_unknown
    return (
      <div className="flex flex-col gap-2 rounded-xl border border-border bg-surface-subtle/60 p-3">
        <div className="flex items-center gap-2">
          <Badge tone="violet">Single condition</Badge>
          <span className="text-xs text-subtle-foreground">
            the whole rule is one signal condition
          </span>
        </div>
        <ConditionRow
          node={leaf}
          negated={false}
          readOnly={readOnly}
          signalTypes={signalTypes}
          idPrefix={idPrefix}
          onChange={(next) => onChange({ ...next, ...policy })}
          onToggleNot={() => onChange({ ...policy, operator: 'NOT', conditions: [leaf] })}
        />
        {readOnly ? null : (
          <div className="flex items-center gap-1">
            <Button
              type="button"
              variant="ghost"
              size="xs"
              onClick={() => onChange({ ...policy, operator: 'AND', conditions: [leaf, {}] })}
            >
              <Plus />
              Condition
            </Button>
            {allowAlways ? (
              <Button type="button" variant="ghost" size="xs" onClick={() => onChange({})}>
                <X />
                Clear rule
              </Button>
            ) : null}
          </div>
        )}
      </div>
    )
  }
  return (
    <div className="flex flex-col gap-2">
      <RuleNodeEditor
        node={root}
        onChange={(next) =>
          onChange(operatorOf(next) || isEmptyNode(next) ? next : { ...next, ...policy })
        }
        readOnly={readOnly}
        depth={0}
        signalTypes={signalTypes}
        idPrefix={idPrefix}
      />
      {allowAlways && !readOnly ? (
        <Button
          type="button"
          variant="ghost"
          size="xs"
          className="w-fit text-subtle-foreground"
          onClick={() => onChange({})}
        >
          <X />
          Clear all conditions
        </Button>
      ) : null}
    </div>
  )
}

interface ConfigPageDecisionRulesEditorProps {
  value: DecisionRuleSet
  onChange?: (value: DecisionRuleSet) => void
  readOnly?: boolean
}

/** A decision's rules: the condition tree and its policy for signals that cannot be evaluated. */
export default function ConfigPageDecisionRulesEditor({
  value,
  onChange,
  readOnly = false,
}: ConfigPageDecisionRulesEditorProps) {
  const rules = (value || {}) as RuleNode
  const conditional = !isEmptyNode(rules) || isSingleConditionRules(value)
  return (
    <div className="flex flex-col gap-4">
      <RuleTreeEditor
        value={rules}
        onChange={(next) => onChange?.(next as DecisionRuleSet)}
        readOnly={readOnly}
      />
      <div className="rounded-lg border border-border bg-surface-sunken/60 px-3 py-2 font-mono text-xs text-muted-foreground">
        <span className="text-subtle-foreground">DSL · </span>
        {ruleExpression(rules)}
      </div>
      {conditional ? (
        <section className="flex flex-col gap-2 rounded-lg border border-border px-3 py-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-col gap-0.5">
              <span className="text-[13px] font-medium">On unknown</span>
              <span className="font-mono text-[11px] text-subtle-foreground">rules.on_unknown</span>
            </div>
            {readOnly ? (
              <Badge tone="outline" className="font-mono">
                {value.on_unknown || 'condition policy'}
              </Badge>
            ) : (
              <OptionSelect
                ariaLabel="On unknown"
                value={value.on_unknown}
                options={['', ...DECISION_RULE_ON_UNKNOWN_OPTIONS]}
                onChange={(next) => {
                  const updated: DecisionRuleSet = { ...value }
                  if (next) updated.on_unknown = next as DecisionRuleSet['on_unknown']
                  else delete updated.on_unknown
                  onChange?.(updated)
                }}
              />
            )}
          </div>
          <p className="text-xs leading-relaxed text-muted-foreground">
            When a signal evaluator fails, no_match skips this decision, match selects it, and
            fail_request rejects the request. Leave this empty to use a classifier condition&apos;s
            on_error policy.
          </p>
        </section>
      ) : null}
    </div>
  )
}
