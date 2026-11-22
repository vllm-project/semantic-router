import * as React from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Check, Info, Plus, Puzzle, Trash2, X } from 'lucide-react'
import {
  Badge,
  Button,
  Input,
  Label,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Textarea,
  Tooltip,
  cn,
} from '../../ui'
import { ALGORITHM_DESCRIPTIONS, ALGORITHM_TYPES } from '@/lib/dslMutations'
import type {
  RouteAlgoInput,
  RouteInput,
  RouteModelInput,
  RoutePluginInput,
} from '@/lib/dslMutations'
import { PLUGIN_TYPES } from '@/lib/dslSchemas'
import type { ASTRouteDecl, DSLFieldObject } from '@/types/dsl'
import ConfigPageSchemaFieldsEditor from '../ConfigPageSchemaFieldsEditor'
import { RuleTreeEditor } from '../ConfigPageDecisionRulesEditor'
import {
  BooleanSwitch,
  NumberInput,
  OptionSelect,
  SuggestInput,
} from '../ConfigPageSchemaFieldsEditorControls'
import { algorithmMeta, pluginLabel } from '../configPageDecisionPresentation'
import type { RuleNode } from '../configPageDecisionRuleSupport'
import type { AvailablePlugin } from '../builderPageTypes'
import { DslPreview, FormFooter, FormSection, NameField } from './EditorForms'
import {
  astAlgoToInput,
  astModelToInput,
  astPluginRefToInput,
  routeDslPreview,
  routeIssues,
  type RouteIssue,
} from './editorRoute'
import { incompleteConditionCount, ruleTreeToWhen, whenToRuleTree } from './editorRules'
import { algorithmFieldSchema, normalizeEntityName, pluginFieldSchema } from './editorSchemas'

interface EditorRouteFormProps {
  route?: ASTRouteDecl
  availableModels: string[]
  availablePlugins: AvailablePlugin[]
  onSave?: (input: RouteInput) => void
  onCreate?: (name: string, input: RouteInput) => void
  onCancel?: () => void
}

const issueStyle: Record<RouteIssue['level'], string> = {
  error: 'text-destructive-foreground',
  warning: 'text-warning-foreground',
  constraint: 'text-info-foreground',
}

function ModelRow({
  index,
  model,
  availableModels,
  onChange,
  onRemove,
}: {
  index: number
  model: RouteModelInput
  availableModels: string[]
  onChange: (patch: Partial<RouteModelInput>) => void
  onRemove: () => void
}) {
  return (
    <motion.li
      layout="position"
      initial={{ opacity: 0, y: -4 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, x: 8 }}
      transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
      className="flex flex-col gap-2.5 rounded-lg border border-border bg-surface px-3 py-2.5 shadow-xs"
    >
      <div className="flex items-center gap-2">
        <span className="grid size-6 shrink-0 place-items-center rounded-md bg-secondary font-mono text-[11px] tabular text-muted-foreground">
          {index + 1}
        </span>
        <SuggestInput
          ariaLabel={`Model ${index + 1} name`}
          value={model.model}
          onChange={(value) => onChange({ model: value })}
          suggestions={availableModels}
          placeholder="Model name, such as qwen3-32b"
          invalid={!model.model.trim()}
          mono
          className="min-w-0 flex-1"
        />
        <Tooltip content="Remove model">
          <Button
            variant="ghost"
            size="icon-xs"
            aria-label={`Remove model ${index + 1}`}
            onClick={onRemove}
          >
            <Trash2 />
          </Button>
        </Tooltip>
      </div>
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 pl-8 text-xs text-muted-foreground">
        <span className="flex items-center gap-2">
          Reasoning
          <BooleanSwitch
            ariaLabel={`Model ${index + 1} reasoning`}
            value={model.reasoning}
            onChange={(value) => onChange({ reasoning: value || undefined })}
          />
        </span>
        <span className="flex items-center gap-2">
          Effort
          <OptionSelect
            ariaLabel={`Model ${index + 1} reasoning effort`}
            value={model.effort ?? ''}
            options={['', 'low', 'medium', 'high']}
            onChange={(value) => onChange({ effort: value || undefined })}
          />
        </span>
        <span className="flex items-center gap-2">
          Weight
          <NumberInput
            ariaLabel={`Model ${index + 1} weight`}
            value={model.weight}
            onChange={(value) => onChange({ weight: value })}
            placeholder="—"
            className="h-7 w-20"
          />
        </span>
        <span className="flex items-center gap-2">
          Param size
          <Input
            aria-label={`Model ${index + 1} param size`}
            value={model.paramSize ?? ''}
            onChange={(event) => onChange({ paramSize: event.target.value || undefined })}
            placeholder="—"
            className="h-7 w-20 font-mono text-[12px]"
          />
        </span>
      </div>
    </motion.li>
  )
}

function PluginConfig({
  plugin,
  pluginType,
  template,
  onChange,
  onRemove,
}: {
  plugin: RoutePluginInput
  pluginType: string
  template: boolean
  onChange: (fields: DSLFieldObject) => void
  onRemove: () => void
}) {
  return (
    <motion.li
      layout="position"
      initial={{ opacity: 0, y: -4 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, x: 8 }}
      transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
      className="rounded-lg border border-border bg-surface shadow-xs"
    >
      <div className="flex items-center gap-2 border-b border-border px-3 py-2">
        <Puzzle className="size-3.5 text-chart-5" />
        <span className="font-mono text-[12.5px] font-medium">{plugin.name}</span>
        <span className="text-xs text-subtle-foreground">{pluginLabel(pluginType)}</span>
        {template ? (
          <Badge tone="rose" size="sm">
            template
          </Badge>
        ) : null}
        <Tooltip content="Turn off for this decision">
          <Button
            variant="ghost"
            size="icon-xs"
            className="ml-auto"
            aria-label={`Remove plugin ${plugin.name}`}
            onClick={onRemove}
          >
            <X />
          </Button>
        </Tooltip>
      </div>
      <div className="px-3 py-3">
        <ConfigPageSchemaFieldsEditor
          schema={pluginFieldSchema(pluginType)}
          value={plugin.fields ?? {}}
          onChange={(fields) => onChange(fields as DSLFieldObject)}
          path={['route', 'plugins', plugin.name]}
        />
        {template ? (
          <p className="mt-3 text-xs text-muted-foreground">
            Fields set here override the template for this decision only.
          </p>
        ) : null}
      </div>
    </motion.li>
  )
}

/** Create or edit a ROUTE: its rule, models, algorithm and plugins. Unshown settings are kept. */
export function EditorRouteForm({
  route,
  availableModels,
  availablePlugins,
  onSave,
  onCreate,
  onCancel,
}: EditorRouteFormProps) {
  const creating = !route
  const initial = React.useMemo(
    () => ({
      description: route?.description ?? '',
      priority: route?.priority ?? 100,
      tree: whenToRuleTree(route?.when),
      models: route ? route.models.map(astModelToInput) : [{ model: '' }],
      algorithm: astAlgoToInput(route?.algorithm),
      plugins: route ? route.plugins.map(astPluginRefToInput) : [],
    }),
    [route],
  )
  const [name, setName] = React.useState('')
  const [description, setDescription] = React.useState(initial.description)
  const [priority, setPriority] = React.useState<number>(initial.priority)
  const [tree, setTree] = React.useState<RuleNode>(initial.tree)
  const [models, setModels] = React.useState<RouteModelInput[]>(initial.models)
  const [algorithm, setAlgorithm] = React.useState<RouteAlgoInput | undefined>(initial.algorithm)
  const [plugins, setPlugins] = React.useState<RoutePluginInput[]>(initial.plugins)
  const [inlinePlugin, setInlinePlugin] = React.useState('')

  const reset = () => {
    setDescription(initial.description)
    setPriority(initial.priority)
    setTree(initial.tree)
    setModels(initial.models)
    setAlgorithm(initial.algorithm)
    setPlugins(initial.plugins)
  }

  const when = ruleTreeToWhen(tree)
  const incomplete = incompleteConditionCount(tree)
  const routeName = creating ? normalizeEntityName(name) : route.name
  const issues = routeIssues(routeName, models, algorithm)
  const blocking = issues.filter((issue) => issue.level === 'error').length
  const dirty =
    JSON.stringify({ description, priority, when, models, algorithm, plugins }) !==
    JSON.stringify({
      description: initial.description,
      priority: initial.priority,
      when: ruleTreeToWhen(initial.tree),
      models: initial.models,
      algorithm: initial.algorithm,
      plugins: initial.plugins,
    })
  const activePlugins = new Set(plugins.map((plugin) => plugin.name))
  const templateType = (pluginName: string) =>
    availablePlugins.find((plugin) => plugin.name === pluginName)?.pluginType

  const input = (): RouteInput => ({
    description: description.trim() || undefined,
    priority,
    when: when || undefined,
    models: creating ? models.filter((model) => model.model.trim()) : models,
    algorithm: algorithm?.algoType ? algorithm : undefined,
    plugins,
  })

  const addInlinePlugin = () => {
    const value = normalizeEntityName(inlinePlugin)
    if (!value || activePlugins.has(value)) return
    setPlugins((current) => [...current, { name: value }])
    setInlinePlugin('')
  }

  return (
    <div className="flex flex-col gap-4">
      <FormSection
        title="Decision"
        description="Higher priority decisions are evaluated first; the first one whose rule matches wins."
      >
        <div className="grid gap-4 sm:grid-cols-[minmax(0,1fr)_9rem]">
          {creating ? (
            <div className="sm:col-span-2">
              <NameField
                id="editor-route-name"
                value={name}
                onChange={setName}
                placeholder="code_review_route"
              />
            </div>
          ) : null}
          <div className="grid gap-1.5">
            <Label htmlFor="editor-route-description">Description</Label>
            <Textarea
              id="editor-route-description"
              value={description}
              onChange={(event) => setDescription(event.target.value)}
              placeholder="What this decision is for, in one sentence."
              rows={2}
              className="min-h-[38px] resize-y"
            />
          </div>
          <div className="grid content-start gap-1.5">
            <Label htmlFor="editor-route-priority">
              Priority{' '}
              <span aria-hidden className="text-destructive-foreground">
                *
              </span>
            </Label>
            <Input
              id="editor-route-priority"
              type="number"
              inputMode="numeric"
              value={Number.isFinite(priority) ? priority : ''}
              onChange={(event) => setPriority(Number(event.target.value) || 0)}
              className="font-mono tabular"
            />
          </div>
        </div>
      </FormSection>

      <FormSection
        title="When"
        description="The signals that must match. Leave it empty for a decision that always matches."
      >
        <div className="flex flex-col gap-3">
          <RuleTreeEditor value={tree} onChange={setTree} />
          <div className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-surface-sunken/60 px-3 py-2 text-xs">
            <span className="text-subtle-foreground">WHEN</span>
            <code
              data-testid="editor-when-expression"
              className="min-w-0 break-words font-mono text-[12px] text-foreground/90"
            >
              {when || 'always'}
            </code>
          </div>
          {incomplete > 0 ? (
            <p className="flex items-center gap-1.5 text-xs text-warning-foreground">
              <Info className="size-3.5" />
              {incomplete} {incomplete === 1 ? 'condition is' : 'conditions are'} not complete and
              will be left out until they name a signal.
            </p>
          ) : null}
        </div>
      </FormSection>

      <FormSection
        title="Models"
        description="Candidates this decision may answer with, in order."
        actions={
          <Button
            variant="outline"
            size="xs"
            onClick={() => setModels((current) => [...current, { model: '' }])}
          >
            <Plus />
            Add model
          </Button>
        }
      >
        {models.length === 0 ? (
          <p className="text-[13px] text-muted-foreground">
            No models. Add at least one, or use a plugin that answers directly.
          </p>
        ) : (
          <ul className="flex flex-col gap-2">
            <AnimatePresence initial={false}>
              {models.map((model, index) => (
                <ModelRow
                  key={index}
                  index={index}
                  model={model}
                  availableModels={availableModels}
                  onChange={(patch) =>
                    setModels((current) =>
                      current.map((item, position) =>
                        position === index ? { ...item, ...patch } : item,
                      ),
                    )
                  }
                  onRemove={() =>
                    setModels((current) => current.filter((_, position) => position !== index))
                  }
                />
              ))}
            </AnimatePresence>
          </ul>
        )}
      </FormSection>

      <FormSection
        title="Algorithm"
        description={
          algorithm
            ? (ALGORITHM_DESCRIPTIONS[algorithm.algoType] ??
              'How the decision picks among its models.')
            : 'Optional. How the decision picks among several models.'
        }
        actions={
          algorithm ? (
            <Button variant="ghost" size="xs" onClick={() => setAlgorithm(undefined)}>
              <X />
              Remove
            </Button>
          ) : (
            <Button
              variant="outline"
              size="xs"
              onClick={() => setAlgorithm({ algoType: 'static', fields: {} })}
            >
              <Plus />
              Add algorithm
            </Button>
          )
        }
      >
        {algorithm ? (
          <div className="flex flex-col gap-4">
            <div className="grid max-w-sm gap-1.5">
              <Label htmlFor="editor-route-algorithm">Algorithm type</Label>
              <Select
                value={algorithm.algoType}
                onValueChange={(value) => setAlgorithm({ algoType: value, fields: {} })}
              >
                <SelectTrigger id="editor-route-algorithm">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {ALGORITHM_TYPES.map((type) => (
                    <SelectItem key={type} value={type}>
                      <span className="flex items-center gap-2">
                        {algorithmMeta(type)?.label ?? type}
                        <span className="font-mono text-[11px] text-subtle-foreground">{type}</span>
                      </span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <ConfigPageSchemaFieldsEditor
              schema={algorithmFieldSchema(algorithm.algoType)}
              value={algorithm.fields}
              onChange={(fields) =>
                setAlgorithm((current) =>
                  current ? { ...current, fields: fields as DSLFieldObject } : current,
                )
              }
              path={['route', 'algorithm', algorithm.algoType]}
            />
          </div>
        ) : (
          <p className="text-[13px] text-muted-foreground">
            {models.filter((model) => model.model.trim()).length >= 2
              ? 'Recommended: with several models, an algorithm decides which one answers.'
              : 'The first model answers.'}
          </p>
        )}
      </FormSection>

      <FormSection
        title="Plugins"
        description="Plugins run for requests this decision matches: caching, prompts, guards and more."
      >
        <div className="flex flex-col gap-3">
          {availablePlugins.length ? (
            <div className="flex flex-col gap-1.5">
              <span className="text-xs font-medium text-muted-foreground">Templates</span>
              <div className="flex flex-wrap gap-1.5">
                {availablePlugins.map((plugin) => {
                  const active = activePlugins.has(plugin.name)
                  return (
                    <button
                      key={plugin.name}
                      type="button"
                      aria-pressed={active}
                      onClick={() =>
                        setPlugins((current) =>
                          active
                            ? current.filter((item) => item.name !== plugin.name)
                            : [...current, { name: plugin.name }],
                        )
                      }
                      className={cn(
                        'inline-flex h-7 items-center gap-1.5 rounded-md border px-2 text-xs outline-none transition-colors focus-visible:ring-[3px] focus-visible:ring-ring/40',
                        active
                          ? 'border-chart-5/40 bg-chart-5/10 text-foreground'
                          : 'border-border bg-surface text-muted-foreground hover:border-border-strong hover:text-foreground',
                      )}
                    >
                      {active ? (
                        <Check className="size-3.5 text-chart-5" />
                      ) : (
                        <Plus className="size-3.5" />
                      )}
                      <span className="font-mono text-[12px]">{plugin.name}</span>
                      <span className="text-subtle-foreground">{plugin.pluginType}</span>
                    </button>
                  )
                })}
              </div>
            </div>
          ) : null}
          {plugins.length ? (
            <ul className="flex flex-col gap-2">
              <AnimatePresence initial={false}>
                {plugins.map((plugin) => {
                  const template = templateType(plugin.name)
                  return (
                    <PluginConfig
                      key={plugin.name}
                      plugin={plugin}
                      pluginType={template ?? plugin.name}
                      template={Boolean(template)}
                      onChange={(fields) =>
                        setPlugins((current) =>
                          current.map((item) =>
                            item.name === plugin.name ? { ...item, fields } : item,
                          ),
                        )
                      }
                      onRemove={() =>
                        setPlugins((current) => current.filter((item) => item.name !== plugin.name))
                      }
                    />
                  )
                })}
              </AnimatePresence>
            </ul>
          ) : (
            <p className="text-[13px] text-muted-foreground">No plugins on this decision.</p>
          )}
          <div className="flex items-center gap-2">
            <SuggestInput
              ariaLabel="Inline plugin type"
              value={inlinePlugin}
              onChange={setInlinePlugin}
              suggestions={PLUGIN_TYPES.filter((type) => !activePlugins.has(type))}
              placeholder="Add a plugin by type, such as system_prompt"
              mono
              className="min-w-0 flex-1"
            />
            <Button
              variant="outline"
              size="sm"
              onClick={addInlinePlugin}
              disabled={
                !normalizeEntityName(inlinePlugin) ||
                activePlugins.has(normalizeEntityName(inlinePlugin))
              }
            >
              <Plus />
              Add
            </Button>
          </div>
        </div>
      </FormSection>

      {issues.length ? (
        <ul className="flex flex-col gap-1 rounded-lg border border-border bg-surface-subtle/60 px-3 py-2.5 text-xs">
          {issues.map((issue) => (
            <li
              key={issue.message}
              className={cn('flex items-start gap-1.5', issueStyle[issue.level])}
            >
              <Info className="mt-px size-3.5 shrink-0" />
              {issue.message}
            </li>
          ))}
        </ul>
      ) : null}

      <DslPreview
        code={routeDslPreview(
          routeName || 'name_here',
          description,
          priority,
          when,
          models,
          algorithm,
          plugins,
        )}
      />
      <FormFooter
        mode={creating ? 'create' : 'edit'}
        dirty={dirty}
        problems={blocking}
        onReset={reset}
        onCancel={onCancel}
        onSubmit={() => (creating ? onCreate?.(routeName, input()) : onSave?.(input()))}
      />
    </div>
  )
}
