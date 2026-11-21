import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { BrainCircuit, Rocket, Save } from 'lucide-react'
import {
  Alert,
  Badge,
  Button,
  Checkbox,
  Dialog,
  DialogBody,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  Field,
  IconTile,
  Input,
  RadioGroup,
  RadioGroupItem,
  Segmented,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Tooltip,
  cn,
} from '../ui'
import type { DecisionDeploymentRow } from './decisionDeploymentRows'
import {
  decisionDeploymentStem,
  formatParameters,
  type DecisionModelOption,
} from './decisionModelSupport'
import { DECISION_RUNTIME_CATALOG } from './decisionRuntimeCatalog'
import type {
  DecisionRuntimeConsumer,
  DecisionRuntimeDeploymentRequest,
} from './decisionRuntimeDeployment'
import { questionTypeLabel } from './decisionQuestionTypes'

export type DecisionBinding = 'default' | 'signals' | 'decisions' | 'none'
export type DecisionDeployTarget =
  | { kind: 'new'; model: DecisionModelOption }
  | { kind: 'bind'; row: DecisionDeploymentRow }

const DEVICES = [
  { value: 'auto', label: 'Automatic', hint: 'a GPU when the runtime finds one' },
  { value: 'cpu', label: 'CPU', hint: 'cpu' },
  { value: 'rocm:0', label: 'ROCm GPU 0', hint: 'rocm:0' },
  { value: 'rocm:1', label: 'ROCm GPU 1', hint: 'rocm:1' },
  { value: 'cuda:0', label: 'CUDA GPU 0', hint: 'cuda:0' },
  { value: 'cuda:1', label: 'CUDA GPU 1', hint: 'cuda:1' },
]

const BINDINGS: { value: DecisionBinding; title: string; description: string }[] = [
  {
    value: 'default',
    title: 'Default decision model',
    description: 'Answers every question that has no binding of its own.',
  },
  {
    value: 'signals',
    title: 'Specific signals',
    description: 'Only the System One signals you pick move here.',
  },
  {
    value: 'decisions',
    title: 'Decisions',
    description: 'Selection algorithms of the decisions you pick.',
  },
  {
    value: 'none',
    title: 'No consumer yet',
    description: 'Save the declaration only; nothing is started.',
  },
]

interface Form {
  name: string
  device: string
  replicas: string
  binding: DecisionBinding
  signals: string[]
  decisions: string[]
}

function initialForm(
  target: DecisionDeployTarget,
  consumers: DecisionRuntimeConsumer[],
  taken: Set<string>,
  binding?: DecisionBinding,
): Form {
  if (target.kind === 'bind') {
    return {
      name: target.row.name,
      device: target.row.device,
      replicas: '1',
      binding: binding ?? 'signals',
      signals: [],
      decisions: [],
    }
  }
  const stem = decisionDeploymentStem(target.model.name)
  let name = stem
  for (let suffix = 2; taken.has(name); suffix++) name = `${stem}-${suffix}`
  const signals = consumers.some(
    (consumer) =>
      consumer.kind === 'question' && target.model.questionTypes.includes(consumer.questionType),
  )
  return {
    name,
    device: 'auto',
    replicas: '1',
    binding: binding ?? (signals ? 'signals' : 'none'),
    signals: [],
    decisions: [],
  }
}

export default function DecisionDeployDialog({
  open,
  onOpenChange,
  target,
  initialBinding,
  consumers,
  takenNames,
  defaultDeployment,
  writable,
  onSubmit,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  target: DecisionDeployTarget | null
  initialBinding?: DecisionBinding
  consumers: DecisionRuntimeConsumer[]
  takenNames: Set<string>
  defaultDeployment: string
  writable: boolean
  onSubmit: (request: DecisionRuntimeDeploymentRequest, binding: DecisionBinding) => Promise<void>
}) {
  const [form, setForm] = useState<Form | null>(null)
  const [submitted, setSubmitted] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [opened, setOpened] = useState<string | null>(null)
  const key = target
    ? `${target.kind}:${target.kind === 'new' ? target.model.name : target.row.name}:${initialBinding ?? ''}`
    : null
  if (open && key !== opened && target) {
    setOpened(key)
    setForm(initialForm(target, consumers, takenNames, initialBinding))
    setSubmitted(false)
    setSaving(false)
    setError(null)
  }
  if (!open && opened !== null) setOpened(null)
  if (!target || !form) return null

  const bindMode = target.kind === 'bind'
  const row = bindMode ? target.row : undefined
  const model = bindMode ? target.row.model : target.model
  const questionTypes = bindMode ? target.row.questionTypes : target.model.questionTypes
  const set = <K extends keyof Form>(field: K, value: Form[K]) =>
    setForm((current) => (current ? { ...current, [field]: value } : current))
  const toggle = (field: 'signals' | 'decisions', id: string) =>
    setForm((current) =>
      current
        ? {
            ...current,
            [field]: current[field].includes(id)
              ? current[field].filter((value) => value !== id)
              : [...current[field], id],
          }
        : current,
    )
  const name = form.name.trim()
  const nameError = bindMode
    ? undefined
    : !name
      ? 'Name the deployment.'
      : name.startsWith('@') || /\s/.test(name)
        ? 'Use a name without spaces or the reserved @ prefix.'
        : takenNames.has(name)
          ? `${name} already exists. Pick another name or bind consumers to it.`
          : undefined
  const selected =
    form.binding === 'signals' ? form.signals : form.binding === 'decisions' ? form.decisions : []
  const consumerError =
    form.binding === 'signals' && !form.signals.length
      ? 'Pick at least one signal, or choose “No consumer yet”.'
      : form.binding === 'decisions' && !form.decisions.length
        ? 'Pick at least one decision.'
        : form.binding === 'default' && row?.isDefault
          ? `${row.name} is already the default decision model.`
          : undefined
  const invalid = Boolean(nameError || consumerError)
  const deviceWarning =
    !bindMode && model?.device === 'gpu' && form.device === 'cpu'
      ? `${model.name} is a larger decoder; on a CPU expect seconds per request.`
      : undefined
  const choices = BINDINGS.filter((option) => !bindMode || option.value !== 'none')
  const listed = consumers.filter((consumer) =>
    form.binding === 'signals' ? consumer.kind === 'question' : consumer.kind === 'selector',
  )
  const deploymentName = name || 'this deployment'
  const summary =
    form.binding === 'none'
      ? 'Saved only · nothing starts'
      : consumerError && form.binding !== 'default'
        ? 'Pick what this deployment should answer'
        : form.binding === 'default'
          ? 'Saved, then activated for every unbound question'
          : `Saved, then activated for ${selected.length} ${form.binding === 'signals' ? (selected.length === 1 ? 'signal' : 'signals') : selected.length === 1 ? 'decision' : 'decisions'}`

  const submit = async () => {
    setSubmitted(true)
    setError(null)
    if (invalid || !writable) return
    const entry =
      target.kind === 'new'
        ? DECISION_RUNTIME_CATALOG.find((value) => value.id === target.model.artifact)
        : undefined
    if (target.kind === 'new' && !entry) {
      setError('This model is no longer in the catalog. Refresh the page and try again.')
      return
    }
    setSaving(true)
    try {
      await onSubmit(
        {
          name,
          ...(entry ? { entry, device: form.device, replicas: Number(form.replicas) } : {}),
          ...(row ? { existingName: row.name, questionTypes: row.questionTypes ?? undefined } : {}),
          consumerIds: form.binding === 'signals' || form.binding === 'decisions' ? selected : [],
          makeDefault: form.binding === 'default',
        },
        form.binding,
      )
      onOpenChange(false)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The deployment could not be saved.')
    } finally {
      setSaving(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={(value) => (saving ? undefined : onOpenChange(value))}>
      <DialogContent size="lg" onOpenAutoFocus={(event) => event.preventDefault()}>
        <DialogHeader className="flex-row items-start gap-3.5 pb-4 pr-12">
          <IconTile icon={BrainCircuit} tone="violet" size="md" />
          <div className="flex min-w-0 flex-col gap-1">
            <DialogTitle>
              {row
                ? `Bind consumers to ${row.name}`
                : `Deploy ${target.kind === 'new' ? target.model.name : ''}`}
            </DialogTitle>
            <DialogDescription className="flex flex-wrap items-center gap-x-1.5 text-xs">
              <span className="font-mono text-[12px]">
                {model?.artifact ?? row?.artifact ?? row?.modelName}
              </span>
              {model ? (
                <>
                  <span className="text-subtle-foreground">·</span>
                  <span>{formatParameters(model.parameters)}</span>
                  <span className="text-subtle-foreground">·</span>
                  <span>{model.family}</span>
                </>
              ) : null}
            </DialogDescription>
          </div>
        </DialogHeader>
        <DialogBody className="flex max-h-[calc(100vh-15rem)] flex-col gap-5 pb-5">
          {bindMode ? (
            <p className="text-xs text-muted-foreground">
              Capacity stays as configured ({row?.device || 'auto'} · {row?.replicas.desired}{' '}
              {row?.replicas.desired === 1 ? 'replica' : 'replicas'}); change it in the
              deployment&apos;s details.
            </p>
          ) : (
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-[minmax(0,1fr)_190px_auto]">
              <Field
                label="Deployment"
                htmlFor="decision-deployment-name"
                error={submitted ? nameError : undefined}
              >
                <Input
                  id="decision-deployment-name"
                  value={form.name}
                  onChange={(event) => set('name', event.target.value)}
                  aria-invalid={submitted && Boolean(nameError)}
                  className="font-mono text-[12.5px]"
                  autoComplete="off"
                  spellCheck={false}
                />
              </Field>
              <Field label="Device" hint={deviceWarning}>
                <Select value={form.device} onValueChange={(value) => set('device', value)}>
                  <SelectTrigger className="w-full" aria-label="Device">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {DEVICES.map((device) => (
                      <SelectItem key={device.value} value={device.value}>
                        {device.label}
                        <span className="ml-1.5 font-mono text-[11px] text-subtle-foreground">
                          {device.value === 'auto' ? 'auto' : ''}
                        </span>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </Field>
              <Field label="Replicas">
                <Segmented
                  aria-label="Replicas"
                  value={form.replicas}
                  onValueChange={(value) => set('replicas', value)}
                  options={['1', '2', '3', '4'].map((value) => ({ value, label: value }))}
                  className="h-9"
                />
              </Field>
            </div>
          )}

          <div className="flex flex-col gap-2">
            <span className="text-[13px] font-medium">Consumers</span>
            <RadioGroup
              value={form.binding}
              onValueChange={(value) => set('binding', value as DecisionBinding)}
              className="grid grid-cols-1 gap-2 sm:grid-cols-2"
              aria-label="Consumers"
            >
              {choices.map((option) => (
                <label
                  key={option.value}
                  className={cn(
                    'flex cursor-pointer gap-3 rounded-lg border bg-card px-3 py-2.5 transition-[border-color,box-shadow]',
                    form.binding === option.value
                      ? 'border-primary shadow-[0_0_0_3px_var(--ring)]'
                      : 'border-border hover:border-border-strong',
                  )}
                >
                  <RadioGroupItem value={option.value} className="mt-0.5" />
                  <span className="flex flex-col gap-0.5">
                    <span className="text-[13px] font-medium">{option.title}</span>
                    <span className="text-xs text-muted-foreground">{option.description}</span>
                  </span>
                </label>
              ))}
            </RadioGroup>

            <AnimatePresence initial={false}>
              {form.binding === 'signals' || form.binding === 'decisions' ? (
                <motion.div
                  key={form.binding}
                  initial={{ opacity: 0, y: -4 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.2 }}
                  className="overflow-hidden rounded-lg border border-border"
                >
                  {listed.length ? (
                    listed.map((consumer) => {
                      const field = form.binding === 'signals' ? 'signals' : 'decisions'
                      const already = Boolean(row) && consumer.deployment === row?.name
                      const unsupported =
                        questionTypes !== null && !questionTypes.includes(consumer.questionType)
                      const checked = already || form[field].includes(consumer.id)
                      const line = (
                        <label
                          key={consumer.id}
                          className={cn(
                            'flex items-center gap-3 border-b border-border px-3 py-2 last:border-0',
                            unsupported || already
                              ? 'cursor-not-allowed opacity-60'
                              : 'cursor-pointer hover:bg-surface-subtle',
                          )}
                        >
                          <Checkbox
                            checked={checked}
                            disabled={unsupported || already}
                            onCheckedChange={() => toggle(field, consumer.id)}
                            aria-label={`${consumer.recipe ? `${consumer.recipe} · ` : ''}${consumer.name}`}
                          />
                          <span className="flex min-w-0 flex-1 items-center gap-1.5">
                            {consumer.recipe ? (
                              <span className="shrink-0 text-xs text-subtle-foreground">
                                {consumer.recipe} ·
                              </span>
                            ) : null}
                            <span className="truncate font-mono text-[12.5px]">
                              {consumer.name}
                            </span>
                          </span>
                          {consumer.kind === 'question' ? (
                            <Badge tone="outline" size="sm">
                              {questionTypeLabel(consumer.questionType)}
                            </Badge>
                          ) : (
                            <span className="text-xs text-subtle-foreground">
                              System One selector
                            </span>
                          )}
                          <span className="w-40 truncate text-right text-xs text-subtle-foreground">
                            {already
                              ? 'already bound here'
                              : `now on ${consumer.deployment || `${defaultDeployment} (default)`}`}
                          </span>
                        </label>
                      )
                      return unsupported ? (
                        <Tooltip
                          key={consumer.id}
                          content={`${model?.name ?? row?.name ?? 'This model'} cannot answer ${questionTypeLabel(consumer.questionType)} questions`}
                        >
                          {line}
                        </Tooltip>
                      ) : (
                        line
                      )
                    })
                  ) : (
                    <p className="px-3 py-3 text-xs text-muted-foreground">
                      {form.binding === 'signals'
                        ? 'No System One signals are configured. Add one in Signals, or save the declaration for now.'
                        : 'No decision uses a System One selector. Add one in Decisions, or save the declaration for now.'}
                    </p>
                  )}
                </motion.div>
              ) : null}
            </AnimatePresence>
            {submitted && consumerError ? (
              <p className="text-xs text-destructive-foreground">{consumerError}</p>
            ) : null}
          </div>

          {form.binding === 'none' ? (
            <Alert tone="neutral" title="Saving an unbound declaration does not start a model">
              {deploymentName} is written to the configuration and stays saved until a signal, a
              decision or the default binding uses it.
            </Alert>
          ) : (
            <Alert tone="info" title="Defaults and other bindings are preserved">
              {form.binding === 'default'
                ? `Signals and decisions that name their own deployment keep it; only questions without a binding move to ${deploymentName}.`
                : `Only the consumers you pick move to ${deploymentName}; every other signal and decision keeps its current deployment, and ${defaultDeployment} stays the default.`}
            </Alert>
          )}
          {!writable ? (
            <Alert tone="warning" title="Read-only access">
              Configuration write access is required to deploy.
            </Alert>
          ) : null}
          {error ? (
            <div role="alert">
              <Alert tone="danger" title="The deployment was not saved">
                {error}
              </Alert>
            </div>
          ) : null}
        </DialogBody>
        <DialogFooter className="justify-between">
          <span className="truncate text-xs text-muted-foreground">{summary}</span>
          <div className="flex shrink-0 items-center gap-2">
            <Button variant="ghost" onClick={() => onOpenChange(false)} disabled={saving}>
              Cancel
            </Button>
            <Button
              onClick={() => void submit()}
              loading={saving}
              disabled={!writable || (submitted && invalid)}
            >
              {saving ? null : form.binding === 'none' ? <Save /> : <Rocket />}
              {form.binding === 'none'
                ? 'Save declaration'
                : bindMode
                  ? 'Save and activate'
                  : 'Deploy and activate'}
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
