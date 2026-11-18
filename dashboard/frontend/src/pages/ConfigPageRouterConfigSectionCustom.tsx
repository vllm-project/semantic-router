import * as React from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowUpRight } from 'lucide-react'
import { Alert, Button } from '../ui'
import type { EditFormData } from '../components/EditModal'
import {
  ChoiceControl,
  FieldActions,
  FieldShell,
  SelectControl,
  TextControl,
} from './ConfigPageRouterConfigSectionControls'
import { FieldGrid, ValueObject } from './ConfigPageRouterConfigSectionFields'
import { useRouterSettingsEditor } from './configPageRouterSettingsEditor'
import { editedRouterSectionPatch, type RouterSectionCard } from './configPageRouterDefaultsSupport'
import { deepEqual, getAt } from './configPageRouterSettingsDraft'
import { settingLabel } from './configPageRouterSettingsModel'
import { getRouterStructuredFieldDefinition } from './configPageRouterStructuredFields'

// Cards whose form maps onto several global paths. Their support modules own
// the mapping; the card turns each form change into a merge patch of the draft.

const FOLLOW = '__follows_decision_model__'
const BINDING_PREFIX = 'binding:'

function useCardPatch(card: RouterSectionCard) {
  const editor = useRouterSettingsEditor()
  return (data: EditFormData) => {
    const delta = editedRouterSectionPatch(card, data)
    if (delta) editor.applyPatch(delta)
  }
}

export function DecisionModelSettings({
  card,
  disabled,
}: {
  card: RouterSectionCard
  disabled?: boolean
}) {
  const editor = useRouterSettingsEditor()
  const navigate = useNavigate()
  const patch = useCardPatch(card)
  const readOnly = editor.readOnly || disabled
  const modelField = card.editFields.find((field) => field.name === 'decision_model')
  const bindingFields = card.editFields.filter((field) => field.name.startsWith(BINDING_PREFIX))
  const modelPath = ['model_catalog', 'system', 'decision_model', 'deployment']
  const modelExplicit = getAt(editor.draft, modelPath) !== undefined
  const modelEdited = !deepEqual(getAt(editor.draft, modelPath), getAt(editor.saved, modelPath))
  const decisionModel = String(card.editData.decision_model ?? '')

  return (
    <div className="flex flex-col gap-5">
      <FieldGrid>
        <FieldShell
          id="setting-decision-model"
          label="Decision model"
          inherited={!modelExplicit}
          edited={modelEdited}
          hint="The deployment that answers built-in signals and decision questions."
          actions={
            <FieldActions
              label="Decision model"
              edited={modelEdited}
              explicit={modelExplicit}
              disabled={readOnly}
              onUndo={() => editor.restore(['model_catalog', 'system', 'decision_model'])}
              onReset={() => editor.unset(['model_catalog', 'system', 'decision_model'])}
            />
          }
        >
          <SelectControl
            id="setting-decision-model"
            label="Decision model"
            value={modelExplicit ? decisionModel : undefined}
            placeholder={decisionModel || 'primary'}
            options={modelField?.options ?? []}
            disabled={readOnly}
            onChange={(next) => patch({ ...card.editData, decision_model: next })}
          />
        </FieldShell>
        <div className="flex flex-col justify-end gap-1.5 pb-0.5">
          <p className="text-xs leading-relaxed text-muted-foreground">
            Deployments come from the model catalog. Start, scale or replace them on the Decision
            models page.
          </p>
          <Button
            variant="outline"
            size="sm"
            className="w-fit"
            onClick={() => navigate('/models/decision')}
          >
            Manage decision models
            <ArrowUpRight />
          </Button>
        </div>
      </FieldGrid>
      <div className="flex flex-col gap-3">
        <div className="flex flex-col gap-0.5">
          <span className="text-[13px] font-medium">Signal bindings</span>
          <span className="text-xs text-muted-foreground">
            Send a built-in signal to another deployment. A signal left on its default follows the
            decision model.
          </span>
        </div>
        <FieldGrid className="gap-y-4">
          {bindingFields.map((field) => {
            const signal = field.name.slice(BINDING_PREFIX.length)
            const path = ['model_catalog', 'bindings', signal]
            const explicit = getAt(editor.draft, path) !== undefined
            const edited = !deepEqual(getAt(editor.draft, path), getAt(editor.saved, path))
            const current = String(card.editData[field.name] ?? '')
            const id = `setting-binding-${signal}`
            const options = (field.options ?? []).filter((option) => option !== '')
            return (
              <FieldShell
                key={field.name}
                id={id}
                label={field.label}
                edited={edited}
                actions={
                  <FieldActions
                    label={field.label}
                    edited={edited}
                    explicit={explicit}
                    disabled={readOnly}
                    onUndo={() => editor.restore(path)}
                    onReset={() => editor.unset(path)}
                  />
                }
              >
                <SelectControl
                  id={id}
                  label={`${field.label} binding`}
                  value={current || FOLLOW}
                  options={[FOLLOW, ...options]}
                  optionLabels={{ [FOLLOW]: `Follows ${decisionModel || 'the decision model'}` }}
                  disabled={readOnly}
                  onChange={(next) =>
                    patch({ ...card.editData, [field.name]: next === FOLLOW ? '' : next })
                  }
                />
              </FieldShell>
            )
          })}
        </FieldGrid>
      </div>
    </div>
  )
}

const PROVIDER_LABELS: Record<string, string> = { local: 'Local runtime', remote: 'Remote API' }
const EMBEDDINGS_INVALID = 'embeddings'

export function EmbeddingSettings({
  card,
  disabled,
}: {
  card: RouterSectionCard
  disabled?: boolean
}) {
  const editor = useRouterSettingsEditor()
  const readOnly = editor.readOnly || disabled
  const signature = JSON.stringify(card.editData)
  const [form, setForm] = React.useState<EditFormData>(card.editData)
  const [problem, setProblem] = React.useState<string | null>(null)
  const { reportInvalid } = editor
  React.useEffect(() => {
    setForm(JSON.parse(signature) as EditFormData)
    setProblem(null)
    reportInvalid(EMBEDDINGS_INVALID, null)
  }, [signature, reportInvalid])

  const change = (name: string, value: unknown) => {
    const next = { ...form, [name]: value }
    setForm(next)
    try {
      const delta = editedRouterSectionPatch(card, next)
      setProblem(null)
      reportInvalid(EMBEDDINGS_INVALID, null)
      if (delta) editor.applyPatch(delta)
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Check the embedding settings.'
      setProblem(message)
      reportInvalid(EMBEDDINGS_INVALID, message)
    }
  }

  const remote = form.provider_type === 'remote'
  const endpointSchema = getRouterStructuredFieldDefinition('embedding_models', 'endpoint').schema
  const providerField = card.editFields.find((field) => field.name === 'provider_type')
  const defaultField = card.editFields.find((field) => field.name === 'default')
  const defaultPath = ['model_catalog', 'embeddings', 'default']
  const defaultExplicit = getAt(editor.draft, defaultPath) !== undefined
  const defaultEdited = !deepEqual(
    getAt(editor.draft, defaultPath),
    getAt(editor.saved, defaultPath),
  )

  return (
    <div className="flex flex-col gap-5">
      {problem ? (
        <Alert tone="warning" title="Finish the remote endpoint">
          {problem}
        </Alert>
      ) : null}
      <FieldGrid>
        <FieldShell
          id="setting-embedding-provider"
          label="Provider"
          hint={providerField?.description}
        >
          <ChoiceControl
            id="setting-embedding-provider"
            label="Embedding provider"
            value={String(form.provider_type ?? 'local')}
            options={['local', 'remote']}
            optionLabels={PROVIDER_LABELS}
            disabled={readOnly}
            onChange={(next) => change('provider_type', next)}
          />
        </FieldShell>
        {remote ? (
          <FieldShell
            id="setting-embedding-protocol"
            label="API protocol"
            hint="OpenAI compatible is the protocol the router speaks to remote embedding APIs."
          >
            <SelectControl
              id="setting-embedding-protocol"
              label="API protocol"
              value={String(form.remote_backend ?? 'openai_compatible')}
              options={['openai_compatible']}
              disabled={readOnly}
              onChange={(next) => change('remote_backend', next)}
            />
          </FieldShell>
        ) : (
          <FieldShell
            id="setting-embedding-default"
            label={settingLabel(defaultField?.label ?? 'Default deployment')}
            inherited={!defaultExplicit}
            edited={defaultEdited}
            hint={defaultField?.description}
            actions={
              <FieldActions
                label="Default deployment"
                edited={defaultEdited}
                explicit={defaultExplicit}
                disabled={readOnly}
                onUndo={() => editor.restore(defaultPath)}
                onReset={() => editor.unset(defaultPath)}
              />
            }
          >
            <TextControl
              id="setting-embedding-default"
              label="Default deployment"
              value={typeof form.default === 'string' && form.default ? form.default : undefined}
              placeholder={defaultField?.placeholder ?? 'vela-embedding'}
              disabled={readOnly}
              onChange={(next) => change('default', next ?? '')}
            />
          </FieldShell>
        )}
      </FieldGrid>
      {remote ? (
        <div className="flex flex-col gap-3 rounded-lg border border-border bg-surface-subtle/40 p-3 @container dark:bg-surface-subtle/30">
          <div className="flex flex-col gap-0.5">
            <span className="text-[13px] font-medium">Remote endpoint</span>
            <span className="text-xs text-muted-foreground">
              Saved as an openai_embeddings deployment that embeddings.default names. Text embedding
              only.
            </span>
          </div>
          <ValueObject
            schema={endpointSchema}
            value={form.endpoint}
            disabled={readOnly}
            idPrefix="setting-embedding-endpoint"
            onChange={(next) => change('endpoint', next)}
          />
        </div>
      ) : null}
    </div>
  )
}
