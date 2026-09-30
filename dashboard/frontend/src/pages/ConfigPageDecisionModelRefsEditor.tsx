import decisionStyles from './ConfigPageDecisionsSection.module.css'
import ProductCheckbox from '../components/ProductCheckbox'
import type {
  DecisionFormState,
  DecisionModelRef,
  NormalizedModel,
  ReasoningFamily,
} from './configPageSupport'
import {
  modelSelectionReasoningState,
  reasoningFamilyForModel,
  reasoningFamilyIsAlwaysOn,
} from './configPageReasoningControlSupport'

type ModelRef = DecisionFormState['modelRefs'][number]
type ModelRefField =
  | 'model'
  | 'use_reasoning'
  | 'reasoning_description'
  | 'reasoning_mode'
  | 'reasoning_effort'
  | 'lora_name'
  | 'weight'
type ModelRefValue = string | boolean | number | undefined

interface EditorProps {
  value: DecisionFormState['modelRefs']
  onChange: (value: DecisionFormState['modelRefs']) => void
  models: NormalizedModel[]
  reasoningFamilies: Record<string, ReasoningFamily>
}

interface RowProps {
  value: ModelRef
  modelOptions: string[]
  family?: ReasoningFamily
  onChange: (field: ModelRefField, value: ModelRefValue) => void
  onRemove: () => void
}

const emptyModelRef = (): ModelRef => ({
  model: '',
  use_reasoning: false,
  reasoning_description: '',
  reasoning_mode: '',
  reasoning_effort: '',
  lora_name: '',
  weight: undefined,
})

const updateModelRef = (
  item: ModelRef,
  field: ModelRefField,
  value: ModelRefValue,
  models: NormalizedModel[],
  families: Record<string, ReasoningFamily>,
): ModelRef => {
  if (field === 'model') {
    const modelName = String(value || '')
    const selected = models.find((model) => model.name === modelName)
    return {
      ...item,
      model: modelName,
      ...modelSelectionReasoningState(reasoningFamilyForModel(selected, families)),
    }
  }
  if (field === 'reasoning_mode') {
    return {
      ...item,
      reasoning_mode: String(value || '') as DecisionModelRef['reasoning_mode'],
      use_reasoning: value !== 'disabled',
      ...(value === 'disabled' ? { reasoning_effort: '' } : {}),
    }
  }
  if (field === 'use_reasoning') {
    return {
      ...item,
      use_reasoning: Boolean(value),
      ...(value
        ? item.reasoning_mode === 'disabled'
          ? { reasoning_mode: '' as const }
          : {}
        : { reasoning_mode: '' as const, reasoning_effort: '' }),
    }
  }
  return { ...item, [field]: value }
}

function ReasoningSelectors({
  value,
  family,
  onChange,
}: Pick<RowProps, 'value' | 'family' | 'onChange'>) {
  const modes = family?.modes || []
  const efforts = family?.levels || []
  if (efforts.length === 0 && modes.length === 0) return <span />
  return (
    <>
      {efforts.length > 0 ? (
        <label className={decisionStyles.editorControlLabel}>
          <span className={decisionStyles.editorControlLabelText}>Reasoning effort</span>
          <select
            value={value.reasoning_effort || ''}
            onChange={(event) => onChange('reasoning_effort', event.target.value)}
            className={decisionStyles.editorSelect}
            disabled={!value.use_reasoning}
          >
            <option value="">Default effort{family?.default ? ` · ${family.default}` : ''}</option>
            {efforts.map((effort) => (
              <option key={effort} value={effort}>
                {effort}
              </option>
            ))}
          </select>
        </label>
      ) : (
        <ThinkingModeSelect value={value} family={family} onChange={onChange} />
      )}
    </>
  )
}

function ThinkingModeSelect({
  value,
  family,
  onChange,
}: Pick<RowProps, 'value' | 'family' | 'onChange'>) {
  const modes = family?.modes || []
  return (
    <label className={decisionStyles.editorControlLabel}>
      <span className={decisionStyles.editorControlLabelText}>Thinking mode</span>
      <select
        value={value.reasoning_mode || ''}
        onChange={(event) => onChange('reasoning_mode', event.target.value)}
        className={decisionStyles.editorSelect}
      >
        <option value="">
          Default mode{family?.default_mode ? ` · ${family.default_mode}` : ''}
        </option>
        {modes.map((mode) => (
          <option key={mode} value={mode}>
            {mode}
          </option>
        ))}
      </select>
    </label>
  )
}

function ModelReferenceMetadata({ value, onChange }: Pick<RowProps, 'value' | 'onChange'>) {
  return (
    <>
      <div className={decisionStyles.editorGridTwo}>
        <label className={decisionStyles.editorControlLabel}>
          <span className={decisionStyles.editorControlLabelText}>LoRA adapter</span>
          <input
            type="text"
            value={value.lora_name || ''}
            onChange={(event) => onChange('lora_name', event.target.value)}
            placeholder="Optional adapter name"
            className={decisionStyles.editorInput}
          />
        </label>
        <label className={decisionStyles.editorControlLabel}>
          <span className={decisionStyles.editorControlLabelText}>Weight</span>
          <input
            type="number"
            value={typeof value.weight === 'number' ? value.weight : ''}
            onChange={(event) =>
              onChange('weight', event.target.value === '' ? undefined : Number(event.target.value))
            }
            placeholder="Optional weight"
            step="0.1"
            min="0"
            className={decisionStyles.editorInput}
          />
        </label>
      </div>
      <label className={decisionStyles.editorControlLabel}>
        <span className={decisionStyles.editorControlLabelText}>Reasoning description</span>
        <input
          type="text"
          value={value.reasoning_description || ''}
          onChange={(event) => onChange('reasoning_description', event.target.value)}
          placeholder="Optional operator note or reasoning hint"
          className={decisionStyles.editorInput}
        />
      </label>
    </>
  )
}

function ModelReferenceRow({ value, modelOptions, family, onChange, onRemove }: RowProps) {
  const alwaysOn = reasoningFamilyIsAlwaysOn(family)
  return (
    <div className={decisionStyles.editorCard}>
      <div className={decisionStyles.editorGridTwo}>
        <label className={decisionStyles.editorControlLabel}>
          <span className={decisionStyles.editorControlLabelText}>Model</span>
          <select
            value={value.model || ''}
            onChange={(event) => onChange('model', event.target.value)}
            className={decisionStyles.editorSelect}
          >
            <option value="">Select model</option>
            {value.model && !modelOptions.includes(value.model) ? (
              <option value={value.model}>{value.model}</option>
            ) : null}
            {modelOptions.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>
        <ReasoningSelectors value={value} family={family} onChange={onChange} />
      </div>
      {(family?.levels?.length || 0) > 0 && (family?.modes?.length || 0) > 1 ? (
        <div className={decisionStyles.editorGridTwo}>
          <ThinkingModeSelect value={value} family={family} onChange={onChange} />
        </div>
      ) : null}
      <div className={decisionStyles.editorMetaRow}>
        <label className={decisionStyles.editorCheckbox}>
          <ProductCheckbox
            checked={alwaysOn || !!value.use_reasoning}
            onChange={(event) => onChange('use_reasoning', event.target.checked)}
            disabled={!family || alwaysOn}
            title={
              !family
                ? 'Assign a reasoning family to configure thinking.'
                : alwaysOn
                  ? 'This model always reasons.'
                  : undefined
            }
          />
          {alwaysOn ? 'Reasoning always on' : 'Use reasoning'}
        </label>
        <button type="button" onClick={onRemove} className={decisionStyles.editorButtonDanger}>
          Remove model reference
        </button>
      </div>
      <ModelReferenceMetadata value={value} onChange={onChange} />
    </div>
  )
}

export default function ConfigPageDecisionModelRefsEditor({
  value,
  onChange,
  models,
  reasoningFamilies,
}: EditorProps) {
  const rows = Array.isArray(value) && value.length ? value : [emptyModelRef()]
  const modelOptions = models.map((model) => model.name)
  const update = (index: number, field: ModelRefField, nextValue: ModelRefValue) =>
    onChange(
      rows.map((item, rowIndex) =>
        rowIndex === index
          ? updateModelRef(item, field, nextValue, models, reasoningFamilies)
          : item,
      ),
    )
  const remove = (index: number) => {
    const next = rows.filter((_, rowIndex) => rowIndex !== index)
    onChange(next.length ? next : [emptyModelRef()])
  }
  return (
    <div className={decisionStyles.editorList}>
      {rows.map((ref, index) => (
        <ModelReferenceRow
          key={index}
          value={ref}
          modelOptions={modelOptions}
          family={reasoningFamilyForModel(
            models.find((model) => model.name === ref.model),
            reasoningFamilies,
          )}
          onChange={(field, nextValue) => update(index, field, nextValue)}
          onRemove={() => remove(index)}
        />
      ))}
      <button
        type="button"
        onClick={() => onChange([...rows, emptyModelRef()])}
        className={decisionStyles.editorButtonSecondary}
      >
        Add Model Reference
      </button>
    </div>
  )
}
