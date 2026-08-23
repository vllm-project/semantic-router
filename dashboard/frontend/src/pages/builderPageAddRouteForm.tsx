import { useCallback, useMemo, useState } from 'react'

import ExpressionBuilder from '@/components/ExpressionBuilder'
import ProductIcon from '@/components/ProductIcon'
import {
  ALGORITHM_DESCRIPTIONS,
  ALGORITHM_TYPES,
  type RouteAlgoInput,
  type RouteInput,
  type RoutePluginInput,
} from '@/lib/dslMutations'
import type { DSLFieldObject } from '@/types/dsl'

import styles from './BuilderPage.module.css'
import { AlgorithmSchemaEditor, PluginSchemaEditor } from './builderPageEntityForms'
import { CustomSelect, RouteIcon } from './builderPageFormPrimitives'
import { ManualPluginAdder } from './builderPageRouteSharedControls'
import type { AvailablePlugin, AvailableSignal } from './builderPageTypes'

interface Props {
  onAdd: (name: string, input: RouteInput) => void
  onCancel: () => void
  availableSignals: AvailableSignal[]
  availablePlugins: AvailablePlugin[]
}

export function AddRouteForm({ onAdd, onCancel, availableSignals, availablePlugins }: Props) {
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [priority, setPriority] = useState(100)
  const [whenExpr, setWhenExpr] = useState('')
  const [algorithm, setAlgorithm] = useState<RouteAlgoInput | undefined>()
  const [plugins, setPlugins] = useState<RoutePluginInput[]>([])
  const activePlugins = useMemo(() => new Set(plugins.map((plugin) => plugin.name)), [plugins])

  const togglePlugin = useCallback((pluginName: string) => {
    setPlugins((current) =>
      current.some((plugin) => plugin.name === pluginName)
        ? current.filter((plugin) => plugin.name !== pluginName)
        : [...current, { name: pluginName }],
    )
  }, [])

  const create = () => {
    const normalizedName = name.trim().replace(/\s+/g, '_')
    if (!normalizedName) return
    onAdd(normalizedName, {
      description: description.trim() || undefined,
      priority,
      when: whenExpr.trim() || undefined,
      models: [],
      algorithm: algorithm?.algoType ? algorithm : undefined,
      plugins,
    })
  }

  return (
    <div className={styles.editorPanel}>
      <div className={styles.editorHeader}>
        <div className={styles.editorTitle}>
          <RouteIcon className={styles.statIcon} /> New Decision
        </div>
        <div className={styles.editorActions}>
          <button type="button" className={styles.toolbarBtn} onClick={onCancel}>
            Cancel
          </button>
          <button
            type="button"
            className={styles.toolbarBtnPrimary}
            onClick={create}
            disabled={!name.trim()}
          >
            <ProductIcon name="plus" /> Create
          </button>
        </div>
      </div>

      <section className={styles.dslPreview}>
        <div className={styles.dslPreviewHeader}>
          <span className={styles.dslPreviewTitle}>Decision</span>
        </div>
        <div className={styles.builderRecipeFormGrid}>
          <label className={styles.fieldGroup}>
            Name
            <input
              className={styles.fieldInput}
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="complex"
              autoFocus
            />
          </label>
          <label className={styles.fieldGroup}>
            Priority
            <input
              className={styles.fieldInput}
              type="number"
              value={priority}
              onChange={(event) => setPriority(Number(event.target.value) || 0)}
            />
          </label>
          <label className={styles.fieldGroup}>
            Description
            <input
              className={styles.fieldInput}
              value={description}
              onChange={(event) => setDescription(event.target.value)}
              placeholder="When should this path be selected?"
            />
          </label>
        </div>
      </section>

      <section className={styles.dslPreview}>
        <div className={styles.dslPreviewHeader}>
          <span className={styles.dslPreviewTitle}>Condition</span>
        </div>
        <div className={styles.builderExpression}>
          <ExpressionBuilder
            value={whenExpr}
            onChange={setWhenExpr}
            availableSignals={availableSignals}
          />
        </div>
      </section>

      <section className={styles.dslPreview}>
        <div className={styles.dslPreviewHeader}>
          <span className={styles.dslPreviewTitle}>Algorithm</span>
          {algorithm ? (
            <button
              type="button"
              className={styles.toolbarBtnDanger}
              onClick={() => setAlgorithm(undefined)}
            >
              Remove
            </button>
          ) : (
            <button
              type="button"
              className={styles.toolbarBtn}
              onClick={() => setAlgorithm({ algoType: 'static', fields: {} })}
            >
              <ProductIcon name="plus" /> Add
            </button>
          )}
        </div>
        {algorithm ? (
          <div className={styles.builderRecipeSectionBody}>
            <CustomSelect
              value={algorithm.algoType}
              options={[...ALGORITHM_TYPES]}
              onChange={(algoType) => setAlgorithm({ algoType, fields: {} })}
            />
            <p className={styles.modalHint}>{ALGORITHM_DESCRIPTIONS[algorithm.algoType]}</p>
            <AlgorithmSchemaEditor
              modelFree
              algoType={algorithm.algoType}
              fields={algorithm.fields}
              onUpdate={(fields) => setAlgorithm({ ...algorithm, fields })}
            />
          </div>
        ) : null}
      </section>

      <section className={styles.dslPreview}>
        <div className={styles.dslPreviewHeader}>
          <span className={styles.dslPreviewTitle}>Plugins</span>
        </div>
        <div className={styles.builderRecipeSectionBody}>
          <div className={styles.pluginToggleGrid}>
            {availablePlugins.map((plugin) => (
              <button
                type="button"
                key={plugin.name}
                className={
                  activePlugins.has(plugin.name) ? styles.pluginToggleActive : styles.pluginToggle
                }
                aria-pressed={activePlugins.has(plugin.name)}
                onClick={() => togglePlugin(plugin.name)}
              >
                <span className={styles.pluginToggleCheck}>
                  {activePlugins.has(plugin.name) ? <ProductIcon name="check" /> : null}
                </span>
                <span className={styles.pluginToggleName}>{plugin.name}</span>
              </button>
            ))}
          </div>
          {plugins.map((plugin) => {
            const pluginType =
              availablePlugins.find((item) => item.name === plugin.name)?.pluginType ?? plugin.name
            return (
              <PluginSchemaEditor
                key={plugin.name}
                compact
                pluginType={pluginType}
                pluginName={plugin.name}
                fields={plugin.fields ?? {}}
                onUpdate={(fields: DSLFieldObject) =>
                  setPlugins((current) =>
                    current.map((item) => (item.name === plugin.name ? { ...item, fields } : item)),
                  )
                }
              />
            )
          })}
          <ManualPluginAdder
            existingNames={activePlugins}
            onAdd={(pluginName) => setPlugins((current) => [...current, { name: pluginName }])}
          />
        </div>
      </section>

      <p className={styles.builderAssignmentNote}>
        Assign models after this Recipe is used by a Mixture-of-Model.
      </p>
    </div>
  )
}
