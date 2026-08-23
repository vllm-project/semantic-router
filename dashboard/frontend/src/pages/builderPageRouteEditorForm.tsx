import { useEffect, useMemo, useState } from 'react'

import ExpressionBuilder from '@/components/ExpressionBuilder'
import ProductIcon from '@/components/ProductIcon'
import {
  ALGORITHM_DESCRIPTIONS,
  ALGORITHM_TYPES,
  serializeBoolExpr,
  type RouteAlgoInput,
  type RouteInput,
  type RoutePluginInput,
} from '@/lib/dslMutations'
import type { ASTRouteDecl, DSLFieldObject } from '@/types/dsl'

import styles from './BuilderPage.module.css'
import { AlgorithmSchemaEditor, PluginSchemaEditor } from './builderPageEntityForms'
import { CustomSelect } from './builderPageFormPrimitives'
import { astAlgoToInput, astPluginRefToInput } from './builderPageRoutePreview'
import { ManualPluginAdder } from './builderPageRouteSharedControls'
import type { AvailablePlugin, AvailableSignal } from './builderPageTypes'

const MODEL_FIELDS = new Set([
  'model',
  'analysis_models',
  'synthesis_model',
  'planner',
  'roles',
  'final',
])
const modelFreeAlgorithm = (algorithm: RouteAlgoInput | undefined): RouteAlgoInput | undefined =>
  algorithm
    ? {
        ...algorithm,
        fields: Object.fromEntries(
          Object.entries(algorithm.fields).filter(([key]) => !MODEL_FIELDS.has(key)),
        ) as DSLFieldObject,
      }
    : undefined

interface Props {
  route: ASTRouteDecl
  onUpdate: (input: RouteInput) => void
  availableSignals: AvailableSignal[]
  availablePlugins: AvailablePlugin[]
}

export function RouteEditorForm({ route, onUpdate, availableSignals, availablePlugins }: Props) {
  const [description, setDescription] = useState(route.description ?? '')
  const [priority, setPriority] = useState(route.priority)
  const [whenExpr, setWhenExpr] = useState(() => serializeBoolExpr(route.when))
  const [algorithm, setAlgorithm] = useState<RouteAlgoInput | undefined>(() =>
    modelFreeAlgorithm(astAlgoToInput(route.algorithm)),
  )
  const [plugins, setPlugins] = useState<RoutePluginInput[]>(() =>
    route.plugins.map(astPluginRefToInput),
  )
  const activePlugins = useMemo(() => new Set(plugins.map((plugin) => plugin.name)), [plugins])

  useEffect(() => {
    setDescription(route.description ?? '')
    setPriority(route.priority)
    setWhenExpr(serializeBoolExpr(route.when))
    setAlgorithm(modelFreeAlgorithm(astAlgoToInput(route.algorithm)))
    setPlugins(route.plugins.map(astPluginRefToInput))
  }, [route])

  const save = () =>
    onUpdate({
      description: description.trim() || undefined,
      priority,
      when: whenExpr.trim() || undefined,
      models: [],
      algorithm: modelFreeAlgorithm(algorithm),
      plugins,
    })

  const togglePlugin = (pluginName: string) =>
    setPlugins((current) =>
      current.some((plugin) => plugin.name === pluginName)
        ? current.filter((plugin) => plugin.name !== pluginName)
        : [...current, { name: pluginName }],
    )

  return (
    <>
      <section className={styles.dslPreview}>
        <div className={styles.dslPreviewHeader}>
          <span className={styles.dslPreviewTitle}>Decision</span>
          <button type="button" className={styles.toolbarBtnPrimary} onClick={save}>
            <ProductIcon name="check" /> Apply
          </button>
        </div>
        <div className={styles.builderRecipeFormGrid}>
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
        Model assignments are managed on the Mixture-of-Models page.
      </p>
    </>
  )
}
