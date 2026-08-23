import ProductIcon from '@/components/ProductIcon'
import type { EditorMode } from '@/types/dsl'
import type { RoutingRecipe } from '@/utils/routingManagementApi'

import styles from './BuilderPage.module.css'

interface BuilderToolbarProps {
  readOnly: boolean
  immutable: boolean
  dirty: boolean
  mode: EditorMode
  wasmReady: boolean
  wasmError: string | null
  dslSource: string
  loading: boolean
  saving: boolean
  recipes: RoutingRecipe[]
  selectedRecipeId: string
  guideOpen: boolean
  outputPanelOpen: boolean
  onRecipeChange: (recipeId: string) => void
  onModeSwitch: (mode: EditorMode) => void
  onImport: () => void
  onCompile: () => void
  onSave: () => void
  onDuplicate: () => void
  onFormat: () => void
  onValidate: () => void
  onToggleGuide: () => void
  onToggleOutput: () => void
  onRevert: () => void
}

export function BuilderToolbar({
  readOnly,
  immutable,
  dirty,
  mode,
  wasmReady,
  wasmError,
  dslSource,
  loading,
  saving,
  recipes,
  selectedRecipeId,
  guideOpen,
  outputPanelOpen,
  onRecipeChange,
  onModeSwitch,
  onImport,
  onCompile,
  onSave,
  onDuplicate,
  onFormat,
  onValidate,
  onToggleGuide,
  onToggleOutput,
  onRevert,
}: BuilderToolbarProps) {
  const editable = !readOnly && !immutable
  const ready = wasmReady && Boolean(dslSource.trim()) && !loading && !saving

  return (
    <div className={styles.toolbar} aria-label="Recipe Builder toolbar">
      <div className={styles.toolbarTitle}>
        <ProductIcon name="mixture" />
        Recipe Builder
        {dirty && editable ? <span className={styles.builderUnsaved}>Unsaved</span> : null}
      </div>

      <label className={styles.builderRecipePicker}>
        <span>Recipe</span>
        <select
          value={selectedRecipeId}
          onChange={(event) => onRecipeChange(event.target.value)}
          aria-label="Recipe"
        >
          {recipes.map((recipe) => (
            <option key={recipe.id} value={recipe.id}>
              {recipe.name}
              {recipe.immutable ? ' · Built-in' : ''}
            </option>
          ))}
        </select>
      </label>

      <span className={styles.divider} />

      <div className={styles.modeSwitcher} aria-label="Editor mode">
        <button
          type="button"
          className={mode === 'visual' ? styles.modeBtnActive : styles.modeBtn}
          onClick={() => onModeSwitch('visual')}
        >
          <ProductIcon name="mixture" /> Visual
        </button>
        <button
          type="button"
          className={mode === 'dsl' ? styles.modeBtnActive : styles.modeBtn}
          onClick={() => onModeSwitch('dsl')}
        >
          <ProductIcon name="code" /> DSL
        </button>
      </div>

      {wasmError ? (
        <span className={styles.statusError}>Compiler unavailable</span>
      ) : !wasmReady ? (
        <span className={styles.statusLoading}>Loading compiler…</span>
      ) : null}

      <div className={styles.toolbarRight}>
        {editable ? (
          <>
            <button
              type="button"
              className={styles.toolbarBtn}
              onClick={onImport}
              disabled={!wasmReady}
            >
              <ProductIcon name="download" /> Import
            </button>
            <button
              type="button"
              className={styles.toolbarBtn}
              onClick={onCompile}
              disabled={!ready}
            >
              <ProductIcon name="check" /> Compile
            </button>
            <button
              type="button"
              className={styles.toolbarBtnPrimary}
              onClick={onSave}
              disabled={!ready || !dirty}
            >
              <ProductIcon name="check" /> {saving ? 'Saving…' : 'Save'}
            </button>
          </>
        ) : null}
        {!readOnly ? (
          <button
            type="button"
            className={styles.toolbarBtn}
            onClick={onDuplicate}
            disabled={!ready}
          >
            <ProductIcon name="copy" /> Duplicate
          </button>
        ) : null}
        {editable ? (
          <>
            <button
              type="button"
              className={styles.toolbarBtn}
              onClick={onFormat}
              disabled={!ready}
            >
              Format
            </button>
            <button
              type="button"
              className={styles.toolbarBtn}
              onClick={onValidate}
              disabled={!ready}
            >
              Validate
            </button>
            <button
              type="button"
              className={styles.toolbarBtn}
              onClick={onRevert}
              disabled={!dirty || saving}
            >
              <ProductIcon name="undo" /> Revert
            </button>
            <button
              type="button"
              className={guideOpen ? styles.toolbarBtnActive : styles.toolbarBtn}
              onClick={onToggleGuide}
            >
              Guide
            </button>
          </>
        ) : null}
        <button
          type="button"
          className={outputPanelOpen ? styles.toolbarBtnActive : styles.toolbarBtn}
          onClick={onToggleOutput}
        >
          <ProductIcon name="code" /> Recipe
        </button>
      </div>
    </div>
  )
}
