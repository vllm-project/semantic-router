import { useId } from 'react'
import { Check, Layers, Plus, Search } from 'lucide-react'

import NativeSelect from '../components/NativeSelect'
import { configCollectionItemYaml } from '../lib/configYaml'
import {
  Alert,
  Badge,
  Button,
  Checkbox,
  ChipInput,
  CodeView,
  DetailSheet,
  DetailSheetBody,
  DetailSheetFooter,
  DetailSheetHeader,
  Field,
  IconTile,
  Input,
  InputGroup,
  SectionLabel,
  cn,
} from '../ui'
import {
  DEFAULT_RECIPE_NAME,
  normalizeEntrypointModelNames,
} from './configPageEntrypointsRecipesSupport'
import { mixtureRecipeKey } from './configPageMixtureSupport'
import {
  useMixtureEditor,
  type ConfigPageMixtureDialogProps,
  type MixtureEditor,
} from './useMixtureEditor'
import { ModelLogo } from './ConfigPageModelLogo'

function entrypointPreview(editor: MixtureEditor): string {
  const names = normalizeEntrypointModelNames([editor.primaryName, ...editor.aliases])
  const entry: Record<string, unknown> = {
    model_names: names.length ? names : ['acme/auto'],
    recipe: editor.choice?.name ?? DEFAULT_RECIPE_NAME,
  }
  if (editor.reviewModel) entry.review_model = editor.reviewModel
  if (editor.memoryModel) entry.memory_model = editor.memoryModel
  return configCollectionItemYaml(['entrypoints'], entry)
}

function AssignmentWorkspace({ editor }: { editor: MixtureEditor }) {
  const { activeDecision, decisions } = editor
  if (!activeDecision || decisions.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-border-strong px-4 py-6 text-center text-[13px] text-muted-foreground">
        Add decisions to this recipe first.
      </div>
    )
  }
  return (
    <div className="grid min-h-[320px] grid-cols-1 overflow-hidden rounded-xl border border-border bg-card md:grid-cols-[minmax(0,240px)_1fr]">
      <div className="flex min-h-0 flex-col border-b border-border bg-surface-subtle/50 md:border-b-0 md:border-r">
        <div className="border-b border-border p-2">
          <InputGroup
            value={editor.decisionSearch}
            onChange={(event) => editor.setDecisionSearch(event.target.value)}
            placeholder="Find a decision"
            aria-label="Find a decision"
            leading={<Search />}
            className="h-8"
          />
        </div>
        <div
          className="flex max-h-[360px] flex-col gap-0.5 overflow-y-auto p-1.5"
          role="tablist"
          aria-label="Recipe decisions"
          aria-orientation="vertical"
        >
          {editor.visibleDecisions.map((decision) => {
            const decisionIndex = decisions.findIndex(
              (candidate) => candidate.name === decision.name,
            )
            const selected = editor.assignments[decision.name] ?? []
            const needsModels = decision.minimumCandidates > 0
            const ready = selected.length >= decision.minimumCandidates
            const isActive = decision.name === activeDecision.name
            return (
              <button
                key={decision.name}
                type="button"
                role="tab"
                aria-selected={isActive}
                onClick={() => {
                  editor.setActiveDecisionName(decision.name)
                  editor.setModelSearch('')
                  editor.setShowSelectedModels(false)
                }}
                className={cn(
                  'flex items-center gap-2.5 rounded-md px-2.5 py-2 text-left outline-none transition-colors focus-visible:ring-[3px] focus-visible:ring-ring/40',
                  isActive ? 'bg-surface shadow-xs ring-1 ring-border' : 'hover:bg-accent',
                )}
              >
                <span className="font-mono text-[11px] tabular text-subtle-foreground">
                  {String(decisionIndex + 1).padStart(2, '0')}
                </span>
                <span className="flex min-w-0 flex-1 flex-col">
                  <strong className="truncate font-mono text-[12.5px] font-medium">
                    {decision.name}
                  </strong>
                  <small className="truncate text-[11px] text-muted-foreground">
                    {decision.description || 'Model path'}
                  </small>
                </span>
                {needsModels ? (
                  <Badge size="sm" tone={ready ? 'success' : 'warning'} className="tabular">
                    {selected.length || '—'}
                  </Badge>
                ) : (
                  <Badge size="sm" tone="outline">
                    No model
                  </Badge>
                )}
              </button>
            )
          })}
          {editor.visibleDecisions.length === 0 ? (
            <p className="px-2 py-4 text-center text-xs text-muted-foreground">
              No matching decisions.
            </p>
          ) : null}
        </div>
      </div>

      <div
        className="flex min-h-0 flex-col"
        role="tabpanel"
        aria-label={`${activeDecision.name} model assignment`}
      >
        <div className="flex flex-wrap items-start justify-between gap-2 border-b border-border px-4 py-3">
          <div className="flex min-w-0 flex-col gap-0.5">
            <strong className="truncate font-mono text-[13px] font-medium">
              {activeDecision.name}
            </strong>
            <span className="text-xs text-muted-foreground">
              {activeDecision.description || 'Choose the models for this path.'}
            </span>
          </div>
          <Badge
            tone={
              activeDecision.minimumCandidates === 0
                ? 'outline'
                : editor.activeSelection.length >= activeDecision.minimumCandidates
                  ? 'success'
                  : 'warning'
            }
          >
            {activeDecision.minimumCandidates > 0
              ? `${editor.activeSelection.length} selected · minimum ${activeDecision.minimumCandidates}`
              : 'No model needed'}
          </Badge>
        </div>
        {activeDecision.minimumCandidates > 0 ? (
          <>
            <div className="flex items-center gap-2 px-4 pt-3">
              <InputGroup
                value={editor.modelSearch}
                onChange={(event) => editor.setModelSearch(event.target.value)}
                placeholder="Find a model"
                aria-label={`Find a model for ${activeDecision.name}`}
                leading={<Search />}
                containerClassName="flex-1"
                className="h-8"
              />
              <Button
                type="button"
                variant={editor.showSelectedModels ? 'soft' : 'ghost'}
                size="sm"
                aria-pressed={editor.showSelectedModels}
                onClick={() => editor.setShowSelectedModels((current) => !current)}
              >
                <Check />
                Selected
                <span className="font-mono text-[11px] tabular">
                  {editor.activeSelection.length}
                </span>
              </Button>
            </div>
            <div className="grid max-h-[300px] grid-cols-1 gap-1.5 overflow-y-auto p-4 pt-3 sm:grid-cols-2">
              {editor.visibleModelOptions.map((modelName) => {
                const checked = editor.activeSelection.includes(modelName)
                const model = editor.models.find((candidate) => candidate.name === modelName)
                return (
                  <label
                    key={modelName}
                    className={cn(
                      'flex min-w-0 cursor-pointer items-center gap-2.5 rounded-lg border px-2.5 py-2 transition-colors',
                      checked
                        ? 'border-primary/40 bg-primary-soft/50'
                        : 'border-border bg-surface hover:border-border-strong',
                    )}
                  >
                    <Checkbox
                      checked={checked}
                      onCheckedChange={() => editor.toggleModel(activeDecision.name, modelName)}
                    />
                    {model ? <ModelLogo model={model} size={14} /> : null}
                    <code className="truncate font-mono text-[12.5px]">{modelName}</code>
                  </label>
                )
              })}
              {editor.visibleModelOptions.length === 0 ? (
                <p className="col-span-full py-6 text-center text-xs text-muted-foreground">
                  {editor.modelOptions.length === 0
                    ? 'Connect a model before publishing this mixture.'
                    : 'No models match this view.'}
                </p>
              ) : null}
            </div>
          </>
        ) : (
          <p className="px-4 py-8 text-center text-[13px] text-muted-foreground">
            This decision responds without calling a model.
          </p>
        )}
      </div>
    </div>
  )
}

/** The editor's fields; the sheet adds the header, error and footer around them. */
export function MixtureEditorBody({ editor }: { editor: MixtureEditor }) {
  const nameId = useId()
  const recipeId = useId()
  const aliasesId = useId()
  const reviewId = useId()
  const memoryId = useId()
  const configRecipeOptions = editor.recipeNames.map((name) => (
    <option key={name} value={mixtureRecipeKey({ name })}>
      {name}
    </option>
  ))
  const builtIn = editor.choice?.bundle !== undefined
  return (
    <div className="flex flex-col gap-6">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <Field
          label="Model name"
          htmlFor={nameId}
          error={editor.nameProblem}
          hint="Apps send this name in the model field."
        >
          <Input
            id={nameId}
            value={editor.primaryName}
            onChange={(event) => editor.setPrimaryName(event.target.value)}
            placeholder="my-company/auto"
            autoComplete="off"
            spellCheck={false}
            aria-invalid={editor.nameProblem ? true : undefined}
            className="font-mono text-[12.5px]"
            data-dialog-initial-focus
          />
        </Field>
        <Field label="Recipe" htmlFor={recipeId} hint={editor.description}>
          <NativeSelect
            id={recipeId}
            value={editor.choice ? mixtureRecipeKey(editor.choice) : ''}
            onChange={(event) => editor.chooseRecipe(event.target.value)}
          >
            {editor.catalog?.bundles.length ? (
              <>
                {configRecipeOptions.length ? (
                  <optgroup label="Configured recipes">{configRecipeOptions}</optgroup>
                ) : null}
                {editor.catalog.bundles.map((bundle) => (
                  <optgroup key={bundle.name} label={`Built-in · ${bundle.name}`}>
                    {bundle.recipes.map((recipe) => (
                      <option
                        key={recipe.name}
                        value={mixtureRecipeKey({ bundle: bundle.name, name: recipe.name })}
                      >
                        {recipe.name}
                      </option>
                    ))}
                  </optgroup>
                ))}
              </>
            ) : (
              configRecipeOptions
            )}
          </NativeSelect>
        </Field>
        <Field
          label="Aliases"
          htmlFor={aliasesId}
          optional
          hint="Other names that call this same model. Press Enter after each."
          className="sm:col-span-2"
        >
          <ChipInput
            id={aliasesId}
            value={editor.aliases}
            onChange={editor.setAliases}
            placeholder="Add an alias"
            caseSensitive
            validate={(candidate) =>
              candidate === editor.primaryName.trim()
                ? 'An alias must differ from the model name.'
                : null
            }
          />
        </Field>
        {builtIn ? null : (
          <>
            <Field label="Reviews go to" htmlFor={reviewId} hint="Agent approval reviews.">
              <NativeSelect
                id={reviewId}
                value={editor.reviewModel}
                onChange={(event) => editor.setReviewModel(event.target.value)}
              >
                <option value="">Session model</option>
                {editor.delegateNames.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </NativeSelect>
            </Field>
            <Field
              label="Memories go to"
              htmlFor={memoryId}
              hint="Memory extraction and consolidation."
            >
              <NativeSelect
                id={memoryId}
                value={editor.memoryModel}
                onChange={(event) => editor.setMemoryModel(event.target.value)}
              >
                <option value="">Session model</option>
                {editor.delegateNames.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </NativeSelect>
            </Field>
          </>
        )}
      </div>

      <section className="flex flex-col gap-2.5">
        <div className="flex flex-wrap items-end justify-between gap-2">
          <div className="flex flex-col gap-0.5">
            <SectionLabel>Model assignments</SectionLabel>
            <span className="text-[13px] font-medium tabular">
              {editor.completedDecisionCount} of {editor.decisions.length} ready
            </span>
          </div>
          <span className="max-w-md text-right text-xs text-muted-foreground">
            {editor.decisions.length && editor.completedDecisionCount === editor.decisions.length
              ? 'Every decision has the models it needs.'
              : 'Complete every decision before publishing.'}
          </span>
        </div>
        <AssignmentWorkspace editor={editor} />
      </section>

      <section className="flex flex-col gap-2">
        <SectionLabel>{builtIn ? 'Publishes to config.yaml' : 'Adds to config.yaml'}</SectionLabel>
        <div className="overflow-hidden rounded-lg border border-border bg-surface-sunken/60">
          <CodeView code={entrypointPreview(editor)} language="yaml" />
        </div>
        {builtIn ? (
          <p className="text-xs text-muted-foreground">
            The installed vllm-sr package composes the recipe with your model assignments when you
            create the model.
          </p>
        ) : null}
      </section>
    </div>
  )
}

export default function ConfigPageMixtureDialog(props: ConfigPageMixtureDialogProps) {
  const editor = useMixtureEditor(props)
  const editing = Boolean(props.entrypoint)
  return (
    <DetailSheet
      open
      onOpenChange={(open) => {
        if (!open && !editor.saving) props.onClose()
      }}
      size="xl"
    >
      <DetailSheetHeader
        icon={<IconTile icon={editing ? Layers : Plus} tone="blue" size="lg" />}
        title={editing ? 'Edit public model' : 'New public model'}
        description="Choose a recipe. Assign the right models."
      />
      <DetailSheetBody className="flex flex-col gap-4 px-5 py-5" aria-busy={editor.saving}>
        {editor.error ? (
          <div role="alert">
            <Alert tone="danger" title="The model was not saved">
              {editor.error}
            </Alert>
          </div>
        ) : null}
        <MixtureEditorBody editor={editor} />
      </DetailSheetBody>
      <DetailSheetFooter>
        <span className="min-w-0 truncate text-xs text-muted-foreground">
          {editor.decisions.length
            ? `${editor.completedDecisionCount} of ${editor.decisions.length} decisions have their models`
            : 'Choose a recipe with at least one decision.'}
        </span>
        <div className="ml-auto flex items-center gap-2">
          <Button variant="ghost" size="sm" onClick={props.onClose} disabled={editor.saving}>
            Cancel
          </Button>
          <Button size="sm" onClick={() => void editor.save()} loading={editor.saving}>
            {editor.saving ? null : editing ? <Check /> : <Plus />}
            {editor.saving ? 'Saving…' : editing ? 'Save public model' : 'Create public model'}
          </Button>
        </div>
      </DetailSheetFooter>
    </DetailSheet>
  )
}
