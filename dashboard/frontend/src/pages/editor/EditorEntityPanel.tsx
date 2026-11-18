import * as React from 'react'
import { ArrowLeft, Code, SearchX, Trash2 } from 'lucide-react'
import { Badge, Button, EmptyState, IconTile, Tooltip } from '../../ui'
import type { RouteInput } from '@/lib/dslMutations'
import type {
  ASTModelDecl,
  ASTPluginDecl,
  ASTProjectionMappingDecl,
  ASTProjectionPartitionDecl,
  ASTProjectionScoreDecl,
  ASTRouteDecl,
  ASTSignalDecl,
  DSLFieldObject,
} from '@/types/dsl'
import type {
  AvailablePlugin,
  BuilderSelectedEntity,
  EntityKind,
  Selection,
} from '../builderPageTypes'
import { ModelForm, PluginForm, ProjectionForm, SignalForm } from './EditorForms'
import { EditorRouteForm } from './EditorRouteForm'
import { groupMeta, groupOf, kindLabel, type EditorEntity } from './editorModel'
import type { ProjectionDeclKind } from './editorSchemas'

export interface EntityHandlers {
  onUpdateModel: (name: string, fields: DSLFieldObject) => void
  onUpdateSignal: (signalType: string, name: string, fields: DSLFieldObject) => void
  onUpdatePlugin: (name: string, pluginType: string, fields: DSLFieldObject) => void
  onUpdateProjection: (kind: ProjectionDeclKind, name: string, fields: DSLFieldObject) => void
  onUpdateRoute: (name: string, input: RouteInput) => void
  onAddModel: (name: string, fields: DSLFieldObject) => void
  onAddSignal: (signalType: string, name: string, fields: DSLFieldObject) => void
  onAddPlugin: (name: string, pluginType: string, fields: DSLFieldObject) => void
  onAddProjection: (kind: ProjectionDeclKind, name: string, fields: DSLFieldObject) => void
  onAddRoute: (name: string, input: RouteInput) => void
}

const projectionKind = (kind: EntityKind): ProjectionDeclKind =>
  kind.replace('projection-', '') as ProjectionDeclKind

function PanelHeader({
  kind,
  title,
  subtitle,
  badge,
  actions,
  onBack,
}: {
  kind: EntityKind
  title: React.ReactNode
  subtitle?: React.ReactNode
  badge?: string
  actions?: React.ReactNode
  onBack: () => void
}) {
  const meta = groupMeta[groupOf(kind)]
  return (
    <header className="flex flex-wrap items-start justify-between gap-3">
      <div className="flex min-w-0 items-start gap-3">
        <Tooltip content="Back to the overview">
          <Button variant="ghost" size="icon-sm" aria-label="Back to the overview" onClick={onBack}>
            <ArrowLeft />
          </Button>
        </Tooltip>
        <IconTile icon={meta.icon} tone={meta.tone} size="md" />
        <div className="flex min-w-0 flex-col gap-1">
          <div className="flex min-w-0 flex-wrap items-center gap-2">
            <h2 className="min-w-0 truncate font-mono text-[15px] font-semibold tracking-[-0.01em]">
              {title}
            </h2>
            {badge ? (
              <Badge
                tone="outline"
                size="sm"
                className="font-mono"
                data-testid="editor-entity-type"
              >
                {badge}
              </Badge>
            ) : null}
          </div>
          {subtitle ? <p className="text-xs text-muted-foreground">{subtitle}</p> : null}
        </div>
      </div>
      {actions ? <div className="flex shrink-0 items-center gap-1.5">{actions}</div> : null}
    </header>
  )
}

/** One entity of the working copy: its header, its source location and its form. */
export function EditorEntityPanel({
  selection,
  entity,
  located,
  availableModels,
  availablePlugins,
  handlers,
  onDelete,
  onShowSource,
  onBack,
}: {
  selection: Selection
  entity: BuilderSelectedEntity
  located: EditorEntity | null
  availableModels: string[]
  availablePlugins: AvailablePlugin[]
  handlers: EntityHandlers
  onDelete: (kind: EntityKind, name: string, subType?: string) => void
  onShowSource: (entity: EditorEntity) => void
  onBack: () => void
}) {
  if (!entity) {
    return (
      <EmptyState
        icon={SearchX}
        tone="slate"
        title={`“${selection.name}” is not in the current program`}
        description="It may have been renamed or removed in the DSL, or the compiler has not parsed the latest edit yet."
        actions={
          <Button variant="outline" size="sm" onClick={onBack}>
            Back to the overview
          </Button>
        }
      />
    )
  }
  const kind = selection.kind
  const subType =
    'signalType' in entity
      ? (entity as ASTSignalDecl).signalType
      : 'pluginType' in entity
        ? (entity as ASTPluginDecl).pluginType
        : undefined
  const badge =
    kind === 'signal' || kind === 'plugin'
      ? subType
      : kind.startsWith('projection-')
        ? `projection ${projectionKind(kind)}`
        : kind === 'route'
          ? `P${(entity as ASTRouteDecl).priority}`
          : undefined
  const description =
    kind === 'route' && (entity as ASTRouteDecl).description
      ? (entity as ASTRouteDecl).description
      : `${kindLabel[kind]} · ${groupMeta[groupOf(kind)].dsl} block`

  let form: React.ReactNode = null
  if (kind === 'model') {
    form = (
      <ModelForm
        model={entity as ASTModelDecl}
        onSave={(fields) => handlers.onUpdateModel(selection.name, fields)}
      />
    )
  } else if (kind === 'signal') {
    const signal = entity as ASTSignalDecl
    form = (
      <SignalForm
        signal={signal}
        onSave={(fields) => handlers.onUpdateSignal(signal.signalType, signal.name, fields)}
      />
    )
  } else if (kind === 'plugin') {
    const plugin = entity as ASTPluginDecl
    form = (
      <PluginForm
        plugin={plugin}
        onSave={(fields) => handlers.onUpdatePlugin(plugin.name, plugin.pluginType, fields)}
      />
    )
  } else if (kind === 'route') {
    form = (
      <EditorRouteForm
        route={entity as ASTRouteDecl}
        availableModels={availableModels}
        availablePlugins={availablePlugins}
        onSave={(input) => handlers.onUpdateRoute(selection.name, input)}
      />
    )
  } else {
    const projection = projectionKind(kind)
    form = (
      <ProjectionForm
        kind={projection}
        entity={
          entity as ASTProjectionPartitionDecl | ASTProjectionScoreDecl | ASTProjectionMappingDecl
        }
        onSave={(fields) => handlers.onUpdateProjection(projection, selection.name, fields)}
      />
    )
  }

  return (
    <div className="mx-auto flex w-full max-w-[880px] flex-col gap-4">
      <PanelHeader
        kind={kind}
        title={selection.name}
        subtitle={description}
        badge={badge}
        onBack={onBack}
        actions={
          <>
            {located ? (
              <Tooltip content="Show this block in the DSL">
                <Button variant="ghost" size="sm" onClick={() => onShowSource(located)}>
                  <Code />
                  <span className="font-mono text-xs tabular">Ln {located.line}</span>
                </Button>
              </Tooltip>
            ) : null}
            <Button
              variant="destructive-ghost"
              size="sm"
              onClick={() => onDelete(kind, selection.name, subType)}
            >
              <Trash2 />
              Delete
            </Button>
          </>
        }
      />
      {form}
    </div>
  )
}

/** A new entity's form, before it exists in the working copy. */
export function EditorCreatePanel({
  kind,
  availableModels,
  availablePlugins,
  handlers,
  onCancel,
}: {
  kind: EntityKind
  availableModels: string[]
  availablePlugins: AvailablePlugin[]
  handlers: EntityHandlers
  onCancel: () => void
}) {
  const label = kindLabel[kind]
  let form: React.ReactNode
  if (kind === 'model') form = <ModelForm onCreate={handlers.onAddModel} onCancel={onCancel} />
  else if (kind === 'signal')
    form = <SignalForm onCreate={handlers.onAddSignal} onCancel={onCancel} />
  else if (kind === 'plugin')
    form = <PluginForm onCreate={handlers.onAddPlugin} onCancel={onCancel} />
  else if (kind === 'route')
    form = (
      <EditorRouteForm
        availableModels={availableModels}
        availablePlugins={availablePlugins}
        onCreate={handlers.onAddRoute}
        onCancel={onCancel}
      />
    )
  else {
    const projection = projectionKind(kind)
    form = (
      <ProjectionForm
        kind={projection}
        onCreate={(name, fields) => handlers.onAddProjection(projection, name, fields)}
        onCancel={onCancel}
      />
    )
  }
  return (
    <div className="mx-auto flex w-full max-w-[880px] flex-col gap-4">
      <PanelHeader
        kind={kind}
        title={<span className="font-sans">New {label.toLowerCase()}</span>}
        subtitle={`Adds a ${groupMeta[groupOf(kind)].dsl} block to the working copy of this scope.`}
        onBack={onCancel}
      />
      {form}
    </div>
  )
}
