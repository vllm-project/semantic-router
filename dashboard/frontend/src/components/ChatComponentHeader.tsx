import { PanelLeft, PanelRight, SquarePen } from 'lucide-react'

import { Badge, Button, Tooltip, cn } from '../ui'
import type { RouterModelOption } from '../utils/routerModelSelection'
import ChatComposerModelSelect from './ChatComposerModelSelect'

interface ChatComponentHeaderProps {
  conversationsOpen: boolean
  inspectorOpen: boolean
  models: RouterModelOption[]
  model: string
  recipe?: string
  modelSelectDisabled: boolean
  onModelChange: (model: string) => void
  onNewConversation: () => void
  onToggleConversations: () => void
  onToggleInspector: () => void
}

/** The chat column's bar: the pane toggles and the public model every message goes to. */
export default function ChatComponentHeader({
  conversationsOpen,
  inspectorOpen,
  models,
  model,
  recipe,
  modelSelectDisabled,
  onModelChange,
  onNewConversation,
  onToggleConversations,
  onToggleInspector,
}: ChatComponentHeaderProps) {
  const conversationsLabel = conversationsOpen ? 'Hide conversations' : 'Show conversations'
  const inspectorLabel = inspectorOpen ? 'Hide routing trace' : 'Show routing trace'
  return (
    <div className="flex h-12 shrink-0 items-center gap-1.5 border-b border-border px-3">
      <Tooltip content={conversationsLabel}>
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label={conversationsLabel}
          aria-pressed={conversationsOpen}
          onClick={onToggleConversations}
        >
          <PanelLeft />
        </Button>
      </Tooltip>
      {conversationsOpen ? null : (
        <Tooltip content="New conversation">
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label="New conversation"
            onClick={onNewConversation}
          >
            <SquarePen />
          </Button>
        </Tooltip>
      )}
      <ChatComposerModelSelect
        disabled={modelSelectDisabled}
        models={models}
        onChange={onModelChange}
        value={model}
      />
      {recipe ? (
        <Badge tone="outline" className="hidden max-w-40 sm:inline-flex" title={`Recipe ${recipe}`}>
          <span className="truncate">recipe {recipe}</span>
        </Badge>
      ) : null}
      <div className="ml-auto flex items-center">
        <Tooltip content={inspectorLabel}>
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label={inspectorLabel}
            aria-pressed={inspectorOpen}
            onClick={onToggleInspector}
            className={cn(inspectorOpen && 'text-foreground')}
          >
            <PanelRight />
          </Button>
        </Tooltip>
      </div>
    </div>
  )
}
