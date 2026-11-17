import { useMemo, useState } from 'react'
import { Ellipsis, Pencil, Search, Trash2 } from 'lucide-react'

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  StatusDot,
  cn,
} from '../ui'
import ConfirmDialog from './ConfirmDialog'
import type { ConversationPreview } from './ChatComponentTypes'
import { groupConversations } from './chatComponentSupport'

interface ChatConversationSidebarProps {
  conversationId: string
  conversationPreviews: ConversationPreview[]
  /** Conversations with a response still running, which keep streaming in the background. */
  runningConversationIds?: ReadonlySet<string>
  onDeleteConversation: (id: string) => void
  onRenameConversation: (id: string, title: string) => void
  onSelectConversation: (id: string) => void
}

export default function ChatConversationSidebar({
  conversationId,
  conversationPreviews,
  runningConversationIds,
  onDeleteConversation,
  onRenameConversation,
  onSelectConversation,
}: ChatConversationSidebarProps) {
  const [query, setQuery] = useState('')
  const [renaming, setRenaming] = useState<{ id: string; title: string } | null>(null)
  const [deleting, setDeleting] = useState<ConversationPreview | null>(null)

  const groups = useMemo(() => {
    const normalized = query.trim().toLowerCase()
    return groupConversations(
      normalized
        ? conversationPreviews.filter((conversation) =>
            conversation.preview.toLowerCase().includes(normalized),
          )
        : conversationPreviews,
    )
  }, [conversationPreviews, query])

  const finishRename = () => {
    if (!renaming) return
    const title = renaming.title.trim()
    if (title) onRenameConversation(renaming.id, title)
    setRenaming(null)
  }

  return (
    <>
      {conversationPreviews.length > 0 ? (
        <div className="relative shrink-0 px-3 pb-2">
          <Search className="pointer-events-none absolute left-5.5 top-1/2 size-3.5 -translate-y-1/2 text-subtle-foreground" />
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search conversations"
            aria-label="Search conversations"
            className="h-8 w-full rounded-md bg-transparent pl-8 pr-2 text-[13px] outline-none transition-colors placeholder:text-subtle-foreground hover:bg-accent/60 focus:bg-accent/60 focus-visible:ring-[3px] focus-visible:ring-ring/40"
          />
        </div>
      ) : null}
      <nav aria-label="Saved conversations" className="min-h-0 flex-1 overflow-y-auto px-2 pb-3">
        {groups.length === 0 ? (
          <p className="px-2 py-6 text-center text-xs leading-relaxed text-subtle-foreground">
            {query.trim()
              ? 'No conversation matches.'
              : 'Your conversations are saved in this browser and show up here.'}
          </p>
        ) : (
          groups.map((group) => (
            <div key={group.label} className="mb-3">
              <div className="px-2 pb-1 text-[11px] font-medium text-subtle-foreground">
                {group.label}
              </div>
              {group.items.map((conversation) => {
                const active = conversation.id === conversationId
                const editing = renaming?.id === conversation.id
                return (
                  <div
                    key={conversation.id}
                    className={cn(
                      'group relative flex h-8 items-center rounded-md transition-colors',
                      active
                        ? 'bg-accent text-foreground'
                        : 'text-muted-foreground hover:bg-accent/70 hover:text-foreground',
                    )}
                  >
                    {editing ? (
                      <input
                        value={renaming.title}
                        onChange={(event) =>
                          setRenaming({ id: conversation.id, title: event.target.value })
                        }
                        onBlur={finishRename}
                        onKeyDown={(event) => {
                          if (event.key === 'Enter') finishRename()
                          if (event.key === 'Escape') setRenaming(null)
                        }}
                        maxLength={80}
                        aria-label="Conversation name"
                        autoFocus
                        className="mx-1 h-7 min-w-0 flex-1 rounded-md border border-ring bg-surface px-2 text-[13px] text-foreground outline-none ring-[3px] ring-ring/30"
                      />
                    ) : (
                      <button
                        type="button"
                        className="flex h-full min-w-0 flex-1 items-center rounded-md pl-2 pr-1 text-left text-[13px] outline-none focus-visible:ring-[3px] focus-visible:ring-ring/40"
                        aria-current={active ? 'true' : undefined}
                        onClick={() => onSelectConversation(conversation.id)}
                      >
                        <span className="truncate">{conversation.preview}</span>
                      </button>
                    )}
                    {runningConversationIds?.has(conversation.id) && !editing ? (
                      <span className="mr-1.5 flex" title="Responding">
                        <StatusDot tone="info" pulse />
                      </span>
                    ) : null}
                    {editing ? null : (
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <button
                            type="button"
                            aria-label={`Conversation options for ${conversation.preview}`}
                            className="mr-1 grid size-6 shrink-0 place-items-center rounded text-subtle-foreground opacity-0 outline-none transition-opacity hover:bg-surface hover:text-foreground focus-visible:opacity-100 focus-visible:ring-[3px] focus-visible:ring-ring/40 group-hover:opacity-100 data-[state=open]:opacity-100"
                          >
                            <Ellipsis className="size-4" />
                          </button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="start" className="w-40">
                          <DropdownMenuItem
                            onSelect={() =>
                              setRenaming({ id: conversation.id, title: conversation.preview })
                            }
                          >
                            <Pencil />
                            Rename
                          </DropdownMenuItem>
                          <DropdownMenuSeparator />
                          <DropdownMenuItem
                            variant="destructive"
                            onSelect={() => setDeleting(conversation)}
                          >
                            <Trash2 />
                            Delete
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    )}
                  </div>
                )
              })}
            </div>
          ))
        )}
      </nav>
      <ConfirmDialog
        isOpen={Boolean(deleting)}
        title="Delete this conversation?"
        description={
          deleting
            ? `“${deleting.preview}” and its queued prompts are removed from this browser.`
            : ''
        }
        confirmLabel="Delete"
        tone="danger"
        onCancel={() => setDeleting(null)}
        onConfirm={() => {
          if (deleting) onDeleteConversation(deleting.id)
          setDeleting(null)
        }}
      />
    </>
  )
}
