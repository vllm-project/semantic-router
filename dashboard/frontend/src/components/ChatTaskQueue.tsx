import { useEffect, useRef, useState } from 'react'
import { CornerDownRight, Ellipsis, GripVertical, Pencil, Trash2 } from 'lucide-react'

import { cn } from '../ui'
import type { PlaygroundTask } from './ChatComponentTypes'

interface ChatTaskQueueProps {
  queuedTasks: PlaygroundTask[]
  onEditTask: (taskId: string) => void
  onDeleteTask: (taskId: string) => void
  onReorderTasks: (sourceTaskId: string, targetTaskId: string) => void
}

const formatPrompt = (prompt: string): string => {
  const normalized = prompt.replace(/\s+/g, ' ').trim()
  if (normalized.length <= 82) {
    return normalized
  }
  return `${normalized.slice(0, 79).trim()}...`
}

const smallButton =
  'grid size-6 shrink-0 place-items-center rounded-md text-subtle-foreground outline-none transition-colors hover:bg-accent hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/40 [&_svg]:size-3.5'

export default function ChatTaskQueue({
  queuedTasks,
  onEditTask,
  onDeleteTask,
  onReorderTasks,
}: ChatTaskQueueProps) {
  const [draggedTaskId, setDraggedTaskId] = useState<string | null>(null)
  const [dropTargetTaskId, setDropTargetTaskId] = useState<string | null>(null)
  const [openMenuTaskId, setOpenMenuTaskId] = useState<string | null>(null)
  const menuRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (!openMenuTaskId) {
      return undefined
    }

    const handlePointerDown = (event: PointerEvent) => {
      if (menuRef.current?.contains(event.target as Node)) {
        return
      }
      setOpenMenuTaskId(null)
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setOpenMenuTaskId(null)
      }
    }

    window.addEventListener('pointerdown', handlePointerDown)
    window.addEventListener('keydown', handleKeyDown)

    return () => {
      window.removeEventListener('pointerdown', handlePointerDown)
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [openMenuTaskId])

  if (queuedTasks.length === 0) {
    return null
  }

  return (
    <section
      className="mx-auto w-full max-w-3xl shrink-0 px-3 pt-1 sm:px-4"
      aria-label="Queued prompts"
      data-testid="playground-task-queue"
    >
      <div className="rounded-xl border border-border bg-surface-subtle/70 p-1.5">
        <div className="flex items-center gap-2 px-2 pb-1 pt-0.5 text-[11px] font-medium text-subtle-foreground">
          <CornerDownRight className="size-3" />
          <span className="tabular">{queuedTasks.length} queued</span>
          <span className="font-normal">· sent in order after the current answer</span>
        </div>
        <div className="flex flex-col gap-1">
          {queuedTasks.map((task) => {
            const promptLabel = formatPrompt(task.prompt)
            const isDragSource = draggedTaskId === task.id
            const isDropTarget = dropTargetTaskId === task.id

            return (
              <div
                key={task.id}
                className={cn(
                  'group relative flex items-center gap-1.5 rounded-lg border bg-card py-1 pl-1 pr-1.5 shadow-xs transition-[border-color,opacity,box-shadow]',
                  isDragSource ? 'opacity-50' : 'opacity-100',
                  isDropTarget ? 'border-primary shadow-[0_0_0_2px_var(--ring)]' : 'border-border',
                )}
                draggable
                data-testid={`playground-task-queue-item-${task.id}`}
                onDragStart={(event) => {
                  setDraggedTaskId(task.id)
                  setDropTargetTaskId(null)
                  setOpenMenuTaskId(null)
                  event.dataTransfer.effectAllowed = 'move'
                  event.dataTransfer.setData('text/plain', task.id)
                }}
                onDragEnd={() => {
                  setDraggedTaskId(null)
                  setDropTargetTaskId(null)
                }}
                onDragOver={(event) => {
                  event.preventDefault()
                  if (draggedTaskId && draggedTaskId !== task.id) {
                    setDropTargetTaskId(task.id)
                  }
                }}
                onDrop={(event) => {
                  event.preventDefault()
                  const sourceTaskId = draggedTaskId || event.dataTransfer.getData('text/plain')
                  if (sourceTaskId && sourceTaskId !== task.id) {
                    onReorderTasks(sourceTaskId, task.id)
                  }
                  setDraggedTaskId(null)
                  setDropTargetTaskId(null)
                }}
              >
                <button
                  type="button"
                  className={cn(smallButton, 'cursor-grab active:cursor-grabbing')}
                  aria-label={`Reorder queued task: ${promptLabel}`}
                  title="Drag to reorder"
                >
                  <GripVertical />
                </button>

                <span className="min-w-0 flex-1 truncate text-[13px]">{promptLabel}</span>

                <div
                  className="relative flex items-center gap-0.5"
                  ref={openMenuTaskId === task.id ? menuRef : null}
                >
                  <button
                    type="button"
                    className={smallButton}
                    aria-label={`Remove queued task: ${promptLabel}`}
                    data-testid={`playground-task-delete-${task.id}`}
                    onClick={() => {
                      setOpenMenuTaskId(null)
                      onDeleteTask(task.id)
                    }}
                    title="Remove queued task"
                  >
                    <Trash2 />
                  </button>

                  <button
                    type="button"
                    className={smallButton}
                    aria-label={`More actions for queued task: ${promptLabel}`}
                    aria-expanded={openMenuTaskId === task.id}
                    aria-haspopup="menu"
                    data-testid={`playground-task-menu-${task.id}`}
                    onClick={() => {
                      setOpenMenuTaskId((current) => (current === task.id ? null : task.id))
                    }}
                    title="More actions"
                  >
                    <Ellipsis />
                  </button>

                  {openMenuTaskId === task.id ? (
                    <div
                      className="absolute bottom-[calc(100%+6px)] right-0 z-40 flex w-44 flex-col gap-0.5 rounded-lg border border-border bg-popover p-1 text-popover-foreground shadow-lg animate-in fade-in-0 zoom-in-95"
                      role="menu"
                      aria-label={`Queued task actions for ${promptLabel}`}
                      data-testid={`playground-task-menu-panel-${task.id}`}
                    >
                      <button
                        type="button"
                        className="flex items-center gap-2 rounded-md px-2 py-1.5 text-left text-[13px] outline-none hover:bg-accent focus-visible:bg-accent [&_svg]:size-3.5 [&_svg]:text-subtle-foreground"
                        role="menuitem"
                        onClick={() => {
                          setOpenMenuTaskId(null)
                          onEditTask(task.id)
                        }}
                      >
                        <Pencil />
                        <span>Edit prompt</span>
                      </button>
                      <button
                        type="button"
                        className="flex items-center gap-2 rounded-md px-2 py-1.5 text-left text-[13px] text-destructive-foreground outline-none hover:bg-destructive-soft focus-visible:bg-destructive-soft [&_svg]:size-3.5"
                        role="menuitem"
                        onClick={() => {
                          setOpenMenuTaskId(null)
                          onDeleteTask(task.id)
                        }}
                      >
                        <Trash2 />
                        <span>Remove from queue</span>
                      </button>
                    </div>
                  ) : null}
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
