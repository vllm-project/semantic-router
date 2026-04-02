import { useEffect, useMemo, useRef, useState } from 'react'

import type { PlaygroundTask } from './ChatComponentTypes'
import styles from './ChatTaskQueue.module.css'

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

  const summary = useMemo(() => `${queuedTasks.length}`, [queuedTasks.length])

  if (queuedTasks.length === 0) {
    return null
  }

  return (
    <section className={styles.queue} data-testid="playground-task-queue">
      <div className={styles.header}>
        <div className={styles.summary}>{summary}</div>
      </div>

      <div className={styles.list}>
        {queuedTasks.map(task => {
          const promptLabel = formatPrompt(task.prompt)
          const isDragSource = draggedTaskId === task.id
          const isDropTarget = dropTargetTaskId === task.id

          return (
            <div
              key={task.id}
              className={[
                styles.item,
                styles.itemQueued,
                isDragSource ? styles.itemDragSource : '',
                isDropTarget ? styles.itemDropTarget : '',
              ].filter(Boolean).join(' ')}
              draggable
              data-testid={`playground-task-queue-item-${task.id}`}
              onDragStart={event => {
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
              onDragOver={event => {
                event.preventDefault()
                if (draggedTaskId && draggedTaskId !== task.id) {
                  setDropTargetTaskId(task.id)
                }
              }}
              onDrop={event => {
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
                className={styles.dragHandle}
                aria-label={`Reorder queued task: ${promptLabel}`}
                title="Drag to reorder"
              >
                <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor" aria-hidden="true">
                  <circle cx="4.25" cy="3.25" r="0.85" />
                  <circle cx="9.75" cy="3.25" r="0.85" />
                  <circle cx="4.25" cy="7" r="0.85" />
                  <circle cx="9.75" cy="7" r="0.85" />
                  <circle cx="4.25" cy="10.75" r="0.85" />
                  <circle cx="9.75" cy="10.75" r="0.85" />
                </svg>
              </button>

              <div className={styles.taskBody}>
                <div className={styles.taskTitle}>
                  <span className={styles.taskPrompt}>{promptLabel}</span>
                </div>
              </div>

              <div className={styles.actions} ref={openMenuTaskId === task.id ? menuRef : null}>
                <button
                  type="button"
                  className={styles.deleteButton}
                  aria-label={`Remove queued task: ${promptLabel}`}
                  data-testid={`playground-task-delete-${task.id}`}
                  onClick={() => {
                    setOpenMenuTaskId(null)
                    onDeleteTask(task.id)
                  }}
                  title="Remove queued task"
                >
                  <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.45" aria-hidden="true">
                    <path
                      d="M3 4.25h10M6 4.25V3a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1v1.25M12.5 4.25v8.25a1 1 0 0 1-1 1h-7a1 1 0 0 1-1-1V4.25M6.5 7v4M9.5 7v4"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                </button>

                <button
                  type="button"
                  className={styles.menuButton}
                  aria-label={`More actions for queued task: ${promptLabel}`}
                  aria-expanded={openMenuTaskId === task.id}
                  aria-haspopup="menu"
                  data-testid={`playground-task-menu-${task.id}`}
                  onClick={() => {
                    setOpenMenuTaskId(current => (current === task.id ? null : task.id))
                  }}
                  title="More actions"
                >
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
                    <circle cx="4" cy="8" r="1.05" />
                    <circle cx="8" cy="8" r="1.05" />
                    <circle cx="12" cy="8" r="1.05" />
                  </svg>
                </button>

                {openMenuTaskId === task.id ? (
                  <div
                    className={styles.menu}
                    role="menu"
                    aria-label={`Queued task actions for ${promptLabel}`}
                    data-testid={`playground-task-menu-panel-${task.id}`}
                  >
                    <button
                      type="button"
                      className={styles.menuItem}
                      role="menuitem"
                      onClick={() => {
                        setOpenMenuTaskId(null)
                        onEditTask(task.id)
                      }}
                    >
                      <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.45" aria-hidden="true">
                        <path d="M10.9 2.4a1.8 1.8 0 0 1 2.6 2.6l-7.2 7.2-3 .6.6-3z" strokeLinecap="round" strokeLinejoin="round" />
                        <path d="m9.9 3.4 2.7 2.7" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                      <span>Edit prompt</span>
                    </button>
                    <button
                      type="button"
                      className={styles.menuItem}
                      role="menuitem"
                      onClick={() => {
                        setOpenMenuTaskId(null)
                        onDeleteTask(task.id)
                      }}
                    >
                      <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.45" aria-hidden="true">
                        <path
                          d="M3 4.25h10M6 4.25V3a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1v1.25M12.5 4.25v8.25a1 1 0 0 1-1 1h-7a1 1 0 0 1-1-1V4.25M6.5 7v4M9.5 7v4"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </svg>
                      <span>Remove from queue</span>
                    </button>
                  </div>
                ) : null}
              </div>
            </div>
          )
        })}
      </div>
    </section>
  )
}
