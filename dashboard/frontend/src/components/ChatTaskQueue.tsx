import { useMemo, useState } from 'react'

import type { PlaygroundTask } from './ChatComponentTypes'
import styles from './ChatTaskQueue.module.css'

interface ChatTaskQueueProps {
  activeTask: PlaygroundTask | null
  queuedTasks: PlaygroundTask[]
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
  activeTask,
  queuedTasks,
  onDeleteTask,
  onReorderTasks,
}: ChatTaskQueueProps) {
  const [draggedTaskId, setDraggedTaskId] = useState<string | null>(null)
  const [dropTargetTaskId, setDropTargetTaskId] = useState<string | null>(null)

  const summary = useMemo(() => {
    const fragments: string[] = []
    if (activeTask) {
      fragments.push('1 running')
    }
    if (queuedTasks.length > 0) {
      fragments.push(`${queuedTasks.length} waiting`)
    }
    return fragments.join(' • ') || 'Idle'
  }, [activeTask, queuedTasks.length])

  if (!activeTask && queuedTasks.length === 0) {
    return null
  }

  return (
    <section className={styles.queue} data-testid="playground-task-queue">
      <div className={styles.header}>
        <div className={styles.eyebrow}>Task Queue</div>
        <div className={styles.summary}>{summary}</div>
      </div>

      <div className={styles.list}>
        {activeTask ? (
          <div
            className={`${styles.item} ${styles.itemRunning}`}
            data-testid="playground-task-running"
          >
            <div className={styles.runningIndicator} aria-hidden="true" />
            <div className={styles.taskBody}>
              <div className={styles.taskTitle}>
                <span className={styles.taskPrompt}>{formatPrompt(activeTask.prompt)}</span>
                <span className={styles.taskBadge}>Running</span>
              </div>
              <div className={styles.taskMeta}>This task is already in progress.</div>
            </div>
            <div />
          </div>
        ) : null}

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
                <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
                  <circle cx="5" cy="4" r="1.1" />
                  <circle cx="11" cy="4" r="1.1" />
                  <circle cx="5" cy="8" r="1.1" />
                  <circle cx="11" cy="8" r="1.1" />
                  <circle cx="5" cy="12" r="1.1" />
                  <circle cx="11" cy="12" r="1.1" />
                </svg>
              </button>

              <div className={styles.taskBody}>
                <div className={styles.taskTitle}>
                  <span className={styles.taskPrompt}>{promptLabel}</span>
                  <span className={styles.taskBadge}>Queued</span>
                </div>
                <div className={styles.taskMeta}>Drag to reprioritize or remove this task.</div>
              </div>

              <button
                type="button"
                className={styles.deleteButton}
                aria-label={`Remove queued task: ${promptLabel}`}
                data-testid={`playground-task-delete-${task.id}`}
                onClick={() => onDeleteTask(task.id)}
                title="Remove queued task"
              >
                <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path
                    d="M2 4h12M5.5 4V2.5h5V4M13 4v9.5a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V4M6.5 7v4M9.5 7v4"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </button>
            </div>
          )
        })}
      </div>
    </section>
  )
}
