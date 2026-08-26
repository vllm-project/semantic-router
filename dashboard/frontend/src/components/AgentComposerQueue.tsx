import type { PlaygroundAttachment } from './playgroundFileAttachments'
import ProductIcon from './ProductIcon'
import styles from './AgentPlayground.module.css'

export interface PlaygroundQueuedTurn {
  attachments: PlaygroundAttachment[]
  id: string
  input: string
}

function queuedTurnLabel(turn: PlaygroundQueuedTurn): string {
  const prompt = turn.input.replace(/\s+/g, ' ').trim()
  if (prompt) return prompt
  const count = turn.attachments.length
  return `${count} attachment${count === 1 ? '' : 's'}`
}

export default function AgentComposerQueue({
  paused,
  turns,
  onRemove,
  onResume,
}: {
  paused: boolean
  turns: PlaygroundQueuedTurn[]
  onRemove: (turnId: string) => void
  onResume: () => void
}) {
  if (!turns.length) return null

  return (
    <section
      className={styles.composerQueue}
      role="region"
      aria-label="Queued messages"
      aria-live="polite"
    >
      <header>
        <span>
          Queued <small>{turns.length}</small>
        </span>
        {paused ? (
          <button type="button" onClick={onResume}>
            Resume
          </button>
        ) : null}
      </header>
      <ol>
        {turns.map((turn, index) => {
          const label = queuedTurnLabel(turn)
          return (
            <li key={turn.id} data-testid={`playground-queued-message-${turn.id}`}>
              <small>{index + 1}</small>
              <span title={label}>{label}</span>
              {turn.attachments.length ? (
                <em>
                  {turn.attachments.length} file{turn.attachments.length === 1 ? '' : 's'}
                </em>
              ) : null}
              <button
                type="button"
                onClick={() => onRemove(turn.id)}
                aria-label={`Remove queued message: ${label}`}
              >
                <ProductIcon name="close" />
              </button>
            </li>
          )
        })}
      </ol>
    </section>
  )
}
