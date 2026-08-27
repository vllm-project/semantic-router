import { useCallback, useRef, useState, type ChangeEvent, type KeyboardEvent } from 'react'

import type { RouterModelOption } from '../utils/routerModelSelection'
import AgentComposerQueue, { type PlaygroundQueuedTurn } from './AgentComposerQueue'
import AgentComposerMenu from './AgentComposerMenu'
import ChatComposerModelSelect from './ChatComposerModelSelect'
import {
  formatPlaygroundFileSize,
  isPlaygroundImageAttachment,
  type PlaygroundAttachment,
} from './playgroundFileAttachments'
import ProductIcon from './ProductIcon'
import type { PlaygroundMode } from './playgroundModes'
import styles from './AgentPlayground.module.css'

interface AgentComposerProps {
  attachments: PlaygroundAttachment[]
  builderAvailable: boolean
  disabledReason?: string
  input: string
  mode: PlaygroundMode
  models: RouterModelOption[]
  queuePaused: boolean
  queuedTurns: PlaygroundQueuedTurn[]
  running: boolean
  selectedModel: string
  targetLocked: boolean
  onAttach: (files: FileList) => void
  onInputChange: (value: string) => void
  onModeChange: (mode: PlaygroundMode) => void
  onModelChange: (model: string) => void
  onQueue: () => void
  onQueueRemove: (turnId: string) => void
  onQueueResume: () => void
  onRemoveAttachment: (id: string) => void
  onSend: () => void
  onStop: () => void
}

export default function AgentComposer({
  attachments,
  builderAvailable,
  disabledReason,
  input,
  mode,
  models,
  queuePaused,
  queuedTurns,
  running,
  selectedModel,
  targetLocked,
  onAttach,
  onInputChange,
  onModeChange,
  onModelChange,
  onQueue,
  onQueueRemove,
  onQueueResume,
  onRemoveAttachment,
  onSend,
  onStop,
}: AgentComposerProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [composing, setComposing] = useState(false)
  const canSend = (Boolean(input.trim()) || attachments.length > 0) && !disabledReason && !running
  const canQueue = (Boolean(input.trim()) || attachments.length > 0) && !disabledReason && running

  const handleFiles = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      if (event.target.files?.length) onAttach(event.target.files)
      event.target.value = ''
    },
    [onAttach],
  )

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== 'Enter' || event.shiftKey || composing || event.nativeEvent.isComposing)
      return
    event.preventDefault()
    if (canQueue) onQueue()
    else if (canSend) onSend()
  }

  return (
    <div className={styles.composerWrap} data-testid="agent-composer">
      <div className={`${styles.composer} ${mode !== 'chat' ? styles.composerBuilder : ''}`}>
        {mode !== 'chat' ? (
          <div className={styles.builderModeBar}>
            <span>
              <ProductIcon name="mixture" />
              Builder
            </span>
            <small>Draft · Validate · Test · Review</small>
          </div>
        ) : null}
        {attachments.length > 0 ? (
          <div className={styles.attachments}>
            {attachments.map((attachment) => (
              <div key={attachment.id} className={styles.attachment}>
                {isPlaygroundImageAttachment(attachment) ? (
                  <img src={attachment.content} alt={`Preview of ${attachment.fileName}`} />
                ) : (
                  <ProductIcon name="code" />
                )}
                <span>
                  <strong>{attachment.fileName}</strong>
                  <small>{formatPlaygroundFileSize(attachment.sizeBytes)}</small>
                </span>
                <button
                  type="button"
                  onClick={() => onRemoveAttachment(attachment.id)}
                  aria-label={`Remove ${attachment.fileName}`}
                >
                  <ProductIcon name="close" />
                </button>
              </div>
            ))}
          </div>
        ) : null}
        <textarea
          value={input}
          onChange={(event) => onInputChange(event.target.value)}
          onCompositionStart={() => setComposing(true)}
          onCompositionEnd={() => setComposing(false)}
          onKeyDown={handleKeyDown}
          placeholder={
            mode === 'builder' ? 'Describe the model path you want to build…' : 'Ask anything…'
          }
          rows={1}
          aria-label={mode === 'builder' ? 'Builder instruction' : 'Message'}
        />
        <div className={styles.composerFooter}>
          <div className={styles.composerTools}>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept="text/*,.json,.md,.yaml,.yml,.csv,.tsv,image/gif,image/jpeg,image/png,image/webp"
              className={styles.fileInput}
              onChange={handleFiles}
              tabIndex={-1}
              aria-hidden="true"
            />
            <AgentComposerMenu
              builderAvailable={builderAvailable}
              builderEnabled={mode === 'builder'}
              disabled={running}
              onAttachFiles={() => fileInputRef.current?.click()}
              onBuilderChange={(enabled) => onModeChange(enabled ? 'builder' : 'chat')}
            />
            <ChatComposerModelSelect
              disabled={targetLocked || running || models.length === 0}
              models={models}
              onChange={onModelChange}
              value={selectedModel}
            />
          </div>
          <div className={styles.composerAction}>
            {disabledReason ? <span>{disabledReason}</span> : null}
            {running && (input.trim() || attachments.length) ? (
              <button
                type="button"
                className={styles.queueButton}
                onClick={onQueue}
                disabled={!canQueue}
              >
                Queue
              </button>
            ) : null}
            <button
              type="button"
              className={running ? styles.stopButton : styles.sendButton}
              onClick={running ? onStop : onSend}
              disabled={running ? false : !canSend}
              aria-label={running ? 'Stop generation' : 'Send message'}
              title={disabledReason || (running ? 'Stop generation' : 'Send message')}
            >
              <ProductIcon name={running ? 'stop' : 'arrow-right'} />
            </button>
          </div>
        </div>
      </div>
      <AgentComposerQueue
        paused={queuePaused}
        turns={queuedTurns}
        onRemove={onQueueRemove}
        onResume={onQueueResume}
      />
      <p className={styles.composerNote}>
        {mode === 'builder'
          ? 'Nothing goes live until you publish.'
          : 'Your access and limits apply.'}
      </p>
    </div>
  )
}
