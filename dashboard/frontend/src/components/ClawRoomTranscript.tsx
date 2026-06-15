import MarkdownRenderer from './MarkdownRenderer'
import ClawRoomMessageMeta from './ClawRoomMessageMeta'
import ClawRoomStreamingToolCards from './ClawRoomStreamingToolCards'
import styles from './ClawRoomChat.module.css'
import {
  parseClawRoomToolTraceFromMessageMetadata,
  type ClawRoomStreamingToolTraceEntry,
} from './clawRoomToolTrace'
import {
  formatMessageTime,
  type RoomMessage,
  type StreamingParticipant,
} from './clawRoomChatSupport'
import { resolveStreamingParticipantDisplay } from './clawRoomStreamingUi'

interface SenderVisual {
  displayName: string
  roleLabel: string
}

interface ClawRoomTranscriptProps {
  selectedRoomId: string
  messages: RoomMessage[]
  streamingMessages: Map<string, string>
  streamingParticipants: Map<string, StreamingParticipant>
  streamingEntries: Array<[string, string]>
  streamingToolTraces: Map<string, ClawRoomStreamingToolTraceEntry>
  resolveSenderVisual: (message: RoomMessage) => SenderVisual
}

const ClawRoomTranscript = ({
  selectedRoomId,
  messages,
  streamingMessages,
  streamingParticipants,
  streamingEntries,
  streamingToolTraces,
  resolveSenderVisual,
}: ClawRoomTranscriptProps) => {
  if (!selectedRoomId) {
    return <div className={styles.stateHint}>Select a room from the left panel.</div>
  }

  if (messages.length === 0 && streamingEntries.length === 0) {
    return <div className={styles.stateHint}>No messages yet. Start the conversation.</div>
  }

  return (
    <>
      {messages.map(message => {
        const isUser = message.senderType === 'user'
        const isSystem = message.senderType === 'system'
        const isLeader = message.senderType === 'leader'
        const isWorker = message.senderType === 'worker'
        const senderVisual = resolveSenderVisual(message)
        const streamingContent = streamingMessages.get(message.id)
        const displayContent = streamingContent && streamingContent !== message.content
          ? streamingContent
          : message.content
        const isStreaming = Boolean(streamingContent && streamingContent !== message.content)
        const liveToolSteps = streamingToolTraces.get(message.id)?.steps || []
        const metadataToolSteps = parseClawRoomToolTraceFromMessageMetadata(message.metadata)
        const toolSteps = liveToolSteps.length > 0 ? liveToolSteps : metadataToolSteps
        return (
          <div
            key={message.id}
            className={`${styles.messageRow} ${isUser ? styles.messageRowUser : styles.messageRowAgent}`}
            data-room-message-id={message.id}
            data-room-message-role={message.senderType}
            data-room-message-streaming={isStreaming ? 'true' : 'false'}
          >
            <div className={styles.messageMain}>
              <ClawRoomMessageMeta
                displayName={senderVisual.displayName}
                isLeader={isLeader}
                isUser={isUser}
                isWorker={isWorker}
                roleLabel={senderVisual.roleLabel}
                timestamp={formatMessageTime(message.createdAt)}
              />
              {!isUser && toolSteps.length > 0 && <ClawRoomStreamingToolCards steps={toolSteps} />}
              <div
                className={`${styles.messageBubble} ${isUser ? styles.messageBubbleUser : styles.messageBubbleAgent} ${isSystem ? styles.messageBubbleSystem : ''} ${isStreaming ? styles.messageBubbleStreaming : ''}`}
                data-room-message-content
              >
                <div className={styles.messageMarkdown}>
                  <MarkdownRenderer content={displayContent} />
                  {isStreaming && <span className={styles.streamingCursor} aria-hidden="true" />}
                </div>
              </div>
            </div>
          </div>
        )
      })}
      {streamingEntries.map(([messageId, content]) => {
        const { participantType, displayName, isLeader, isWorker } = resolveStreamingParticipantDisplay(
          messageId,
          streamingParticipants
        )
        const toolSteps = streamingToolTraces.get(messageId)?.steps || []
        return (
          <div
            key={`streaming-${messageId}`}
            className={`${styles.messageRow} ${styles.messageRowAgent}`}
            data-room-message-id={messageId}
            data-room-message-role={participantType}
            data-room-message-streaming="true"
          >
            <div className={styles.messageMain}>
              <ClawRoomMessageMeta
                displayName={displayName}
                isLeader={isLeader}
                isUser={false}
                isWorker={isWorker}
                roleLabel={isLeader ? 'LEADER' : 'WORKER'}
                timestamp="..."
              />
              <ClawRoomStreamingToolCards steps={toolSteps} />
              {content.trim() ? (
                <div
                  className={`${styles.messageBubble} ${styles.messageBubbleAgent} ${styles.messageBubbleStreaming}`}
                  data-room-message-content
                >
                  <div className={styles.messageMarkdown}>
                    <MarkdownRenderer content={content} />
                    <span className={styles.streamingCursor} aria-hidden="true" />
                  </div>
                </div>
              ) : null}
            </div>
          </div>
        )
      })}
    </>
  )
}

export default ClawRoomTranscript
