import styles from './ClawRoomChat.module.css'

interface ClawRoomMessageMetaProps {
  displayName: string
  isLeader: boolean
  isUser: boolean
  isWorker: boolean
  roleLabel: string
  timestamp: string
}

const OPENCLAW_LOGO_SRC = '/openclaw.svg'

export default function ClawRoomMessageMeta({
  displayName,
  isLeader,
  isUser,
  isWorker,
  roleLabel,
  timestamp,
}: ClawRoomMessageMetaProps) {
  return (
    <div className={styles.messageMeta}>
      <span className={styles.senderIdentity}>
        {!isUser ? (
          <img
            src={OPENCLAW_LOGO_SRC}
            alt=""
            aria-hidden="true"
            className={styles.senderLogo}
            data-room-sender-logo
          />
        ) : null}
        <span className={`${styles.senderName} ${isLeader ? styles.senderNameLeader : ''}`}>{displayName}</span>
      </span>
      <span
        className={`${styles.senderType} ${isLeader ? styles.senderTypeLeader : ''} ${isWorker ? styles.senderTypeWorker : ''}`}
      >
        {roleLabel}
      </span>
      <span className={styles.timestamp}>{timestamp}</span>
    </div>
  )
}
