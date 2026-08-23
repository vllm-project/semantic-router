import styles from './ClawRoomChat.module.css'
import type { RoomTransportMode } from './clawRoomChatSupport'
import {
  resolveTransportStatusClassName,
  resolveTransportStatusLabel,
  resolveTransportStatusTitle,
} from './clawRoomStreamingUi'

interface ClawRoomTransportStatusProps {
  transportMode: RoomTransportMode
  wsConnected: boolean
}

const ClawRoomTransportStatus = ({ transportMode, wsConnected }: ClawRoomTransportStatusProps) => {
  const label = resolveTransportStatusLabel(transportMode, wsConnected)
  const className = resolveTransportStatusClassName(transportMode, wsConnected, {
    wsConnected: styles.wsConnected,
    wsFallback: styles.wsFallback,
    wsDisconnected: styles.wsDisconnected,
  })
  const title = resolveTransportStatusTitle(transportMode, wsConnected)

  return (
    <span
      className={`${styles.chatTitleStatus} ${className}`}
      title={title}
      data-testid="claw-room-transport-status"
    >
      <span className={styles.transportIndicator} aria-hidden="true" />
      {label}
    </span>
  )
}

export default ClawRoomTransportStatus
