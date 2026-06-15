import styles from './ChatComponent.module.css'

interface ChatComponentRoomToggleProps {
  disabled: boolean
  isTeamRoomView: boolean
  onToggle: () => void
}

export default function ChatComponentRoomToggle({
  disabled,
  isTeamRoomView,
  onToggle,
}: ChatComponentRoomToggleProps) {
  return (
    <button
      type="button"
      className={`${styles.teamToggleButton} ${isTeamRoomView ? styles.teamToggleButtonActive : ''}`}
      onClick={onToggle}
      aria-pressed={isTeamRoomView}
      aria-label={isTeamRoomView ? 'Exit ClawRoom view' : 'Open ClawRoom view'}
      title={isTeamRoomView ? 'Switch to chat view' : 'Switch to room chat view'}
      disabled={disabled}
    >
      <img src="/openclaw.svg" alt="" aria-hidden="true" className={styles.clawToggleIcon} />
      <span className={styles.teamToggleLabel}>ClawRoom</span>
    </button>
  )
}
