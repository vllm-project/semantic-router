import styles from './ClawRoomChat.module.css'
import type { MentionAutocompleteState, MentionOption } from './clawRoomChatSupport'

interface ClawRoomMentionMenuProps {
  mentionAutocomplete: MentionAutocompleteState
  onSelect: (option: MentionOption) => void
  onActiveIndexChange: (index: number) => void
}

const ClawRoomMentionMenu = ({
  mentionAutocomplete,
  onSelect,
  onActiveIndexChange,
}: ClawRoomMentionMenuProps) => {
  if (mentionAutocomplete.options.length === 0) {
    return null
  }

  return (
    <div className={styles.mentionMenu} role="listbox" aria-label="Mention suggestions">
      {mentionAutocomplete.options.map((option, index) => {
        const isActive = mentionAutocomplete.activeIndex === index
        return (
          <button
            key={option.token}
            type="button"
            className={`${styles.mentionItem} ${isActive ? styles.mentionItemActive : ''}`}
            onMouseDown={event => {
              event.preventDefault()
              onSelect(option)
            }}
            onMouseEnter={() => onActiveIndexChange(index)}
          >
            <span className={styles.mentionToken}>{option.token}</span>
            <span className={styles.mentionDescription}>{option.description}</span>
          </button>
        )
      })}
    </div>
  )
}

export default ClawRoomMentionMenu
