import styles from './ChatComponent.module.css'
import type { ConversationPreview } from './ChatComponentTypes'

interface ChatConversationSidebarProps {
  conversationId: string
  conversationPreviews: ConversationPreview[]
  onDeleteConversation: (id: string) => void
  onSelectConversation: (id: string) => void
}

export default function ChatConversationSidebar({
  conversationId,
  conversationPreviews,
  onDeleteConversation,
  onSelectConversation,
}: ChatConversationSidebarProps) {
  return (
    <div className={styles.sidebar}>
      <div className={styles.sidebarHeader}>
        <div>
          <div className={styles.sidebarEyebrow}>History</div>
          <div className={styles.sidebarTitle}>Chats</div>
          <div className={styles.sidebarSubtitle}>
            {conversationPreviews.length ? `${conversationPreviews.length} saved` : 'No saved conversations'}
          </div>
        </div>
      </div>
      <div className={styles.sidebarList}>
        {conversationPreviews.length === 0 ? (
          <div className={styles.sidebarEmpty}>Start a conversation to see it here.</div>
        ) : (
          conversationPreviews.map(conv => (
            <div
              key={conv.id}
              className={`${styles.sidebarItem} ${conv.id === conversationId ? styles.sidebarItemActive : ''}`}
              onClick={() => onSelectConversation(conv.id)}
              role="button"
              tabIndex={0}
              onKeyDown={event => {
                if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault()
                  onSelectConversation(conv.id)
                }
              }}
            >
              <div className={styles.sidebarItemText}>
                <div className={styles.sidebarItemTitle}>{conv.preview}</div>
                <div className={styles.sidebarItemMeta}>
                  {new Date(conv.updatedAt).toLocaleString(undefined, {
                    hour: '2-digit',
                    minute: '2-digit',
                    month: 'short',
                    day: 'numeric',
                  })}
                </div>
              </div>
              <button
                type="button"
                className={styles.sidebarDeleteButton}
                onClick={event => {
                  event.stopPropagation()
                  onDeleteConversation(conv.id)
                }}
                title="Delete conversation"
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
          ))
        )}
      </div>
    </div>
  )
}
