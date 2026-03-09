import styles from './ChatComponent.module.css'

interface ChatComponentTopBarProps {
  isSidebarOpen: boolean
  isTeamRoomView: boolean
  createDisabled?: boolean
  onCreate: () => void
  onToggleSidebar: () => void
}

export default function ChatComponentTopBar({
  isSidebarOpen,
  isTeamRoomView,
  createDisabled = false,
  onCreate,
  onToggleSidebar,
}: ChatComponentTopBarProps) {
  return (
    <div className={styles.chatTopBar}>
      <div className={styles.chatTopBarActions}>
        <button
          type="button"
          className={styles.chatTopBarButton}
          onClick={onToggleSidebar}
          title={isSidebarOpen ? 'Close sidebar' : 'Open sidebar'}
          aria-label={isSidebarOpen ? 'Close sidebar' : 'Open sidebar'}
        >
          <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
            <rect x="3" y="4" width="18" height="16" rx="3" />
            <path d="M9 4v16" />
            {isSidebarOpen ? (
              <path d="M15 9l-2.5 3 2.5 3" strokeLinecap="round" strokeLinejoin="round" />
            ) : (
              <path d="M12.5 9l2.5 3-2.5 3" strokeLinecap="round" strokeLinejoin="round" />
            )}
          </svg>
        </button>
        <button
          type="button"
          className={styles.chatTopBarButton}
          onClick={onCreate}
          disabled={createDisabled}
          title={isTeamRoomView ? 'New room' : 'New conversation'}
          aria-label={isTeamRoomView ? 'New room' : 'New conversation'}
        >
          <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
            <path
              d="M14 4h-6a3 3 0 0 0-3 3v8a3 3 0 0 0 3 3h1v3l3.6-3H14a3 3 0 0 0 3-3v-2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <path d="M17 3v6M14 6h6" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
      </div>
    </div>
  )
}
