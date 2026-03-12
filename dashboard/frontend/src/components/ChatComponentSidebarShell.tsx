import type { ReactNode } from 'react'

import styles from './ChatComponent.module.css'
import PlaygroundRailAccountControl from './PlaygroundRailAccountControl'

interface ChatComponentSidebarShellProps {
  children?: ReactNode
  createDisabled?: boolean
  isOpen: boolean
  isTeamRoomView: boolean
  onCreate: () => void
  onToggleSidebar: () => void
}

export default function ChatComponentSidebarShell({
  children,
  createDisabled = false,
  isOpen,
  isTeamRoomView,
  onCreate,
  onToggleSidebar,
}: ChatComponentSidebarShellProps) {
  return (
    <aside
      className={`${styles.playgroundSidebarShell} ${isOpen ? styles.playgroundSidebarShellOpen : ''}`}
      data-testid="playground-sidebar-shell"
    >
      <div className={styles.playgroundSidebarRail}>
        <button
          type="button"
          className={`${styles.playgroundSidebarRailButton} ${isOpen ? styles.playgroundSidebarRailButtonActive : ''}`}
          onClick={onToggleSidebar}
          title={isOpen ? 'Close sidebar' : 'Open sidebar'}
          aria-label={isOpen ? 'Close sidebar' : 'Open sidebar'}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9">
            <rect x="3" y="4" width="18" height="16" rx="3" />
            <path d="M9 4v16" />
            {isOpen ? (
              <path d="M15 9l-2.5 3 2.5 3" strokeLinecap="round" strokeLinejoin="round" />
            ) : (
              <path d="M12.5 9l2.5 3-2.5 3" strokeLinecap="round" strokeLinejoin="round" />
            )}
          </svg>
        </button>
        <button
          type="button"
          className={styles.playgroundSidebarRailButton}
          onClick={onCreate}
          disabled={createDisabled}
          title={isTeamRoomView ? 'New room' : 'New conversation'}
          aria-label={isTeamRoomView ? 'New room' : 'New conversation'}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9">
            <path
              d="M14 4h-6a3 3 0 0 0-3 3v8a3 3 0 0 0 3 3h1v3l3.6-3H14a3 3 0 0 0 3-3v-2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <path d="M17 3v6M14 6h6" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
        <div className={styles.playgroundSidebarRailFooter}>
          <PlaygroundRailAccountControl />
        </div>
      </div>
      {children ? (
        <div
          className={`${styles.playgroundSidebarPanel} ${isOpen ? styles.playgroundSidebarPanelVisible : ''}`}
          aria-hidden={!isOpen}
        >
          <div className={styles.playgroundSidebarPanelInner}>
            {children}
          </div>
        </div>
      ) : null}
    </aside>
  )
}
