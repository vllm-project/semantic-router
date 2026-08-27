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
      <div className={styles.playgroundSidebarChrome}>
        <button
          type="button"
          className={styles.playgroundSidebarBrand}
          onClick={onToggleSidebar}
          title={isOpen ? 'Close sidebar' : 'Open sidebar'}
          aria-label={isOpen ? 'Close sidebar' : 'Open sidebar'}
        >
          <img
            src={isOpen ? '/vllm-sr-logo.white.png' : '/vllm.png'}
            alt={isOpen ? 'vLLM Semantic Router' : 'vLLM'}
          />
        </button>
        <button
          type="button"
          className={styles.playgroundSidebarCreate}
          onClick={onCreate}
          disabled={createDisabled}
          title={isTeamRoomView ? 'New room' : 'New conversation'}
          aria-label={isTeamRoomView ? 'New room' : 'New conversation'}
        >
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.9"
          >
            <path
              d="M14 4h-6a3 3 0 0 0-3 3v8a3 3 0 0 0 3 3h1v3l3.6-3H14a3 3 0 0 0 3-3v-2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <path d="M17 3v6M14 6h6" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          {isOpen ? <span>{isTeamRoomView ? 'New room' : 'New chat'}</span> : null}
        </button>
        {children ? (
          <div className={styles.playgroundSidebarPanel} aria-hidden={!isOpen}>
            {isOpen ? <div className={styles.playgroundSidebarPanelInner}>{children}</div> : null}
          </div>
        ) : null}
        <div className={styles.playgroundSidebarFooter}>
          <PlaygroundRailAccountControl />
        </div>
      </div>
    </aside>
  )
}
