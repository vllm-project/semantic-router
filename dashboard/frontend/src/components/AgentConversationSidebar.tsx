import { useEffect, useRef, useState } from 'react'

import type { AgentSession } from '../generated/managementApiContract'
import ProductIcon from './ProductIcon'
import styles from './AgentPlayground.module.css'

interface AgentConversationSidebarProps {
  activeSessionId: string | null
  busy: boolean
  loading: boolean
  open: boolean
  search: string
  sessions: AgentSession[]
  sessionsHaveMore: boolean
  onDeleteRequest: (session: AgentSession) => void
  onLoadMore: () => void
  onNewChat: () => void
  onSearchChange: (value: string) => void
  onSelect: (session: AgentSession) => void
  onToggle: () => void
}

function relativeTime(value: string): string {
  const timestamp = Date.parse(value)
  if (!Number.isFinite(timestamp)) return ''
  const seconds = Math.round((timestamp - Date.now()) / 1000)
  const formatter = new Intl.RelativeTimeFormat('en-US', { numeric: 'auto' })
  if (Math.abs(seconds) < 60) return formatter.format(seconds, 'second')
  const minutes = Math.round(seconds / 60)
  if (Math.abs(minutes) < 60) return formatter.format(minutes, 'minute')
  const hours = Math.round(minutes / 60)
  if (Math.abs(hours) < 24) return formatter.format(hours, 'hour')
  return formatter.format(Math.round(hours / 24), 'day')
}

export default function AgentConversationSidebar({
  activeSessionId,
  busy,
  loading,
  open,
  search,
  sessions,
  sessionsHaveMore,
  onDeleteRequest,
  onLoadMore,
  onNewChat,
  onSearchChange,
  onSelect,
  onToggle,
}: AgentConversationSidebarProps) {
  const [menuSessionId, setMenuSessionId] = useState<string | null>(null)
  const rootRef = useRef<HTMLElement>(null)
  const menuTriggerRefs = useRef(new Map<string, HTMLButtonElement>())

  useEffect(() => {
    if (!menuSessionId) return
    const closeOnOutsidePress = (event: PointerEvent) => {
      const target = event.target
      if (!(target instanceof Element) || !target.closest('[data-agent-session-menu]')) {
        setMenuSessionId(null)
      }
    }
    requestAnimationFrame(() => {
      rootRef.current
        ?.querySelector<HTMLButtonElement>('[data-agent-session-menu] [role="menuitem"]')
        ?.focus()
    })
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        const trigger = menuTriggerRefs.current.get(menuSessionId)
        setMenuSessionId(null)
        requestAnimationFrame(() => trigger?.focus())
      }
    }
    document.addEventListener('pointerdown', closeOnOutsidePress)
    document.addEventListener('keydown', closeOnEscape)
    return () => {
      document.removeEventListener('pointerdown', closeOnOutsidePress)
      document.removeEventListener('keydown', closeOnEscape)
    }
  }, [menuSessionId])

  useEffect(() => {
    if (!open) return
    const closeOnEscape = (event: KeyboardEvent) => {
      if (
        event.key === 'Escape' &&
        !menuSessionId &&
        window.matchMedia('(max-width: 959px)').matches
      ) {
        event.preventDefault()
        onToggle()
      }
    }
    document.addEventListener('keydown', closeOnEscape)
    return () => document.removeEventListener('keydown', closeOnEscape)
  }, [menuSessionId, onToggle, open])

  return (
    <aside
      ref={rootRef}
      className={`${styles.sidebarShell} ${open ? styles.sidebarShellOpen : ''}`}
      aria-label="Conversations"
      data-testid="agent-conversation-sidebar"
    >
      <div className={styles.sidebarRail}>
        <button
          type="button"
          className={`${styles.railButton} ${open ? styles.railButtonActive : ''}`}
          onClick={onToggle}
          aria-label={open ? 'Close conversations' : 'Open conversations'}
          aria-expanded={open}
        >
          <ProductIcon name={open ? 'chevron-left' : 'chevron-right'} />
        </button>
        <button
          type="button"
          className={styles.railButton}
          onClick={onNewChat}
          aria-label="New chat"
          disabled={busy}
        >
          <ProductIcon name="plus" />
        </button>
      </div>
      {open ? (
        <div className={styles.sidebarPanel}>
          <div className={styles.sidebarHeader}>
            <div>
              <strong>Conversations</strong>
              <span>Recent</span>
            </div>
            <button type="button" className={styles.sidebarNew} onClick={onNewChat} disabled={busy}>
              <ProductIcon name="plus" />
              New
            </button>
          </div>
          <label className={styles.sidebarSearch}>
            <ProductIcon name="search" aria-hidden="true" />
            <span className={styles.srOnly}>Search conversations</span>
            <input
              type="search"
              value={search}
              onChange={(event) => onSearchChange(event.target.value)}
              placeholder="Search"
            />
          </label>
          <div className={styles.sessionList} aria-busy={loading}>
            {loading && sessions.length === 0 ? (
              <div className={styles.sessionEmpty} role="status">
                Loading conversations…
              </div>
            ) : null}
            {!loading && sessions.length === 0 ? (
              <div className={styles.sessionEmpty}>Your conversations will appear here.</div>
            ) : null}
            {sessions.map((session) => (
              <div
                key={session.id}
                className={`${styles.sessionRow} ${session.id === activeSessionId ? styles.sessionRowActive : ''}`}
              >
                <button
                  type="button"
                  className={styles.sessionSelect}
                  onClick={() => onSelect(session)}
                  disabled={busy}
                  aria-current={session.id === activeSessionId ? 'page' : undefined}
                >
                  <span className={styles.sessionTitle}>{session.title}</span>
                  <span className={styles.sessionMeta}>
                    <span>{session.mode === 'builder' ? 'Builder' : 'Chat'}</span>
                    <span aria-hidden="true">·</span>
                    <time dateTime={session.updatedAt}>{relativeTime(session.updatedAt)}</time>
                  </span>
                </button>
                <button
                  ref={(element) => {
                    if (element) menuTriggerRefs.current.set(session.id, element)
                    else menuTriggerRefs.current.delete(session.id)
                  }}
                  type="button"
                  className={styles.sessionMore}
                  data-agent-session-menu=""
                  onClick={() =>
                    setMenuSessionId((current) => (current === session.id ? null : session.id))
                  }
                  aria-label={`More options for ${session.title}`}
                  aria-expanded={menuSessionId === session.id}
                  disabled={busy}
                >
                  <ProductIcon name="more" aria-hidden="true" />
                </button>
                {menuSessionId === session.id ? (
                  <div className={styles.sessionMenu} role="menu" data-agent-session-menu="">
                    <button
                      type="button"
                      role="menuitem"
                      disabled={busy}
                      onClick={() => {
                        setMenuSessionId(null)
                        onDeleteRequest(session)
                      }}
                    >
                      <ProductIcon name="trash" />
                      Delete
                    </button>
                  </div>
                ) : null}
              </div>
            ))}
            {sessionsHaveMore ? (
              <button
                type="button"
                className={styles.loadMore}
                onClick={onLoadMore}
                disabled={loading || busy}
              >
                {loading ? 'Loading…' : 'Load more'}
              </button>
            ) : null}
          </div>
        </div>
      ) : null}
    </aside>
  )
}
