import { useEffect, useId, useRef, useState } from 'react'

import useAccessibleDialog from '../hooks/useAccessibleDialog'
import ProductIcon from './ProductIcon'
import type { PlaygroundMode } from './playgroundModes'
import styles from './AgentPlayground.module.css'

export interface PlaygroundConversationListItem {
  id: string
  mode: PlaygroundMode
  source: 'router'
  title: string
  updatedAt: string
}

interface AgentConversationSidebarProps {
  activeSessionId: string | null
  busy: boolean
  loading: boolean
  open: boolean
  search: string
  sessions: PlaygroundConversationListItem[]
  sessionsHaveMore: boolean
  onDeleteRequest: (session: PlaygroundConversationListItem) => void
  onLoadMore: () => void
  onNewChat: () => void
  onSearchChange: (value: string) => void
  onSelect: (session: PlaygroundConversationListItem) => void
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
  const [mobile, setMobile] = useState(() =>
    typeof window === 'undefined' ? false : window.matchMedia('(max-width: 959px)').matches,
  )
  const titleId = useId()
  const menuTriggerRefs = useRef(new Map<string, HTMLButtonElement>())
  const closeSidebar = () => {
    setMenuSessionId(null)
    onToggle()
  }
  const sidebarRef = useAccessibleDialog<HTMLElement>({
    isOpen: mobile && open,
    onClose: closeSidebar,
    dismissible: !menuSessionId,
  })

  useEffect(() => {
    const query = window.matchMedia('(max-width: 959px)')
    const handleChange = (event: MediaQueryListEvent) => setMobile(event.matches)
    setMobile(query.matches)
    query.addEventListener('change', handleChange)
    return () => query.removeEventListener('change', handleChange)
  }, [])

  useEffect(() => {
    if (!menuSessionId) return
    const closeOnOutsidePress = (event: PointerEvent) => {
      const target = event.target
      if (!(target instanceof Element) || !target.closest('[data-agent-session-menu]')) {
        setMenuSessionId(null)
      }
    }
    requestAnimationFrame(() => {
      sidebarRef.current
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
  }, [menuSessionId, sidebarRef])

  if (mobile && !open) return null

  return (
    <>
      {mobile ? (
        <div
          className={styles.sidebarBackdrop}
          role="presentation"
          data-testid="agent-conversation-backdrop"
          onMouseDown={closeSidebar}
        />
      ) : null}
      <aside
        id="agent-conversation-navigation"
        ref={sidebarRef}
        className={`${styles.sidebarShell} ${open ? styles.sidebarShellOpen : ''} ${mobile ? styles.sidebarDrawer : ''}`}
        aria-label={mobile ? undefined : 'Conversations'}
        aria-labelledby={mobile ? titleId : undefined}
        aria-modal={mobile ? 'true' : undefined}
        role={mobile ? 'dialog' : undefined}
        tabIndex={mobile ? -1 : undefined}
        data-testid="agent-conversation-sidebar"
      >
        <div className={styles.sidebarRail}>
          <button
            type="button"
            className={`${styles.railButton} ${open ? styles.railButtonActive : ''}`}
            onClick={closeSidebar}
            aria-label={open ? 'Close conversations' : 'Open conversations'}
            aria-controls="agent-conversation-navigation"
            aria-expanded={open}
            data-dialog-initial-focus={mobile ? true : undefined}
          >
            <ProductIcon name={open ? 'chevron-left' : 'chevron-right'} />
          </button>
          <button
            type="button"
            className={styles.railButton}
            onClick={() => {
              setMenuSessionId(null)
              onNewChat()
            }}
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
                <strong id={titleId}>Conversations</strong>
                <span>Recent</span>
              </div>
              <button
                type="button"
                className={styles.sidebarNew}
                onClick={() => {
                  setMenuSessionId(null)
                  onNewChat()
                }}
                disabled={busy}
              >
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
                    onClick={() => {
                      setMenuSessionId(null)
                      onSelect(session)
                    }}
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
    </>
  )
}
