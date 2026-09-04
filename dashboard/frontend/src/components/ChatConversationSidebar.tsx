import { useEffect, useRef, useState } from 'react'

import ProductIcon from './ProductIcon'
import styles from './ChatComponent.module.css'
import type { ConversationPreview } from './ChatComponentTypes'

interface ChatConversationSidebarProps {
  conversationId: string
  conversationPreviews: ConversationPreview[]
  onDeleteConversation: (id: string) => void
  onRenameConversation: (id: string, title: string) => void
  onSelectConversation: (id: string) => void
}

export default function ChatConversationSidebar({
  conversationId,
  conversationPreviews,
  onDeleteConversation,
  onRenameConversation,
  onSelectConversation,
}: ChatConversationSidebarProps) {
  const [openMenuId, setOpenMenuId] = useState<string | null>(null)
  const [renaming, setRenaming] = useState<{ id: string; title: string } | null>(null)
  const [isScrolling, setIsScrolling] = useState(false)
  const scrollTimer = useRef<number | null>(null)

  useEffect(() => {
    const closeMenu = (event: PointerEvent) => {
      if (!(event.target as Element | null)?.closest('[data-conversation-menu]')) {
        setOpenMenuId(null)
      }
    }
    const closeWithEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setOpenMenuId(null)
        setRenaming(null)
      }
    }
    document.addEventListener('pointerdown', closeMenu)
    document.addEventListener('keydown', closeWithEscape)
    return () => {
      document.removeEventListener('pointerdown', closeMenu)
      document.removeEventListener('keydown', closeWithEscape)
      if (scrollTimer.current !== null) window.clearTimeout(scrollTimer.current)
    }
  }, [])

  const finishRename = () => {
    if (!renaming) return
    const title = renaming.title.trim()
    if (title) onRenameConversation(renaming.id, title)
    setRenaming(null)
  }

  return (
    <div className={styles.sidebar}>
      <div className={styles.sidebarHeader}>
        <div className={styles.sidebarTitle}>Chats</div>
      </div>
      <div
        className={`${styles.sidebarList} ${isScrolling ? styles.sidebarListScrolling : ''}`}
        onScroll={() => {
          setIsScrolling(true)
          if (scrollTimer.current !== null) window.clearTimeout(scrollTimer.current)
          scrollTimer.current = window.setTimeout(() => setIsScrolling(false), 650)
        }}
      >
        {conversationPreviews.length === 0 ? (
          <div className={styles.sidebarEmpty}>Start a conversation to see it here.</div>
        ) : (
          conversationPreviews.map((conversation) => {
            const active = conversation.id === conversationId
            const editing = renaming?.id === conversation.id
            return (
              <div
                key={conversation.id}
                className={`${styles.sidebarItem} ${active ? styles.sidebarItemActive : ''}`}
              >
                {editing ? (
                  <div className={styles.sidebarRenameRow}>
                    <input
                      value={renaming.title}
                      onChange={(event) =>
                        setRenaming({ id: conversation.id, title: event.target.value })
                      }
                      onBlur={finishRename}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter') finishRename()
                        if (event.key === 'Escape') setRenaming(null)
                      }}
                      maxLength={80}
                      aria-label="Conversation name"
                      autoFocus
                    />
                  </div>
                ) : (
                  <button
                    type="button"
                    className={styles.sidebarItemSelect}
                    onClick={() => onSelectConversation(conversation.id)}
                  >
                    <span className={styles.sidebarItemTitle}>{conversation.preview}</span>
                  </button>
                )}
                <div className={styles.sidebarMenu} data-conversation-menu>
                  <button
                    type="button"
                    className={styles.sidebarMenuTrigger}
                    onClick={() =>
                      setOpenMenuId((current) =>
                        current === conversation.id ? null : conversation.id,
                      )
                    }
                    aria-label={`Conversation options for ${conversation.preview}`}
                    aria-expanded={openMenuId === conversation.id}
                  >
                    <ProductIcon name="more" />
                  </button>
                  {openMenuId === conversation.id ? (
                    <div className={styles.sidebarMenuPopover} role="menu">
                      <button
                        type="button"
                        role="menuitem"
                        onClick={() => {
                          setRenaming({ id: conversation.id, title: conversation.preview })
                          setOpenMenuId(null)
                        }}
                      >
                        <ProductIcon name="edit" />
                        Rename
                      </button>
                      <button
                        type="button"
                        role="menuitem"
                        className={styles.sidebarMenuDanger}
                        onClick={() => {
                          setOpenMenuId(null)
                          onDeleteConversation(conversation.id)
                        }}
                      >
                        <ProductIcon name="trash" />
                        Delete
                      </button>
                    </div>
                  ) : null}
                </div>
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
