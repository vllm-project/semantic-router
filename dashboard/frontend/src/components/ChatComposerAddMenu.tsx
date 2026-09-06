import {
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactNode,
} from 'react'

import styles from './ChatComposerAddMenu.module.css'

interface ClawRoomMenuOption {
  active: boolean
  disabled: boolean
  onToggle: () => void
}

interface ChatComposerAddMenuProps {
  attachFilesDisabled?: boolean
  clawModeDisabled: boolean
  clawModeEnabled: boolean
  clawRoom?: ClawRoomMenuOption
  onAttachFiles?: () => void
  onToggleClawMode: () => void
  onToggleWebSearch?: () => void
  webSearchDisabled?: boolean
  webSearchEnabled: boolean
  webSearchLocked?: boolean
}

interface ComposerMenuItem {
  checked?: boolean
  closeOnSelect?: boolean
  description: string
  disabled?: boolean
  icon: ReactNode
  id: string
  label: string
  onSelect: () => void
  restoreFocusOnSelect?: boolean
}

const AttachmentIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
    <path
      d="M21.44 11.05 12.25 20.24a6 6 0 1 1-8.49-8.49l9.19-9.19a4 4 0 1 1 5.66 5.66l-9.2 9.19a2 2 0 1 1-2.83-2.83l8.49-8.48"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
)

const WebSearchIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
    <circle cx="12" cy="12" r="9" />
    <path d="M3 12h18M12 3c2.4 2.45 3.65 5.45 3.65 9S14.4 18.55 12 21c-2.4-2.45-3.65-5.45-3.65-9S9.6 5.45 12 3Z" />
  </svg>
)

const RoomIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
    <path d="M4 5.5h16v10H8l-4 3v-13Z" strokeLinecap="round" strokeLinejoin="round" />
    <path d="M8 9h8M8 12h5" strokeLinecap="round" />
  </svg>
)

const CheckIcon = () => (
  <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
    <path d="m3 8.2 3.1 3.1L13 4.8" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
)

export default function ChatComposerAddMenu({
  attachFilesDisabled = false,
  clawModeDisabled,
  clawModeEnabled,
  clawRoom,
  onAttachFiles,
  onToggleClawMode,
  onToggleWebSearch,
  webSearchDisabled = false,
  webSearchEnabled,
  webSearchLocked = false,
}: ChatComposerAddMenuProps) {
  const [isOpen, setIsOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)
  const triggerRef = useRef<HTMLButtonElement>(null)
  const menuRef = useRef<HTMLDivElement>(null)
  const generatedId = useId()
  const menuId = `chat-composer-add-menu-${generatedId.replace(/:/g, '')}`

  const closeMenu = useCallback((restoreFocus = false) => {
    setIsOpen(false)
    if (restoreFocus) {
      requestAnimationFrame(() => triggerRef.current?.focus())
    }
  }, [])

  const items = useMemo<ComposerMenuItem[]>(() => {
    const nextItems: ComposerMenuItem[] = []

    if (onAttachFiles) {
      nextItems.push({
        closeOnSelect: true,
        description: 'Add files up to 10 MB each',
        disabled: attachFilesDisabled,
        icon: <AttachmentIcon />,
        id: 'attach-files',
        label: 'Attach files',
        onSelect: onAttachFiles,
        restoreFocusOnSelect: true,
      })
    }

    nextItems.push({
      checked: webSearchEnabled,
      description: webSearchLocked
        ? 'Always on in ClawRoom'
        : 'Use current information from the web',
      disabled: webSearchDisabled || !onToggleWebSearch,
      icon: <WebSearchIcon />,
      id: 'web-search',
      label: 'Web search',
      onSelect: onToggleWebSearch ?? (() => undefined),
    })

    nextItems.push({
      checked: clawModeEnabled,
      description: 'Recruit and coordinate Claw specialists',
      disabled: clawModeDisabled,
      icon: <img src="/openclaw.svg" alt="" aria-hidden="true" />,
      id: 'hire-claw',
      label: 'HireClaw',
      onSelect: onToggleClawMode,
    })

    if (clawRoom) {
      nextItems.push({
        checked: clawRoom.active,
        closeOnSelect: true,
        description: clawRoom.active
          ? 'Return to the standard playground chat'
          : 'Open the collaborative room',
        disabled: clawRoom.disabled,
        icon: <RoomIcon />,
        id: 'claw-room',
        label: clawRoom.active ? 'Exit ClawRoom' : 'Open ClawRoom',
        onSelect: clawRoom.onToggle,
      })
    }

    return nextItems
  }, [
    attachFilesDisabled,
    clawModeDisabled,
    clawModeEnabled,
    clawRoom,
    onAttachFiles,
    onToggleClawMode,
    onToggleWebSearch,
    webSearchDisabled,
    webSearchEnabled,
    webSearchLocked,
  ])

  useEffect(() => {
    if (!isOpen) {
      return undefined
    }

    requestAnimationFrame(() => {
      menuRef.current?.querySelector<HTMLButtonElement>('button:not(:disabled)')?.focus()
    })

    const handlePointerDown = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) {
        closeMenu()
      }
    }

    const handleKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        closeMenu(true)
      }
    }

    document.addEventListener('pointerdown', handlePointerDown)
    document.addEventListener('keydown', handleKeyDown)
    return () => {
      document.removeEventListener('pointerdown', handlePointerDown)
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [closeMenu, isOpen])

  const handleMenuKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) {
      return
    }

    const menuItems = Array.from(
      menuRef.current?.querySelectorAll<HTMLButtonElement>('button:not(:disabled)') ?? [],
    )
    if (menuItems.length === 0) {
      return
    }

    event.preventDefault()
    const currentIndex = menuItems.indexOf(document.activeElement as HTMLButtonElement)
    if (event.key === 'Home') {
      menuItems[0]?.focus()
      return
    }
    if (event.key === 'End') {
      menuItems[menuItems.length - 1]?.focus()
      return
    }

    const direction = event.key === 'ArrowDown' ? 1 : -1
    const fallbackIndex = direction === 1 ? -1 : 0
    const nextIndex = (currentIndex === -1 ? fallbackIndex : currentIndex) + direction
    menuItems[(nextIndex + menuItems.length) % menuItems.length]?.focus()
  }

  return (
    <div
      ref={rootRef}
      className={styles.root}
      onBlur={(event) => {
        const nextTarget = event.relatedTarget
        if (!(nextTarget instanceof Node) || !event.currentTarget.contains(nextTarget)) {
          closeMenu()
        }
      }}
    >
      <button
        ref={triggerRef}
        type="button"
        className={`${styles.trigger} ${isOpen ? styles.triggerOpen : ''}`}
        aria-controls={menuId}
        aria-expanded={isOpen}
        aria-haspopup="menu"
        aria-label="Add to prompt"
        data-testid="playground-composer-add"
        onClick={() => setIsOpen((current) => !current)}
        title="Add to prompt"
      >
        <svg
          viewBox="0 0 20 20"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.7"
          aria-hidden="true"
        >
          <path d="M10 3v14M3 10h14" strokeLinecap="round" />
        </svg>
      </button>

      {isOpen ? (
        <div
          ref={menuRef}
          id={menuId}
          className={styles.menu}
          role="menu"
          aria-label="Add to prompt"
          data-testid="playground-composer-add-menu"
          onKeyDown={handleMenuKeyDown}
        >
          {items.map((item) => {
            const isToggle = typeof item.checked === 'boolean'
            const ariaLabel =
              item.id === 'hire-claw'
                ? `${item.checked ? 'Disable' : 'Enable'} HireClaw`
                : item.id === 'web-search'
                  ? `${item.checked ? 'Disable' : 'Enable'} Web Search`
                  : item.id === 'claw-room'
                    ? `${clawRoom?.active ? 'Exit' : 'Open'} ClawRoom view`
                    : item.label
            return (
              <button
                key={item.id}
                type="button"
                className={`${styles.item} ${item.checked ? styles.itemActive : ''}`}
                disabled={item.disabled}
                role={isToggle ? 'menuitemcheckbox' : 'menuitem'}
                aria-checked={isToggle ? item.checked : undefined}
                aria-label={ariaLabel}
                data-testid={
                  item.id === 'attach-files'
                    ? 'playground-attach-files'
                    : `playground-composer-add-${item.id}`
                }
                onClick={() => {
                  item.onSelect()
                  if (item.closeOnSelect) {
                    closeMenu(item.restoreFocusOnSelect)
                  }
                }}
              >
                <span className={styles.itemIcon}>{item.icon}</span>
                <span className={styles.itemCopy}>
                  <span className={styles.itemLabel}>{item.label}</span>
                  <span className={styles.itemDescription}>{item.description}</span>
                </span>
                {item.checked ? (
                  <span className={styles.itemCheck}>
                    <CheckIcon />
                  </span>
                ) : null}
              </button>
            )
          })}
        </div>
      ) : null}
    </div>
  )
}
