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
import { Check, Globe, Paperclip, Plus } from 'lucide-react'

import { cn } from '../ui'

interface ChatComposerAddMenuProps {
  attachFilesDisabled?: boolean
  onAttachFiles?: () => void
  onToggleWebSearch?: () => void
  webSearchDisabled?: boolean
  webSearchEnabled: boolean
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

export default function ChatComposerAddMenu({
  attachFilesDisabled = false,
  onAttachFiles,
  onToggleWebSearch,
  webSearchDisabled = false,
  webSearchEnabled,
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
        icon: <Paperclip />,
        id: 'attach-files',
        label: 'Attach files',
        onSelect: onAttachFiles,
        restoreFocusOnSelect: true,
      })
    }

    nextItems.push({
      checked: webSearchEnabled,
      description: 'Use current information from the web',
      disabled: webSearchDisabled || !onToggleWebSearch,
      icon: <Globe />,
      id: 'web-search',
      label: 'Web search',
      onSelect: onToggleWebSearch ?? (() => undefined),
    })

    return nextItems
  }, [attachFilesDisabled, onAttachFiles, onToggleWebSearch, webSearchDisabled, webSearchEnabled])

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
      className="relative"
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
        className={cn(
          'grid size-8 place-items-center rounded-full text-muted-foreground outline-none transition-[color,background-color,transform] duration-150 hover:bg-accent hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/40 active:scale-95',
          isOpen && 'bg-accent text-foreground',
        )}
        aria-controls={menuId}
        aria-expanded={isOpen}
        aria-haspopup="menu"
        aria-label="Add to prompt"
        data-testid="playground-composer-add"
        onClick={() => setIsOpen((current) => !current)}
        title="Add to prompt"
      >
        <Plus className={cn('size-4 transition-transform duration-200', isOpen && 'rotate-45')} />
      </button>

      {isOpen ? (
        <div
          ref={menuRef}
          id={menuId}
          className="absolute bottom-[calc(100%+8px)] left-0 z-40 flex w-[min(17rem,calc(100vw-2.5rem))] flex-col gap-0.5 rounded-xl border border-border bg-popover p-1.5 text-popover-foreground shadow-lg animate-in fade-in-0 zoom-in-95 slide-in-from-bottom-1"
          role="menu"
          aria-label="Add to prompt"
          data-testid="playground-composer-add-menu"
          onKeyDown={handleMenuKeyDown}
        >
          {items.map((item) => {
            const isToggle = typeof item.checked === 'boolean'
            const ariaLabel =
              item.id === 'web-search'
                ? `${item.checked ? 'Disable' : 'Enable'} Web Search`
                : item.label
            return (
              <button
                key={item.id}
                type="button"
                className="flex items-start gap-2.5 rounded-lg px-2.5 py-2 text-left outline-none transition-colors hover:bg-accent focus-visible:bg-accent disabled:pointer-events-none disabled:opacity-45"
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
                <span
                  className={cn(
                    'mt-0.5 grid size-7 shrink-0 place-items-center rounded-md border border-border bg-surface [&_svg]:size-3.5',
                    item.checked ? 'text-primary' : 'text-muted-foreground',
                  )}
                >
                  {item.icon}
                </span>
                <span className="flex min-w-0 flex-1 flex-col">
                  <span className="text-[13px] font-medium">{item.label}</span>
                  <span className="text-xs text-subtle-foreground">{item.description}</span>
                </span>
                {item.checked ? (
                  <Check className="mt-1 size-3.5 shrink-0 text-primary" strokeWidth={2.6} />
                ) : null}
              </button>
            )
          })}
        </div>
      ) : null}
    </div>
  )
}
