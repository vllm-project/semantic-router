import { useCallback, useEffect, useId, useRef, useState, type KeyboardEvent } from 'react'

import ProductIcon from './ProductIcon'
import styles from './AgentComposerMenu.module.css'

interface AgentComposerMenuProps {
  builderAvailable: boolean
  builderEnabled: boolean
  disabled?: boolean
  onAttachFiles: () => void
  onBuilderChange: (enabled: boolean) => void
}

export default function AgentComposerMenu({
  builderAvailable,
  builderEnabled,
  disabled = false,
  onAttachFiles,
  onBuilderChange,
}: AgentComposerMenuProps) {
  const [open, setOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)
  const triggerRef = useRef<HTMLButtonElement>(null)
  const menuRef = useRef<HTMLDivElement>(null)
  const menuId = `agent-composer-menu-${useId().replace(/:/g, '')}`

  const close = useCallback((restoreFocus = false) => {
    setOpen(false)
    if (restoreFocus) requestAnimationFrame(() => triggerRef.current?.focus())
  }, [])

  useEffect(() => {
    if (!open) return
    requestAnimationFrame(() =>
      menuRef.current?.querySelector<HTMLButtonElement>('button:not(:disabled)')?.focus(),
    )
    const onPointerDown = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) close()
    }
    const onKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        close(true)
      }
    }
    document.addEventListener('pointerdown', onPointerDown)
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('pointerdown', onPointerDown)
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [close, open])

  const moveFocus = (event: KeyboardEvent<HTMLDivElement>) => {
    if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return
    const buttons = Array.from(
      menuRef.current?.querySelectorAll<HTMLButtonElement>('button:not(:disabled)') ?? [],
    )
    if (!buttons.length) return
    event.preventDefault()
    const current = buttons.indexOf(document.activeElement as HTMLButtonElement)
    if (event.key === 'Home') return buttons[0]?.focus()
    if (event.key === 'End') return buttons[buttons.length - 1]?.focus()
    const direction = event.key === 'ArrowDown' ? 1 : -1
    const next = (current + direction + buttons.length) % buttons.length
    buttons[next]?.focus()
  }

  return (
    <div ref={rootRef} className={styles.root}>
      <button
        ref={triggerRef}
        type="button"
        className={`${styles.trigger} ${open ? styles.triggerOpen : ''}`}
        aria-controls={menuId}
        aria-expanded={open}
        aria-haspopup="menu"
        aria-label="Add to conversation"
        data-testid="playground-composer-add"
        disabled={disabled}
        onClick={() => setOpen((current) => !current)}
      >
        <ProductIcon name="plus" />
      </button>
      {open ? (
        <div
          ref={menuRef}
          id={menuId}
          className={styles.menu}
          role="menu"
          aria-label="Add to conversation"
          data-testid="playground-composer-add-menu"
          onKeyDown={moveFocus}
        >
          <button
            type="button"
            className={styles.item}
            role="menuitem"
            onClick={() => {
              onAttachFiles()
              close(true)
            }}
          >
            <span className={styles.icon}>
              <ProductIcon name="attachment" />
            </span>
            <span className={styles.copy}>
              <strong>Attach files</strong>
              <small>Images, text, and structured data</small>
            </span>
          </button>
          {builderAvailable ? (
            <>
              <div className={styles.divider} role="separator" />
              {builderAvailable ? (
                <button
                  type="button"
                  className={`${styles.item} ${builderEnabled ? styles.itemActive : ''}`}
                  role="menuitemcheckbox"
                  aria-checked={builderEnabled}
                  data-testid="playground-builder-mode"
                  onClick={() => {
                    onBuilderChange(!builderEnabled)
                    close(true)
                  }}
                >
                  <span className={styles.icon}>
                    <ProductIcon name="mixture" />
                  </span>
                  <span className={styles.copy}>
                    <strong>Builder</strong>
                    <small>Design and test a model path</small>
                  </span>
                  {builderEnabled ? <ProductIcon className={styles.check} name="check" /> : null}
                </button>
              ) : null}
            </>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}
