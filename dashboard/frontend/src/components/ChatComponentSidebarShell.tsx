import { useEffect, type ReactNode } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { SquarePen } from 'lucide-react'

import { Button, cn } from '../ui'

interface ChatComponentSidebarShellProps {
  children?: ReactNode
  isOpen: boolean
  /** Float over the chat, as on a narrow viewport, instead of docking beside it. */
  overlay: boolean
  onCreate: () => void
  onClose: () => void
}

const EASE = [0.16, 1, 0.3, 1] as const

/** The playground's conversation pane: a new-conversation action above the saved conversations. */
export default function ChatComponentSidebarShell({
  children,
  isOpen,
  overlay,
  onCreate,
  onClose,
}: ChatComponentSidebarShellProps) {
  useEffect(() => {
    if (!isOpen || !overlay) return undefined
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', closeOnEscape)
    return () => document.removeEventListener('keydown', closeOnEscape)
  }, [isOpen, onClose, overlay])

  return (
    <AnimatePresence initial={false}>
      {isOpen && overlay ? (
        <motion.div
          key="scrim"
          aria-hidden
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="absolute inset-0 z-20 bg-background/60 backdrop-blur-[1px]"
          onClick={onClose}
        />
      ) : null}
      {isOpen ? (
        <motion.aside
          key="pane"
          aria-label="Conversations"
          data-testid="playground-sidebar-shell"
          initial={overlay ? { x: -16, opacity: 0 } : { width: 0, opacity: 0 }}
          animate={overlay ? { x: 0, opacity: 1 } : { width: 256, opacity: 1 }}
          exit={overlay ? { x: -16, opacity: 0 } : { width: 0, opacity: 0 }}
          transition={{ duration: 0.28, ease: EASE }}
          className={cn(
            'flex shrink-0 flex-col overflow-hidden border-r border-border',
            overlay
              ? 'absolute inset-y-0 left-0 z-30 w-64 bg-surface shadow-xl'
              : 'bg-surface-subtle/50',
          )}
        >
          <div className="flex w-64 shrink-0 flex-col gap-2 p-3">
            <Button variant="outline" className="justify-start" onClick={onCreate}>
              <SquarePen />
              New conversation
            </Button>
          </div>
          <div className="flex min-h-0 w-64 flex-1 flex-col">{children}</div>
        </motion.aside>
      ) : null}
    </AnimatePresence>
  )
}
