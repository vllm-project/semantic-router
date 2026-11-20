import { useCallback, useEffect, useState } from 'react'

const PANES_STORAGE_KEY = 'sr:playground:panes'
// The conversation list docks from the lg breakpoint and the inspector from
// 1360px, where the chat column keeps a comfortable reading width beside both.
const DOCK_CONVERSATIONS = '(min-width: 1024px)'
const DOCK_INSPECTOR = '(min-width: 1360px)'

interface StoredPanes {
  conversations?: boolean
  inspector?: boolean
}

function readStoredPanes(): StoredPanes {
  try {
    const parsed = JSON.parse(window.localStorage.getItem(PANES_STORAGE_KEY) ?? '{}') as unknown
    return parsed && typeof parsed === 'object' ? (parsed as StoredPanes) : {}
  } catch {
    return {}
  }
}

function writeStoredPanes(next: StoredPanes) {
  try {
    window.localStorage.setItem(PANES_STORAGE_KEY, JSON.stringify(next))
  } catch {
    // the layout lasts for this visit only
  }
}

function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() => window.matchMedia(query).matches)
  useEffect(() => {
    const media = window.matchMedia(query)
    const update = () => setMatches(media.matches)
    update()
    media.addEventListener('change', update)
    return () => media.removeEventListener('change', update)
  }, [query])
  return matches
}

export interface PlaygroundPanes {
  conversationsOpen: boolean
  inspectorOpen: boolean
  /** The pane floats over the chat instead of docking beside it. */
  conversationsOverlay: boolean
  inspectorOverlay: boolean
  setConversationsOpen: (open: boolean) => void
  setInspectorOpen: (open: boolean) => void
}

/**
 * Which side panes show: docked panes follow the viewer's last choice, and a
 * narrow viewport starts with both closed and floats a pane it opens.
 */
export function usePlaygroundPanes(): PlaygroundPanes {
  const conversationsDock = useMediaQuery(DOCK_CONVERSATIONS)
  const inspectorDock = useMediaQuery(DOCK_INSPECTOR)
  const [conversationsOpen, setConversations] = useState(
    () => conversationsDock && (readStoredPanes().conversations ?? true),
  )
  const [inspectorOpen, setInspector] = useState(
    () => inspectorDock && (readStoredPanes().inspector ?? true),
  )

  useEffect(() => {
    if (!conversationsDock) setConversations(false)
  }, [conversationsDock])
  useEffect(() => {
    if (!inspectorDock) setInspector(false)
  }, [inspectorDock])

  const setConversationsOpen = useCallback(
    (open: boolean) => {
      setConversations(open)
      if (conversationsDock) writeStoredPanes({ ...readStoredPanes(), conversations: open })
    },
    [conversationsDock],
  )
  const setInspectorOpen = useCallback(
    (open: boolean) => {
      setInspector(open)
      if (inspectorDock) writeStoredPanes({ ...readStoredPanes(), inspector: open })
    },
    [inspectorDock],
  )

  return {
    conversationsOpen,
    inspectorOpen,
    conversationsOverlay: !conversationsDock,
    inspectorOverlay: !inspectorDock,
    setConversationsOpen,
    setInspectorOpen,
  }
}
