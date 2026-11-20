import * as React from 'react'

interface ShellContextValue {
  collapsed: boolean
  setCollapsed: (next: boolean) => void
  mobileOpen: boolean
  setMobileOpen: (next: boolean) => void
  commandOpen: boolean
  setCommandOpen: (next: boolean) => void
}

const ShellContext = React.createContext<ShellContextValue | null>(null)
const COLLAPSE_KEY = 'vsr-sidebar-collapsed'

export function ShellProvider({ children }: { children: React.ReactNode }) {
  const [collapsed, setCollapsedState] = React.useState(() => {
    try {
      return localStorage.getItem(COLLAPSE_KEY) === '1'
    } catch {
      return false
    }
  })
  const [mobileOpen, setMobileOpen] = React.useState(false)
  const [commandOpen, setCommandOpen] = React.useState(false)

  const setCollapsed = React.useCallback((next: boolean) => {
    try {
      localStorage.setItem(COLLAPSE_KEY, next ? '1' : '0')
    } catch {
      /* storage unavailable */
    }
    setCollapsedState(next)
  }, [])

  React.useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const mod = event.metaKey || event.ctrlKey
      if (mod && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        setCommandOpen((open) => !open)
      } else if (mod && event.key.toLowerCase() === 'b') {
        event.preventDefault()
        setCollapsedState((current) => {
          const next = !current
          try {
            localStorage.setItem(COLLAPSE_KEY, next ? '1' : '0')
          } catch {
            /* storage unavailable */
          }
          return next
        })
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const value = React.useMemo(
    () => ({ collapsed, setCollapsed, mobileOpen, setMobileOpen, commandOpen, setCommandOpen }),
    [collapsed, setCollapsed, mobileOpen, commandOpen],
  )
  return <ShellContext.Provider value={value}>{children}</ShellContext.Provider>
}

export function useShell(): ShellContextValue {
  const context = React.useContext(ShellContext)
  if (!context) throw new Error('useShell must be used inside ShellProvider')
  return context
}
