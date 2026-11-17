import * as React from 'react'

export type ThemePreference = 'light' | 'dark' | 'system'
export type ResolvedTheme = 'light' | 'dark'
export type AccentPreset = 'blue' | 'gold'

interface ThemeContextValue {
  preference: ThemePreference
  resolved: ResolvedTheme
  accent: AccentPreset
  setPreference: (next: ThemePreference, origin?: { x: number; y: number }) => void
  toggle: (origin?: { x: number; y: number }) => void
  setAccent: (next: AccentPreset) => void
}

const ThemeContext = React.createContext<ThemeContextValue | null>(null)

const THEME_KEY = 'vsr-theme'
const ACCENT_KEY = 'vsr-accent'

function systemTheme(): ResolvedTheme {
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

function readPreference(): ThemePreference {
  try {
    const stored = localStorage.getItem(THEME_KEY)
    if (stored === 'light' || stored === 'dark' || stored === 'system') return stored
  } catch {
    /* storage unavailable */
  }
  return 'system'
}

function readAccent(): AccentPreset {
  try {
    return localStorage.getItem(ACCENT_KEY) === 'gold' ? 'gold' : 'blue'
  } catch {
    return 'blue'
  }
}

function apply(resolved: ResolvedTheme, accent: AccentPreset) {
  const root = document.documentElement
  root.classList.toggle('dark', resolved === 'dark')
  root.dataset.accent = accent
  root.style.colorScheme = resolved
}

type ViewTransitionDocument = Document & {
  startViewTransition?: (callback: () => void) => { ready: Promise<void> }
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [preference, setPreferenceState] = React.useState<ThemePreference>(readPreference)
  const [accent, setAccentState] = React.useState<AccentPreset>(readAccent)
  const [system, setSystem] = React.useState<ResolvedTheme>(systemTheme)
  const resolved: ResolvedTheme = preference === 'system' ? system : preference

  React.useEffect(() => {
    const media = window.matchMedia('(prefers-color-scheme: dark)')
    const onChange = () => setSystem(media.matches ? 'dark' : 'light')
    media.addEventListener('change', onChange)
    return () => media.removeEventListener('change', onChange)
  }, [])

  React.useLayoutEffect(() => apply(resolved, accent), [resolved, accent])

  const setPreference = React.useCallback(
    (next: ThemePreference, origin?: { x: number; y: number }) => {
      const nextResolved = next === 'system' ? systemTheme() : next
      const commit = () => {
        try {
          localStorage.setItem(THEME_KEY, next)
        } catch {
          /* storage unavailable */
        }
        apply(nextResolved, accent)
        setPreferenceState(next)
      }
      const doc = document as ViewTransitionDocument
      const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
      if (!origin || !doc.startViewTransition || reduce || nextResolved === resolved) {
        commit()
        return
      }
      const radius = Math.hypot(Math.max(origin.x, window.innerWidth - origin.x), Math.max(origin.y, window.innerHeight - origin.y))
      const transition = doc.startViewTransition(commit)
      void transition.ready.then(() => {
        document.documentElement.animate(
          { clipPath: [`circle(0px at ${origin.x}px ${origin.y}px)`, `circle(${radius}px at ${origin.x}px ${origin.y}px)`] },
          { duration: 520, easing: 'cubic-bezier(0.16, 1, 0.3, 1)', pseudoElement: '::view-transition-new(root)' },
        )
      })
    },
    [accent, resolved],
  )

  const setAccent = React.useCallback((next: AccentPreset) => {
    try {
      localStorage.setItem(ACCENT_KEY, next)
    } catch {
      /* storage unavailable */
    }
    setAccentState(next)
  }, [])

  const value = React.useMemo<ThemeContextValue>(
    () => ({
      preference,
      resolved,
      accent,
      setPreference,
      toggle: (origin) => setPreference(resolved === 'dark' ? 'light' : 'dark', origin),
      setAccent,
    }),
    [preference, resolved, accent, setPreference, setAccent],
  )

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}

export function useTheme(): ThemeContextValue {
  const context = React.useContext(ThemeContext)
  if (!context) throw new Error('useTheme must be used inside ThemeProvider')
  return context
}
