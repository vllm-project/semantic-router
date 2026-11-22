import { useTheme as useShellTheme, type ResolvedTheme } from '../shell/theme'

export type Theme = ResolvedTheme

/** Resolved light or dark theme for components that draw their own canvases or editors. */
export function useTheme() {
  const { resolved, setPreference, toggle } = useShellTheme()
  return {
    theme: resolved,
    setTheme: (theme: Theme) => setPreference(theme),
    toggleTheme: () => toggle(),
    isDark: resolved === 'dark',
    isLight: resolved === 'light',
  }
}
