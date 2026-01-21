// hooks/useTheme.ts - Theme management hook (Dark mode only)

import { useEffect } from 'react'

export type Theme = 'dark' | 'light'

export function useTheme() {
  // Always use dark theme
  const theme: Theme = 'dark'

  // Apply dark theme to document on mount
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', 'dark')
    document.documentElement.style.colorScheme = 'dark'
  }, [])

  return {
    theme,
    setTheme: () => {}, // No-op
    toggleTheme: () => {}, // No-op
    isDark: true,
    isLight: false,
  }
}
