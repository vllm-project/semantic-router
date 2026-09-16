import type { ReactNode } from 'react'
import { useColorMode, useThemeConfig } from '@docusaurus/theme-common'
import ColorModeToggle from '@theme/ColorModeToggle'
import type { Props } from '@theme/Navbar/ColorModeToggle'
import React from 'react'

/*
 * Two-state toggle: light <-> dark, never a third "system" step.
 *
 * `respectPrefersColorScheme` stays true in the config so a first visit follows
 * the OS, but upstream also uses that flag to make the button a three-way
 * cycle. Passing `false` here keeps the default while dropping the extra step.
 *
 * `colorMode` (what is on screen) rather than `colorModeChoice` (null until the
 * reader picks one) so the button flips correctly on the very first click.
 * shell.css handles the icon for that pre-choice state.
 */
export default function NavbarColorModeToggle({ className }: Props): ReactNode {
  const { disableSwitch } = useThemeConfig().colorMode
  const { colorMode, setColorMode } = useColorMode()

  if (disableSwitch) {
    return null
  }

  return (
    <ColorModeToggle
      className={className}
      respectPrefersColorScheme={false}
      value={colorMode}
      onChange={setColorMode}
    />
  )
}
