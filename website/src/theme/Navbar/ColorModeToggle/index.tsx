import type { ReactNode } from 'react'
import { useColorMode, useThemeConfig } from '@docusaurus/theme-common'
import ColorModeToggle from '@theme/ColorModeToggle'
import type { Props } from '@theme/Navbar/ColorModeToggle'
import React from 'react'

/*
 * Two-state toggle: light <-> dark, never a third "system" step.
 *
 * `colorMode.respectPrefersColorScheme` stays `true` in docusaurus.config.ts so
 * a first-time visitor still lands on whatever their OS is set to. Upstream
 * couples that flag to the toggle as well, which turns it into a three-way
 * light -> dark -> system cycle. The two behaviours are worth separating: the
 * OS preference is a good *default*, but "system" is a confusing thing to make
 * someone click past every time they want the other theme.
 *
 * So: pass `respectPrefersColorScheme={false}` to get the two-value cycle, and
 * feed it `colorMode` (the mode actually on screen) rather than
 * `colorModeChoice` (which is `null` until the reader picks one). Together
 * those mean the button always flips to the opposite of what is being
 * displayed, including on the very first click.
 *
 * The icon is picked in CSS from `html[data-theme-choice]`, which is still
 * `system` before that first click — shell.css remaps that case so the button
 * shows a plain sun/moon instead of the system glyph.
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
