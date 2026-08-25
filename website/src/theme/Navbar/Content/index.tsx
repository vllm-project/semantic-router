import type { ReactNode } from 'react'
import {
  ErrorCauseBoundary,
  ThemeClassNames,
  useThemeConfig,
} from '@docusaurus/theme-common'
import {
  splitNavbarItems,
  useNavbarMobileSidebar,
} from '@docusaurus/theme-common/internal'
import NavbarColorModeToggle from '@theme/Navbar/ColorModeToggle'
import NavbarLogo from '@theme/Navbar/Logo'
import NavbarMobileSidebarToggle from '@theme/Navbar/MobileSidebar/Toggle'
import NavbarSearch from '@theme/Navbar/Search'
import NavbarItem from '@theme/NavbarItem'
import type { Props as NavbarItemConfig } from '@theme/NavbarItem'
import SearchBar from '@theme/SearchBar'
import clsx from 'clsx'
import React from 'react'
import WebsiteMegaNav from '@site/src/components/site/WebsiteMegaNav'
import styles from './styles.module.css'

const CTA_CLASS_NAME = 'nav-dashboard-cta'

function useNavbarItems(): NavbarItemConfig[] {
  return useThemeConfig().navbar.items as NavbarItemConfig[]
}

function NavbarItems({ items }: { items: NavbarItemConfig[] }): ReactNode {
  return (
    <>
      {items.map((item, index) => (
        <ErrorCauseBoundary
          key={index}
          onError={() =>
            new Error(
              `A theme navbar item failed to render.\nPlease double-check the following navbar item (themeConfig.navbar.items) of your Docusaurus config:\n${JSON.stringify(item, null, 2)}`,
            )}
        >
          <NavbarItem {...item} />
        </ErrorCauseBoundary>
      ))}
    </>
  )
}

function NavbarContentLayout({
  left,
  right,
}: {
  left: ReactNode
  right: ReactNode
}): ReactNode {
  return (
    <div className="navbar__inner">
      <div
        className={clsx(
          ThemeClassNames.layout.navbar.containerLeft,
          'navbar__items',
        )}
      >
        {left}
      </div>
      <div
        className={clsx(
          ThemeClassNames.layout.navbar.containerRight,
          'navbar__items navbar__items--right',
        )}
      >
        {right}
      </div>
    </div>
  )
}

export default function NavbarContent(): ReactNode {
  const mobileSidebar = useNavbarMobileSidebar()
  const items = useNavbarItems()
  const [, rightItems] = splitNavbarItems(items)
  const searchBarItem = items.find(item => item.type === 'search')

  /*
   * The call to action is pulled out so it can sit at the far end, after the
   * utility controls. Search used to render after it, which put the widest
   * control on the outside edge and stranded the primary action mid-row.
   *
   * Reading order, left to right: brand, sections, search, utilities, theme,
   * action.
   */
  const ctaItem = rightItems.find(item => item.className === CTA_CLASS_NAME)
  const utilityItems = rightItems.filter(item => item !== ctaItem)

  return (
    <NavbarContentLayout
      left={(
        <>
          {!mobileSidebar.disabled && <NavbarMobileSidebarToggle />}
          <NavbarLogo />
          <WebsiteMegaNav />
        </>
      )}
      right={(
        <>
          {!searchBarItem && (
            <NavbarSearch>
              <SearchBar />
            </NavbarSearch>
          )}
          <NavbarItems items={utilityItems} />
          <NavbarColorModeToggle
            className={clsx(styles.colorModeToggle, 'navbar-color-mode-toggle')}
          />
          {ctaItem && <NavbarItem {...ctaItem} />}
        </>
      )}
    />
  )
}
