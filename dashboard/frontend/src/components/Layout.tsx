import React, { useEffect, useRef, useState, type ReactNode } from 'react'
import { NavLink, useLocation, useNavigate } from 'react-router-dom'
import styles from './Layout.module.css'
import LayoutAccountControl from './LayoutAccountControl'
import LayoutMegaMenu from './LayoutMegaMenu'
import LayoutMobileNavigation from './LayoutMobileNavigation'
import PlatformBranding from './PlatformBranding'
import {
  ANALYZE_MENU_CATEGORIES,
  BUILD_MENU_CATEGORIES,
  filterLayoutMenuCategories,
  findActiveLayoutMenuCategory,
  hasActiveLayoutMenuCategory,
  isLayoutMenuItemActive,
  OPERATE_MENU_CATEGORIES,
  PRIMARY_NAV_LINKS,
  type LayoutDropdownKey,
  type LayoutMenuCategory,
  type LayoutMenuItem,
  type LayoutNavLink,
} from './LayoutNavSupport'
import { useAuth } from '../contexts/AuthContext'
import { useReadonly } from '../contexts/ReadonlyContext'
import { canAccessMLSetup, canViewUsers } from '../utils/accessControl'

interface LayoutProps {
  children: ReactNode
  configSection?: string
  onConfigSectionChange?: (section: string) => void
  hideHeaderOnMobile?: boolean
  hideAccountControl?: boolean
}

const DESKTOP_MENU_IDS: Record<LayoutDropdownKey, string> = {
  build: 'layout-mega-menu-build',
  analyze: 'layout-mega-menu-analyze',
  operate: 'layout-mega-menu-operate',
}

const DESKTOP_MENU_TRIGGER_IDS: Record<LayoutDropdownKey, string> = {
  build: 'layout-mega-menu-trigger-build',
  analyze: 'layout-mega-menu-trigger-analyze',
  operate: 'layout-mega-menu-trigger-operate',
}

const Layout: React.FC<LayoutProps> = ({
  children,
  configSection,
  onConfigSectionChange,
  hideHeaderOnMobile,
  hideAccountControl = false,
}) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [openMobileSection, setOpenMobileSection] = useState<LayoutDropdownKey | null>(null)
  const [openDropdown, setOpenDropdown] = useState<LayoutDropdownKey | null>(null)
  const [isAccountDialogOpen, setIsAccountDialogOpen] = useState(false)
  const dropdownTriggerRefs = useRef<Partial<Record<LayoutDropdownKey, HTMLButtonElement | null>>>(
    {},
  )
  const pendingMenuFocusRef = useRef<'active-tab' | 'last-link' | null>(null)
  const { user, logout } = useAuth()
  const { fleetSimEnabled } = useReadonly()
  const location = useLocation()
  const navigate = useNavigate()
  const canAccessUsers = canViewUsers(user)
  const canUseMLSetup = canAccessMLSetup(user)
  const buildMenuCategories = filterLayoutMenuCategories(BUILD_MENU_CATEGORIES, () => true)
  const analyzeMenuCategories = filterLayoutMenuCategories(
    ANALYZE_MENU_CATEGORIES,
    (item, category) =>
      (fleetSimEnabled || category.key !== 'fleet-simulation') &&
      (canUseMLSetup || item.kind !== 'route' || item.to !== '/ml-setup'),
  )
  const operateMenuCategories = filterLayoutMenuCategories(
    OPERATE_MENU_CATEGORIES,
    (item) => canAccessUsers || item.kind !== 'route' || item.to !== '/users',
  )
  const accountName = user?.name?.trim() || 'Account'
  const accountEmail = user?.email?.trim() || 'Session pending'
  const accountPermissions = user?.permissions ?? []

  const isConfigPage = location.pathname === '/config' || location.pathname.startsWith('/config/')
  const isBuildActive = hasActiveLayoutMenuCategory(
    buildMenuCategories,
    location.pathname,
    isConfigPage,
    configSection,
  )
  const isAnalyzeActive = hasActiveLayoutMenuCategory(
    analyzeMenuCategories,
    location.pathname,
    isConfigPage,
    configSection,
  )
  const isOperateActive = hasActiveLayoutMenuCategory(
    operateMenuCategories,
    location.pathname,
    isConfigPage,
    configSection,
  )

  const activeBuildCategory = findActiveLayoutMenuCategory(
    buildMenuCategories,
    location.pathname,
    isConfigPage,
    configSection,
  )
  const activeAnalyzeCategory = findActiveLayoutMenuCategory(
    analyzeMenuCategories,
    location.pathname,
    isConfigPage,
    configSection,
  )
  const activeOperateCategory = findActiveLayoutMenuCategory(
    operateMenuCategories,
    location.pathname,
    isConfigPage,
    configSection,
  )

  const closeMenus = () => {
    setOpenDropdown(null)
    setMobileMenuOpen(false)
    setOpenMobileSection(null)
    setIsAccountDialogOpen(false)
  }

  const toggleDropdown = (dropdown: LayoutDropdownKey) => {
    setIsAccountDialogOpen(false)
    setOpenDropdown((currentDropdown) => {
      const nextDropdown = currentDropdown === dropdown ? null : dropdown
      pendingMenuFocusRef.current = nextDropdown ? 'active-tab' : null
      return nextDropdown
    })
  }

  const openDropdownFromKeyboard = (
    dropdown: LayoutDropdownKey,
    focusTarget: 'active-tab' | 'last-link',
  ) => {
    setIsAccountDialogOpen(false)
    pendingMenuFocusRef.current = focusTarget

    if (openDropdown === dropdown) {
      const menu = document.getElementById(DESKTOP_MENU_IDS[dropdown])
      const menuLinks = Array.from(menu?.querySelectorAll<HTMLElement>('[data-mega-link]') ?? [])
      const target =
        focusTarget === 'last-link'
          ? menuLinks[menuLinks.length - 1]
          : menu?.querySelector<HTMLElement>('[role="tab"][aria-selected="true"]')
      target?.focus()
      pendingMenuFocusRef.current = null
      return
    }

    setOpenDropdown(dropdown)
  }

  const toggleAccountDialog = () => {
    setOpenDropdown(null)
    setMobileMenuOpen(false)
    setIsAccountDialogOpen((prev) => !prev)
  }

  const handleMenuItemSelect = (item: LayoutMenuItem) => {
    if (item.kind === 'config') {
      onConfigSectionChange?.(item.configSection)
      navigate(`/config/${item.configSection}`)
    } else {
      navigate(item.to)
    }
    closeMenus()
  }

  const handleLogout = () => {
    logout()
    closeMenus()
    navigate('/login', { replace: true })
  }

  const renderTopNavLink = (link: LayoutNavLink) => (
    <NavLink
      key={link.to}
      end={link.matchMode !== 'prefix'}
      to={link.to}
      className={({ isActive }) =>
        isActive ? `${styles.navLink} ${styles.navLinkActive}` : styles.navLink
      }
    >
      {link.label}
    </NavLink>
  )

  const renderDesktopDropdown = (
    dropdown: LayoutDropdownKey,
    label: string,
    categories: LayoutMenuCategory[],
    active: boolean,
    activeCategoryKey?: string,
  ) => {
    const isOpen = openDropdown === dropdown
    const menuId = DESKTOP_MENU_IDS[dropdown]
    const triggerId = DESKTOP_MENU_TRIGGER_IDS[dropdown]

    return (
      <div className={styles.navDropdown}>
        <button
          id={triggerId}
          ref={(element) => {
            dropdownTriggerRefs.current[dropdown] = element
          }}
          type="button"
          aria-controls={menuId}
          aria-expanded={isOpen}
          aria-haspopup="dialog"
          className={`${styles.navLink} ${active ? styles.navLinkActive : ''}`}
          onClick={(event) => {
            event.stopPropagation()
            toggleDropdown(dropdown)
          }}
          onKeyDown={(event) => {
            if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') {
              return
            }

            event.preventDefault()
            openDropdownFromKeyboard(dropdown, event.key === 'ArrowUp' ? 'last-link' : 'active-tab')
          }}
          onBlur={(event) => {
            const nextTarget = event.relatedTarget
            const menu = document.getElementById(menuId)
            if (nextTarget instanceof Node && menu?.contains(nextTarget)) {
              return
            }

            if (openDropdown === dropdown) {
              setOpenDropdown(null)
            }
          }}
        >
          {label}
          <svg
            width="12"
            height="12"
            viewBox="0 0 12 12"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            className={`${styles.dropdownArrow} ${isOpen ? styles.dropdownArrowOpen : ''}`}
            aria-hidden="true"
          >
            <path d="M3 4.5L6 7.5L9 4.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>

        {isOpen ? (
          <LayoutMegaMenu
            id={menuId}
            triggerId={triggerId}
            label={label}
            categories={categories}
            activeCategoryKey={activeCategoryKey}
            isItemActive={(item) =>
              isLayoutMenuItemActive(item, location.pathname, isConfigPage, configSection)
            }
            onConfigSelect={handleMenuItemSelect}
            onNavigate={closeMenus}
          />
        ) : null}
      </div>
    )
  }

  useEffect(() => {
    if (!openDropdown) {
      return
    }

    const menu = document.getElementById(DESKTOP_MENU_IDS[openDropdown])
    if (pendingMenuFocusRef.current === 'last-link') {
      const menuLinks = Array.from(menu?.querySelectorAll<HTMLElement>('[data-mega-link]') ?? [])
      menuLinks[menuLinks.length - 1]?.focus()
    } else if (pendingMenuFocusRef.current === 'active-tab') {
      menu?.querySelector<HTMLElement>('[role="tab"][aria-selected="true"]')?.focus()
    }
    pendingMenuFocusRef.current = null

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') {
        return
      }

      const trigger = dropdownTriggerRefs.current[openDropdown]
      event.preventDefault()
      setOpenDropdown(null)
      trigger?.focus()
    }

    document.addEventListener('keydown', handleEscape)
    return () => document.removeEventListener('keydown', handleEscape)
  }, [openDropdown])

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as HTMLElement
      if (!target.closest(`.${styles.navDropdown}`)) {
        setOpenDropdown(null)
      }
    }

    document.addEventListener('click', handleClickOutside)
    return () => document.removeEventListener('click', handleClickOutside)
  }, [])

  return (
    <div className={`${styles.container} ${hideHeaderOnMobile ? styles.hideHeaderMobile : ''}`}>
      <header className={`${styles.header} ${hideHeaderOnMobile ? styles.headerHideMobile : ''}`}>
        <div className={styles.headerContent}>
          <NavLink to="/" className={styles.brand}>
            <img src="/vllm.png" alt="vLLM" className={styles.logo} />
            <span className={styles.brandText}>Semantic Router</span>
          </NavLink>

          <nav className={styles.nav} aria-label="Global navigation">
            <div className={styles.navSection} role="group" aria-label="Primary navigation">
              {PRIMARY_NAV_LINKS.map(renderTopNavLink)}
            </div>

            <div className={styles.navDivider} />

            <div
              className={`${styles.navSection} ${styles.navSectionSecondary}`}
              role="group"
              aria-label="Workflow navigation"
            >
              {renderDesktopDropdown(
                'build',
                'Build',
                buildMenuCategories,
                isBuildActive,
                activeBuildCategory,
              )}
              {renderDesktopDropdown(
                'analyze',
                'Analyze',
                analyzeMenuCategories,
                isAnalyzeActive,
                activeAnalyzeCategory,
              )}
              {renderDesktopDropdown(
                'operate',
                'Operate',
                operateMenuCategories,
                isOperateActive,
                activeOperateCategory,
              )}
            </div>
          </nav>

          <div className={styles.headerRight}>
            <PlatformBranding variant="inline" className={styles.headerBranding} />
            <a
              href="https://vllm-semantic-router.com"
              target="_blank"
              rel="noopener noreferrer"
              className={styles.iconButton}
              aria-label="Documentation"
              title="Documentation"
            >
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
                <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
              </svg>
            </a>
            <a
              href="https://github.com/vllm-project/semantic-router"
              target="_blank"
              rel="noopener noreferrer"
              className={styles.iconButton}
              aria-label="GitHub"
              title="GitHub Repository"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
              </svg>
            </a>
            {hideAccountControl ? null : (
              <>
                <span className={styles.utilityDivider} aria-hidden="true" />
                <LayoutAccountControl
                  accountName={accountName}
                  accountEmail={accountEmail}
                  accountRole={user?.role}
                  accountPermissions={accountPermissions}
                  isOpen={isAccountDialogOpen}
                  onToggle={toggleAccountDialog}
                  onClose={closeMenus}
                  onLogout={handleLogout}
                />
              </>
            )}

            <button
              type="button"
              className={styles.mobileMenuButton}
              onClick={() => {
                setOpenMobileSection(null)
                setMobileMenuOpen((prev) => !prev)
              }}
              aria-label="Toggle menu"
              aria-controls="mobile-navigation"
              aria-expanded={mobileMenuOpen}
            >
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
              >
                {mobileMenuOpen ? (
                  <>
                    <path d="M18 6L6 18" />
                    <path d="M6 6L18 18" />
                  </>
                ) : (
                  <>
                    <path d="M4 6h16" />
                    <path d="M4 12h16" />
                    <path d="M4 18h16" />
                  </>
                )}
              </svg>
            </button>
          </div>
        </div>

        {mobileMenuOpen ? (
          <LayoutMobileNavigation
            configSection={configSection}
            isConfigPage={isConfigPage}
            openSection={openMobileSection}
            pathname={location.pathname}
            sections={[
              { key: 'build', label: 'Build', categories: buildMenuCategories },
              { key: 'analyze', label: 'Analyze', categories: analyzeMenuCategories },
              { key: 'operate', label: 'Operate', categories: operateMenuCategories },
            ]}
            onConfigSelect={handleMenuItemSelect}
            onNavigate={closeMenus}
            onSectionToggle={(section) =>
              setOpenMobileSection((current) => (current === section ? null : section))
            }
          />
        ) : null}
      </header>

      <main className={styles.main}>
        <div className={styles.mainContent}>{children}</div>
      </main>
    </div>
  )
}

export default Layout
