import React, { useEffect, useRef, useState, type ReactNode } from 'react'
import { NavLink, useLocation, useNavigate } from 'react-router-dom'
import styles from './Layout.module.css'
import LayoutAccountControl from './LayoutAccountControl'
import LayoutMegaMenu from './LayoutMegaMenu'
import PlatformBranding from './PlatformBranding'
import {
  ANALYSIS_OPERATIONS_MENU_SECTIONS,
  FLEET_SIM_MENU_SECTIONS,
  filterLayoutMenuSections,
  hasActiveLayoutMenuSection,
  isLayoutMenuItemActive,
  KNOWLEDGE_BASE_MENU_SECTIONS,
  MANAGER_MENU_SECTIONS,
  PRIMARY_NAV_LINKS,
  SECONDARY_NAV_LINKS,
  type LayoutDropdownKey,
  type LayoutMenuItem,
  type LayoutMenuSection,
  type LayoutNavLink,
} from './LayoutNavSupport'
import { useAuth } from '../contexts/AuthContext'
import { useReadonly } from '../contexts/ReadonlyContext'
import { canAccessMLSetup } from '../utils/accessControl'

interface LayoutProps {
  children: ReactNode
  configSection?: string
  onConfigSectionChange?: (section: string) => void
  hideHeaderOnMobile?: boolean
  hideAccountControl?: boolean
}

const DESKTOP_MENU_IDS: Record<LayoutDropdownKey, string> = {
  manager: 'layout-mega-menu-manager',
  simulator: 'layout-mega-menu-simulator',
  analysisOps: 'layout-mega-menu-system',
}

const DESKTOP_MENU_TRIGGER_IDS: Record<LayoutDropdownKey, string> = {
  manager: 'layout-mega-menu-trigger-manager',
  simulator: 'layout-mega-menu-trigger-simulator',
  analysisOps: 'layout-mega-menu-trigger-system',
}

const Layout: React.FC<LayoutProps> = ({
  children,
  configSection,
  onConfigSectionChange,
  hideHeaderOnMobile,
  hideAccountControl = false,
}) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [openDropdown, setOpenDropdown] = useState<LayoutDropdownKey | null>(null)
  const [isAccountDialogOpen, setIsAccountDialogOpen] = useState(false)
  const dropdownTriggerRefs = useRef<Partial<Record<LayoutDropdownKey, HTMLButtonElement | null>>>(
    {},
  )
  const pendingMenuFocusRef = useRef<'first' | 'last'>('first')
  const { user, logout } = useAuth()
  const { fleetSimEnabled } = useReadonly()
  const location = useLocation()
  const navigate = useNavigate()
  const canManageUsers = user?.role === 'admin'
  const canUseMLSetup = canAccessMLSetup(user)
  const secondaryNavLinks = SECONDARY_NAV_LINKS.filter(
    (link) => link.to !== '/users' || canManageUsers,
  )
  const managerMenuSections = filterLayoutMenuSections(
    [...MANAGER_MENU_SECTIONS, ...KNOWLEDGE_BASE_MENU_SECTIONS],
    (item) => canManageUsers || item.kind !== 'route' || item.to !== '/users',
  )
  const analysisOperationsMenuSections = filterLayoutMenuSections(
    ANALYSIS_OPERATIONS_MENU_SECTIONS,
    (item) => canUseMLSetup || item.kind !== 'route' || item.to !== '/ml-setup',
  )
  const systemMenuSections = analysisOperationsMenuSections
  const simulatorMenuSections = fleetSimEnabled ? FLEET_SIM_MENU_SECTIONS : []
  const accountName = user?.name?.trim() || 'Account'
  const accountEmail = user?.email?.trim() || 'Session pending'
  const accountPermissions = user?.permissions ?? []

  const isConfigPage = location.pathname === '/config' || location.pathname.startsWith('/config/')
  const isManagerActive = hasActiveLayoutMenuSection(
    managerMenuSections,
    location.pathname,
    isConfigPage,
    configSection,
  )
  const isAnalysisOpsActive = hasActiveLayoutMenuSection(
    systemMenuSections,
    location.pathname,
    isConfigPage,
    configSection,
  )
  const isSimulatorActive = hasActiveLayoutMenuSection(
    simulatorMenuSections,
    location.pathname,
    isConfigPage,
    configSection,
  )

  const closeMenus = () => {
    setOpenDropdown(null)
    setMobileMenuOpen(false)
    setIsAccountDialogOpen(false)
  }

  const toggleDropdown = (dropdown: LayoutDropdownKey) => {
    setIsAccountDialogOpen(false)
    pendingMenuFocusRef.current = 'first'
    setOpenDropdown((prev) => (prev === dropdown ? null : dropdown))
  }

  const openDropdownFromKeyboard = (
    dropdown: LayoutDropdownKey,
    focusTarget: 'first' | 'last',
  ) => {
    setIsAccountDialogOpen(false)
    pendingMenuFocusRef.current = focusTarget

    if (openDropdown === dropdown) {
      const menu = document.getElementById(DESKTOP_MENU_IDS[dropdown])
      const menuItems = menu?.querySelectorAll<HTMLElement>('[role="menuitem"]:not([disabled])')
      const targetIndex = focusTarget === 'last' ? (menuItems?.length ?? 1) - 1 : 0
      menuItems?.[targetIndex]?.focus()
      pendingMenuFocusRef.current = 'first'
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

  const renderMenuItem = (
    item: LayoutMenuItem,
    key: string,
    className: string,
    activeClassName: string,
    useMenuRole: boolean,
  ) => {
    const active = isLayoutMenuItemActive(item, location.pathname, isConfigPage, configSection)
    const roleProps = useMenuRole ? { role: 'menuitem' as const } : {}

    if (item.kind === 'config') {
      return (
        <button
          key={key}
          type="button"
          {...roleProps}
          className={`${className} ${active ? activeClassName : ''}`}
          onClick={() => handleMenuItemSelect(item)}
        >
          {item.label}
        </button>
      )
    }

    return (
      <NavLink
        key={key}
        {...roleProps}
        to={item.to}
        className={`${className} ${active ? activeClassName : ''}`}
        onClick={closeMenus}
      >
        {item.label}
      </NavLink>
    )
  }

  const renderMobileMenuSection = (title: string, sections: LayoutMenuSection[]) => (
    <div className={styles.mobileNavSection}>
      <div className={styles.mobileNavSectionTitle}>{title}</div>
      {sections.map((section, sectionIndex) => (
        <React.Fragment key={`${title}-${section.title || sectionIndex}`}>
          {section.title ? (
            <div className={styles.mobileNavSubsectionTitle}>{section.title}</div>
          ) : null}
          {section.items.map((item) =>
            renderMenuItem(
              item,
              `${title}-${section.title || 'items'}-${item.label}`,
              styles.mobileNavLink,
              styles.mobileNavLinkActive,
              false,
            ),
          )}
        </React.Fragment>
      ))}
    </div>
  )

  const renderDesktopDropdown = (
    dropdown: LayoutDropdownKey,
    label: string,
    sections: LayoutMenuSection[],
    active: boolean,
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
          aria-haspopup="menu"
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
            openDropdownFromKeyboard(dropdown, event.key === 'ArrowUp' ? 'last' : 'first')
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
            sections={sections}
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
    const menuItems = menu?.querySelectorAll<HTMLElement>('[role="menuitem"]:not([disabled])')
    const targetIndex = pendingMenuFocusRef.current === 'last' ? (menuItems?.length ?? 1) - 1 : 0
    menuItems?.[targetIndex]?.focus()
    pendingMenuFocusRef.current = 'first'

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
              aria-label="Secondary navigation"
            >
              {secondaryNavLinks.map(renderTopNavLink)}
              {renderDesktopDropdown('manager', 'Manager', managerMenuSections, isManagerActive)}
              {fleetSimEnabled
                ? renderDesktopDropdown(
                    'simulator',
                    'Simulator',
                    simulatorMenuSections,
                    isSimulatorActive,
                  )
                : null}
              {renderDesktopDropdown(
                'analysisOps',
                'System',
                systemMenuSections,
                isAnalysisOpsActive,
              )}
            </div>
          </nav>

          <div className={styles.headerRight}>
            <PlatformBranding variant="inline" className={styles.headerBranding} />
            {hideAccountControl ? null : (
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
            )}
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

            <button
              type="button"
              className={styles.mobileMenuButton}
              onClick={() => setMobileMenuOpen((prev) => !prev)}
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
          <div
            id="mobile-navigation"
            className={styles.mobileNav}
            role="navigation"
            aria-label="Mobile navigation"
          >
            {PRIMARY_NAV_LINKS.map((link) => (
              <NavLink
                key={`mobile-${link.to}`}
                end
                to={link.to}
                className={styles.mobileNavLink}
                onClick={closeMenus}
              >
                {link.label}
              </NavLink>
            ))}
            {secondaryNavLinks.map((link) => (
              <NavLink
                key={`mobile-${link.to}`}
                end
                to={link.to}
                className={styles.mobileNavLink}
                onClick={closeMenus}
              >
                {link.label}
              </NavLink>
            ))}
            {renderMobileMenuSection('Manager', managerMenuSections)}
            {fleetSimEnabled ? renderMobileMenuSection('Simulator', simulatorMenuSections) : null}
            {renderMobileMenuSection('System', systemMenuSections)}
          </div>
        ) : null}
      </header>

      <main className={styles.main}>
        <div className={styles.mainContent}>{children}</div>
      </main>
    </div>
  )
}

export default Layout
