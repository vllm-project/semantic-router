import React, { useEffect, useState, type ReactNode } from 'react'
import { NavLink, useLocation, useNavigate } from 'react-router-dom'
import styles from './Layout.module.css'
import LayoutAccountControl from './LayoutAccountControl'
import {
  ANALYSIS_OPERATIONS_MENU_SECTIONS,
  filterLayoutMenuSections,
  hasActiveLayoutMenuSection,
  isLayoutMenuItemActive,
  MANAGER_MENU_SECTIONS,
  PRIMARY_NAV_LINKS,
  SECONDARY_NAV_LINKS,
  type LayoutDropdownKey,
  type LayoutMenuItem,
  type LayoutMenuSection,
  type LayoutNavLink,
} from './LayoutNavSupport'
import { useAuth } from '../contexts/AuthContext'
import { canAccessMLSetup } from '../utils/accessControl'

interface LayoutProps {
  children: ReactNode
  configSection?: string
  onConfigSectionChange?: (section: string) => void
  hideHeaderOnMobile?: boolean
  hideAccountControl?: boolean
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
  const { user, logout } = useAuth()
  const location = useLocation()
  const navigate = useNavigate()
  const canManageUsers = user?.role === 'admin'
  const canUseMLSetup = canAccessMLSetup(user)
  const secondaryNavLinks = SECONDARY_NAV_LINKS.filter((link) => link.to !== '/users' || canManageUsers)
  const analysisOperationsMenuSections = filterLayoutMenuSections(
    ANALYSIS_OPERATIONS_MENU_SECTIONS,
    item => canUseMLSetup || item.kind !== 'route' || item.to !== '/ml-setup'
  )
  const accountName = user?.name?.trim() || 'Account'
  const accountEmail = user?.email?.trim() || 'Session pending'
  const accountPermissions = user?.permissions ?? []

  const isConfigPage = location.pathname === '/config' || location.pathname.startsWith('/config/')
  const isManagerActive = hasActiveLayoutMenuSection(
    MANAGER_MENU_SECTIONS,
    location.pathname,
    isConfigPage,
    configSection
  )
  const isAnalysisOpsActive = hasActiveLayoutMenuSection(
    analysisOperationsMenuSections,
    location.pathname,
    isConfigPage,
    configSection
  )

  const closeMenus = () => {
    setOpenDropdown(null)
    setMobileMenuOpen(false)
    setIsAccountDialogOpen(false)
  }

  const toggleDropdown = (dropdown: LayoutDropdownKey) => {
    setIsAccountDialogOpen(false)
    setOpenDropdown(prev => (prev === dropdown ? null : dropdown))
  }

  const toggleAccountDialog = () => {
    setOpenDropdown(null)
    setMobileMenuOpen(false)
    setIsAccountDialogOpen(prev => !prev)
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
      end
      to={link.to}
      className={({ isActive }) => (isActive ? `${styles.navLink} ${styles.navLinkActive}` : styles.navLink)}
    >
      {link.label}
    </NavLink>
  )

  const renderMenuItem = (
    item: LayoutMenuItem,
    key: string,
    className: string,
    activeClassName: string,
    useMenuRole: boolean
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

  const renderDropdownMenu = (
    sections: LayoutMenuSection[],
    className: string,
    label: string
  ) => (
    <div className={className} role="menu" aria-label={label}>
      {sections.map((section, sectionIndex) => (
        <React.Fragment key={`${label}-${section.title || sectionIndex}`}>
          {sectionIndex > 0 ? <div className={styles.dropdownDivider} /> : null}
          {section.title ? <div className={styles.dropdownSectionLabel}>{section.title}</div> : null}
          {section.items.map(item =>
            renderMenuItem(
              item,
              `${label}-${section.title || 'items'}-${item.label}`,
              styles.dropdownItem,
              styles.dropdownItemActive,
              true
            )
          )}
        </React.Fragment>
      ))}
    </div>
  )

  const renderMobileMenuSection = (title: string, sections: LayoutMenuSection[]) => (
    <div className={styles.mobileNavSection}>
      <div className={styles.mobileNavSectionTitle}>{title}</div>
      {sections.map((section, sectionIndex) => (
        <React.Fragment key={`${title}-${section.title || sectionIndex}`}>
          {section.title ? <div className={styles.mobileNavSubsectionTitle}>{section.title}</div> : null}
          {section.items.map(item =>
            renderMenuItem(
              item,
              `${title}-${section.title || 'items'}-${item.label}`,
              styles.mobileNavLink,
              styles.mobileNavLinkActive,
              false
            )
          )}
        </React.Fragment>
      ))}
    </div>
  )

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
            <span className={styles.brandText}></span>
          </NavLink>

          <nav className={styles.nav} aria-label="Global navigation">
            <div className={styles.navSection} role="group" aria-label="Primary navigation">
              {PRIMARY_NAV_LINKS.map(renderTopNavLink)}
              <div className={styles.navDropdown}>
                <button
                  type="button"
                  aria-expanded={openDropdown === 'manager'}
                  aria-haspopup="menu"
                  className={`${styles.navLink} ${isManagerActive ? styles.navLinkActive : ''}`}
                  onClick={(e) => {
                    e.stopPropagation()
                    toggleDropdown('manager')
                  }}
                >
                  Manager
                  <svg
                    width="12"
                    height="12"
                    viewBox="0 0 12 12"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    className={`${styles.dropdownArrow} ${openDropdown === 'manager' ? styles.dropdownArrowOpen : ''}`}
                  >
                    <path d="M3 4.5L6 7.5L9 4.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </button>
                {openDropdown === 'manager'
                  ? renderDropdownMenu(MANAGER_MENU_SECTIONS, styles.dropdownMenu, 'Manager')
                  : null}
              </div>
            </div>

            <div className={styles.navDivider} />

            <div className={`${styles.navSection} ${styles.navSectionSecondary}`} role="group" aria-label="Secondary navigation">
              {secondaryNavLinks.map(renderTopNavLink)}
              <div className={styles.navDropdown}>
                <button
                  type="button"
                  aria-expanded={openDropdown === 'analysisOps'}
                  aria-haspopup="menu"
                  className={`${styles.navLink} ${isAnalysisOpsActive ? styles.navLinkActive : ''}`}
                  onClick={(e) => {
                    e.stopPropagation()
                    toggleDropdown('analysisOps')
                  }}
                >
                  System
                  <svg
                    width="12"
                    height="12"
                    viewBox="0 0 12 12"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    className={`${styles.dropdownArrow} ${openDropdown === 'analysisOps' ? styles.dropdownArrowOpen : ''}`}
                  >
                    <path d="M3 4.5L6 7.5L9 4.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </button>
                {openDropdown === 'analysisOps'
                  ? renderDropdownMenu(
                      analysisOperationsMenuSections,
                      styles.dropdownMenuRight,
                      'System'
                    )
                  : null}
              </div>
            </div>
          </nav>

          <div className={styles.headerRight}>
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
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
                <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
              </svg>
            </a>

            <button
              type="button"
              className={styles.mobileMenuButton}
              onClick={() => setMobileMenuOpen(prev => !prev)}
              aria-label="Toggle menu"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
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
          <div className={styles.mobileNav}>
            {PRIMARY_NAV_LINKS.map(link => (
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
            {secondaryNavLinks.map(link => (
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
            {renderMobileMenuSection('Manager', MANAGER_MENU_SECTIONS)}
            {renderMobileMenuSection('System', analysisOperationsMenuSections)}
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
