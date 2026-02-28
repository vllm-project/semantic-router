import React, { useState, useEffect, ReactNode } from 'react'
import { NavLink, useLocation, useNavigate } from 'react-router-dom'
import styles from './Layout.module.css'

interface LayoutProps {
  children: ReactNode
  configSection?: string
  onConfigSectionChange?: (section: string) => void
  hideHeaderOnMobile?: boolean
}

const Layout: React.FC<LayoutProps> = ({ children, configSection, onConfigSectionChange, hideHeaderOnMobile }) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [openDropdown, setOpenDropdown] = useState<'analysis' | 'operations' | null>(null)
  const location = useLocation()
  const navigate = useNavigate()

  const isConfigPage = location.pathname === '/config'

  // Active state detection
  const isModelsActive = isConfigPage && configSection === 'models'
  const isSignalsActive = isConfigPage && configSection === 'signals'
  const isDecisionsActive = isConfigPage && configSection === 'decisions'
  const isAnalysisChildActive =
    location.pathname === '/evaluation' ||
    location.pathname === '/replay' ||
    location.pathname === '/ratings'
  const isOperationsChildActive =
    location.pathname === '/ml-setup' ||
    (isConfigPage && configSection === 'router-config') ||
    (isConfigPage && configSection === 'mcp') ||
    ['/status', '/logs', '/monitoring', '/tracing'].includes(location.pathname)

  const toggleDropdown = (dropdown: 'analysis' | 'operations') => {
    setOpenDropdown(prev => prev === dropdown ? null : dropdown)
  }

  // Close dropdown when clicking outside
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
      {/* Top Navigation Bar */}
      <header className={`${styles.header} ${hideHeaderOnMobile ? styles.headerHideMobile : ''}`}>
        <div className={styles.headerContent}>
          {/* Left: Brand */}
          <NavLink to="/" className={styles.brand}>
            <img src="/vllm.png" alt="vLLM" className={styles.logo} />
            <span className={styles.brandText}></span>
          </NavLink>

          {/* Center: Navigation - Big Three + Dropdowns */}
          <nav className={styles.nav}>
            {/* Primary: Playground */}
            <NavLink
              to="/playground"
              className={({ isActive }) =>
                isActive ? `${styles.navLink} ${styles.navLinkActive} ${styles.playgroundActive}` : styles.navLink
              }
            >
              Playground
            </NavLink>

            {/* Primary: Brain */}
            <NavLink
              to="/topology"
              className={({ isActive }) =>
                isActive ? `${styles.navLink} ${styles.navLinkActive}` : styles.navLink
              }
            >
              Brain
            </NavLink>

            {/* Primary: Models */}
            <button
              className={`${styles.navLink} ${isModelsActive ? styles.navLinkActive : ''}`}
              onClick={() => {
                onConfigSectionChange?.('models')
                navigate('/config')
              }}
            >
              Models
            </button>

            {/* Primary: Signals */}
            <button
              className={`${styles.navLink} ${isSignalsActive ? styles.navLinkActive : ''}`}
              onClick={() => {
                onConfigSectionChange?.('signals')
                navigate('/config')
              }}
            >
              Signals
            </button>

            {/* Primary: Decisions */}
            <button
              className={`${styles.navLink} ${isDecisionsActive ? styles.navLinkActive : ''}`}
              onClick={() => {
                onConfigSectionChange?.('decisions')
                navigate('/config')
              }}
            >
              Decisions
            </button>

            {/* Divider */}
            <div className={styles.navDivider} />

            <NavLink
              to="/builder"
              className={({ isActive }) =>
                isActive ? `${styles.navLink} ${styles.navLinkActive}` : styles.navLink
              }
            >
              Builder
            </NavLink>

            {/* Analysis Dropdown */}
            <div className={styles.navDropdown}>
              <button
                className={`${styles.navLink} ${styles.dropdownTrigger} ${isAnalysisChildActive ? styles.navLinkActive : ''}`}
                onClick={(e) => {
                  e.stopPropagation()
                  toggleDropdown('analysis')
                }}
              >
                Analysis
                <svg
                  width="12"
                  height="12"
                  viewBox="0 0 12 12"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  className={`${styles.dropdownArrow} ${openDropdown === 'analysis' ? styles.dropdownArrowOpen : ''}`}
                >
                  <path d="M3 4.5L6 7.5L9 4.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </button>
              {openDropdown === 'analysis' && (
                <div className={styles.dropdownMenu}>
                  <NavLink
                    to="/evaluation"
                    className={`${styles.dropdownItem} ${location.pathname === '/evaluation' ? styles.dropdownItemActive : ''}`}
                    onClick={() => setOpenDropdown(null)}
                  >
                    Evaluation
                  </NavLink>
                  <NavLink
                    to="/replay"
                    className={`${styles.dropdownItem} ${location.pathname === '/replay' ? styles.dropdownItemActive : ''}`}
                    onClick={() => setOpenDropdown(null)}
                  >
                    Replay
                  </NavLink>
                  <NavLink
                    to="/ratings"
                    className={`${styles.dropdownItem} ${location.pathname === '/ratings' ? styles.dropdownItemActive : ''}`}
                    onClick={() => setOpenDropdown(null)}
                  >
                    Ratings
                  </NavLink>
                </div>
              )}
            </div>

            {/* Operations Dropdown */}
            <div className={styles.navDropdown}>
              <button
                className={`${styles.navLink} ${styles.dropdownTrigger} ${isOperationsChildActive ? styles.navLinkActive : ''}`}
                onClick={(e) => {
                  e.stopPropagation()
                  toggleDropdown('operations')
                }}
              >
                Operations
                <svg
                  width="12"
                  height="12"
                  viewBox="0 0 12 12"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  className={`${styles.dropdownArrow} ${openDropdown === 'operations' ? styles.dropdownArrowOpen : ''}`}
                >
                  <path d="M3 4.5L6 7.5L9 4.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </button>
              {openDropdown === 'operations' && (
                <div className={styles.dropdownMenuRight}>
                  {/* ML Tools */}
                  <NavLink
                    to="/ml-setup"
                    className={`${styles.dropdownItem} ${location.pathname === '/ml-setup' ? styles.dropdownItemActive : ''}`}
                    onClick={() => setOpenDropdown(null)}
                  >
                    ML Setup
                  </NavLink>
                  <div className={styles.dropdownDivider}></div>
                  {/* Config */}
                  <button
                    className={`${styles.dropdownItem} ${isConfigPage && configSection === 'router-config' ? styles.dropdownItemActive : ''}`}
                    onClick={() => {
                      onConfigSectionChange?.('router-config')
                      navigate('/config')
                      setOpenDropdown(null)
                    }}
                  >
                    Router Config
                  </button>
                  <button
                    className={`${styles.dropdownItem} ${isConfigPage && configSection === 'mcp' ? styles.dropdownItemActive : ''}`}
                    onClick={() => {
                      onConfigSectionChange?.('mcp')
                      navigate('/config')
                      setOpenDropdown(null)
                    }}
                  >
                    MCP Servers
                  </button>
                  <div className={styles.dropdownDivider}></div>
                  {/* Observability */}
                  <NavLink
                    to="/status"
                    className={`${styles.dropdownItem} ${location.pathname === '/status' ? styles.dropdownItemActive : ''}`}
                    onClick={() => setOpenDropdown(null)}
                  >
                    Status
                  </NavLink>
                  <NavLink
                    to="/logs"
                    className={`${styles.dropdownItem} ${location.pathname === '/logs' ? styles.dropdownItemActive : ''}`}
                    onClick={() => setOpenDropdown(null)}
                  >
                    Logs
                  </NavLink>
                  <NavLink
                    to="/monitoring"
                    className={`${styles.dropdownItem} ${location.pathname === '/monitoring' ? styles.dropdownItemActive : ''}`}
                    onClick={() => setOpenDropdown(null)}
                  >
                    Grafana
                  </NavLink>
                  <NavLink
                    to="/tracing"
                    className={`${styles.dropdownItem} ${location.pathname === '/tracing' ? styles.dropdownItemActive : ''}`}
                    onClick={() => setOpenDropdown(null)}
                  >
                    Tracing
                  </NavLink>
                </div>
              )}
            </div>
          </nav>

          {/* Right: Actions */}
          <div className={styles.headerRight}>
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

            {/* Mobile menu button */}
            <button
              className={styles.mobileMenuButton}
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
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

        {/* Mobile Navigation */}
        {mobileMenuOpen && (
          <div className={styles.mobileNav}>
            {/* Primary items */}
            <NavLink to="/playground" className={styles.mobileNavLink} onClick={() => setMobileMenuOpen(false)}>
              Playground
            </NavLink>
            <NavLink to="/topology" className={styles.mobileNavLink} onClick={() => setMobileMenuOpen(false)}>
              Brain
            </NavLink>
            <button
              className={styles.mobileNavLink}
              onClick={() => {
                onConfigSectionChange?.('models')
                navigate('/config')
                setMobileMenuOpen(false)
              }}
            >
              Models
            </button>
            <button
              className={styles.mobileNavLink}
              onClick={() => {
                onConfigSectionChange?.('signals')
                navigate('/config')
                setMobileMenuOpen(false)
              }}
            >
              Signals
            </button>
            <button
              className={styles.mobileNavLink}
              onClick={() => {
                onConfigSectionChange?.('decisions')
                navigate('/config')
                setMobileMenuOpen(false)
              }}
            >
              Decisions
            </button>
            <NavLink to="/builder" className={styles.mobileNavLink} onClick={() => setMobileMenuOpen(false)}>
              Builder
            </NavLink>

            {/* Analysis section */}
            <div className={styles.mobileNavSection}>
              <div className={styles.mobileNavSectionTitle}>Analysis</div>
              <NavLink to="/evaluation" className={styles.mobileNavLink} onClick={() => setMobileMenuOpen(false)}>
                Evaluation
              </NavLink>
              <NavLink to="/replay" className={styles.mobileNavLink} onClick={() => setMobileMenuOpen(false)}>
                Replay
              </NavLink>
              <NavLink to="/ratings" className={styles.mobileNavLink} onClick={() => setMobileMenuOpen(false)}>
                Ratings
              </NavLink>
            </div>

            {/* Operations section */}
            <div className={styles.mobileNavSection}>
              <div className={styles.mobileNavSectionTitle}>Operations</div>
              <NavLink to="/ml-setup" className={styles.mobileNavLink} onClick={() => setMobileMenuOpen(false)}>
                ML Setup
              </NavLink>
              <button
                className={styles.mobileNavLink}
                onClick={() => {
                  onConfigSectionChange?.('router-config')
                  navigate('/config')
                  setMobileMenuOpen(false)
                }}
              >
                Router Config
              </button>
              <button
                className={styles.mobileNavLink}
                onClick={() => {
                  onConfigSectionChange?.('mcp')
                  navigate('/config')
                  setMobileMenuOpen(false)
                }}
              >
                MCP Servers
              </button>
              <NavLink to="/status" className={styles.mobileNavLink} onClick={() => setMobileMenuOpen(false)}>
                Status
              </NavLink>
              <NavLink to="/logs" className={styles.mobileNavLink} onClick={() => setMobileMenuOpen(false)}>
                Logs
              </NavLink>
              <NavLink to="/monitoring" className={styles.mobileNavLink} onClick={() => setMobileMenuOpen(false)}>
                Grafana
              </NavLink>
              <NavLink to="/tracing" className={styles.mobileNavLink} onClick={() => setMobileMenuOpen(false)}>
                Tracing
              </NavLink>
            </div>
          </div>
        )}
      </header>

      {/* Main Content */}
      <main className={styles.main}>
        <div className={styles.mainContent}>{children}</div>
      </main>
    </div>
  )
}

export default Layout
