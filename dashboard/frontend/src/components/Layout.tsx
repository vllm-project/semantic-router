import React, { useState, useEffect, ReactNode } from 'react'
import { NavLink, useLocation, useNavigate } from 'react-router-dom'
import styles from './Layout.module.css'

interface LayoutProps {
  children: ReactNode
  configSection?: string
  onConfigSectionChange?: (section: string) => void
}

const Layout: React.FC<LayoutProps> = ({ children, configSection, onConfigSectionChange }) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [configDropdownOpen, setConfigDropdownOpen] = useState(false)
  const [observabilityDropdownOpen, setObservabilityDropdownOpen] = useState(false)
  const location = useLocation()
  const navigate = useNavigate()
  const isConfigPage = location.pathname === '/config'
  const isObservabilityPage = ['/status', '/logs', '/monitoring', '/tracing'].includes(location.pathname)

  useEffect(() => {
    // Always use dark theme
    document.documentElement.setAttribute('data-theme', 'dark')
  }, [])

  // Close dropdowns when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as HTMLElement
      if (!target.closest(`.${styles.configDropdown}`)) {
        setConfigDropdownOpen(false)
      }
      if (!target.closest(`.${styles.observabilityDropdown}`)) {
        setObservabilityDropdownOpen(false)
      }
    }
    document.addEventListener('click', handleClickOutside)
    return () => document.removeEventListener('click', handleClickOutside)
  }, [])

  // Config sections for dropdown - no emojis, clean text like NVIDIA Brev
  const configSections = [
    { id: 'models', title: 'Models' },
    { id: 'prompt-guard', title: 'Prompt Guard' },
    { id: 'similarity-cache', title: 'Similarity Cache' },
    { id: 'intelligent-routing', title: 'Intelligent Routing' },
    { id: 'topology', title: 'Topology' },
    { id: 'tools-selection', title: 'Tools Selection' },
    { id: 'observability', title: 'Observability' },
    { id: 'classification-api', title: 'Classification API' }
  ]

  return (
    <div className={styles.container}>
      {/* Top Navigation Bar */}
      <header className={styles.header}>
        <div className={styles.headerContent}>
          {/* Left: Brand */}
          <NavLink to="/" className={styles.brand}>
            <img src="/vllm.png" alt="vLLM" className={styles.logo} />
            <span className={styles.brandText}>vLLM Semantic Router</span>
          </NavLink>

          {/* Center: Navigation - Clean text like NVIDIA Brev */}
          <nav className={styles.nav}>
            <NavLink
              to="/playground"
              className={({ isActive }) =>
                isActive ? `${styles.navLink} ${styles.navLinkActive}` : styles.navLink
              }
            >
              Playground
            </NavLink>

            {/* Config Dropdown */}
            <div className={styles.configDropdown}>
              <button
                className={`${styles.navLink} ${isConfigPage || location.pathname === '/topology' ? styles.navLinkActive : ''}`}
                onClick={(e) => {
                  e.stopPropagation()
                  setConfigDropdownOpen(!configDropdownOpen)
                }}
              >
                Config
                <svg className={`${styles.dropdownArrow} ${configDropdownOpen ? styles.dropdownArrowOpen : ''}`} width="10" height="10" viewBox="0 0 12 12" fill="none">
                  <path d="M3 4.5L6 7.5L9 4.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </button>

              {configDropdownOpen && (
                <div className={styles.dropdownMenu}>
                  {configSections.map((section) => (
                    <button
                      key={section.id}
                      className={`${styles.dropdownItem} ${
                        (section.id === 'topology' && location.pathname === '/topology') ||
                        (isConfigPage && configSection === section.id)
                          ? styles.dropdownItemActive
                          : ''
                      }`}
                      onClick={() => {
                        if (section.id === 'topology') {
                          navigate('/topology')
                        } else {
                          onConfigSectionChange?.(section.id)
                          navigate('/config')
                        }
                        setConfigDropdownOpen(false)
                      }}
                    >
                      {section.title}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Observability Dropdown */}
            <div className={styles.observabilityDropdown}>
              <button
                className={`${styles.navLink} ${styles.dropdownTrigger} ${isObservabilityPage ? styles.navLinkActive : ''}`}
                onClick={(e) => {
                  e.stopPropagation()
                  setObservabilityDropdownOpen(!observabilityDropdownOpen)
                  setConfigDropdownOpen(false)
                }}
              >
                Observability
                <svg
                  width="12"
                  height="12"
                  viewBox="0 0 12 12"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  className={`${styles.dropdownArrow} ${observabilityDropdownOpen ? styles.dropdownArrowOpen : ''}`}
                >
                  <path d="M3 4.5L6 7.5L9 4.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </button>
              {observabilityDropdownOpen && (
                <div className={styles.dropdownMenu}>
                  <NavLink
                    to="/status"
                    className={`${styles.dropdownItem} ${location.pathname === '/status' ? styles.dropdownItemActive : ''}`}
                    onClick={() => setObservabilityDropdownOpen(false)}
                  >
                    Status
                  </NavLink>
                  <NavLink
                    to="/logs"
                    className={`${styles.dropdownItem} ${location.pathname === '/logs' ? styles.dropdownItemActive : ''}`}
                    onClick={() => setObservabilityDropdownOpen(false)}
                  >
                    Logs
                  </NavLink>
                  <NavLink
                    to="/monitoring"
                    className={`${styles.dropdownItem} ${location.pathname === '/monitoring' ? styles.dropdownItemActive : ''}`}
                    onClick={() => setObservabilityDropdownOpen(false)}
                  >
                    Grafana
                  </NavLink>
                  <NavLink
                    to="/tracing"
                    className={`${styles.dropdownItem} ${location.pathname === '/tracing' ? styles.dropdownItemActive : ''}`}
                    onClick={() => setObservabilityDropdownOpen(false)}
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
            <NavLink to="/playground" className={styles.mobileNavLink} onClick={() => setMobileMenuOpen(false)}>
              Playground
            </NavLink>
            <NavLink to="/config" className={styles.mobileNavLink} onClick={() => setMobileMenuOpen(false)}>
              Config
            </NavLink>
            <NavLink to="/monitoring" className={styles.mobileNavLink} onClick={() => setMobileMenuOpen(false)}>
              Monitoring
            </NavLink>
            <NavLink to="/tracing" className={styles.mobileNavLink} onClick={() => setMobileMenuOpen(false)}>
              Tracing
            </NavLink>
            <NavLink to="/status" className={styles.mobileNavLink} onClick={() => setMobileMenuOpen(false)}>
              Status
            </NavLink>
            <NavLink to="/logs" className={styles.mobileNavLink} onClick={() => setMobileMenuOpen(false)}>
              Logs
            </NavLink>
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
