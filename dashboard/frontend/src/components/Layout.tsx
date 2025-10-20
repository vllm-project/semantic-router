import React, { useState, useEffect, ReactNode } from 'react'
import { NavLink, useLocation, useNavigate } from 'react-router-dom'
import styles from './Layout.module.css'

interface LayoutProps {
  children: ReactNode
  configSection?: string
  onConfigSectionChange?: (section: string) => void
}

const Layout: React.FC<LayoutProps> = ({ children, configSection, onConfigSectionChange }) => {
  const [theme, setTheme] = useState<'light' | 'dark'>('dark')
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const location = useLocation()
  const navigate = useNavigate()
  const isConfigPage = location.pathname === '/config'

  useEffect(() => {
    // Check stored preference, default to light theme
    const stored = localStorage.getItem('theme') as 'light' | 'dark' | null
    const initialTheme = stored || 'light' // Default to light theme
    setTheme(initialTheme)
    document.documentElement.setAttribute('data-theme', initialTheme)
  }, [])

  useEffect(() => {
    // Check stored sidebar state
    const storedCollapsed = localStorage.getItem('sidebarCollapsed') === 'true'
    setSidebarCollapsed(storedCollapsed)
  }, [])

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light'
    setTheme(newTheme)
    localStorage.setItem('theme', newTheme)
    document.documentElement.setAttribute('data-theme', newTheme)
  }

  const toggleSidebar = () => {
    const newCollapsed = !sidebarCollapsed
    setSidebarCollapsed(newCollapsed)
    localStorage.setItem('sidebarCollapsed', String(newCollapsed))
  }

  return (
    <div className={styles.container}>
      <aside className={`${styles.sidebar} ${sidebarCollapsed ? styles.sidebarCollapsed : ''}`}>
        <div className={styles.brandContainer}>
          <NavLink to="/" className={styles.brand}>
            <img src="/vllm.png" alt="vLLM" className={styles.logo} />
            {!sidebarCollapsed && <span className={styles.brandText}>Semantic Router</span>}
          </NavLink>
        </div>
        <nav className={styles.nav}>
          <NavLink
            to="/huggingchat"
            className={({ isActive }) =>
              isActive ? `${styles.navLink} ${styles.navLinkActive}` : styles.navLink
            }
            title="HuggingChat"
          >
            <span className={styles.navIcon}>🤗</span>
            {!sidebarCollapsed && <span className={styles.navText}>HuggingChat</span>}
          </NavLink>

          <NavLink
            to="/playground"
            className={({ isActive }) =>
              isActive ? `${styles.navLink} ${styles.navLinkActive}` : styles.navLink
            }
            title="Playground"
          >
            <span className={styles.navIcon}>🎮</span>
            {!sidebarCollapsed && <span className={styles.navText}>Playground</span>}
          </NavLink>

          {/* Configuration sections - Same level as other nav items */}
          {onConfigSectionChange && (
            <>
              {[
                { id: 'models', icon: '🤖', title: 'Models' },
                { id: 'prompt-guard', icon: '🛡️', title: 'Prompt Guard' },
                { id: 'similarity-cache', icon: '⚡', title: 'Similarity Cache' },
                { id: 'intelligent-routing', icon: '🧠', title: 'Intelligent Routing' },
                { id: 'topology', icon: '🗺️', title: 'Topology' },
                { id: 'tools-selection', icon: '🔧', title: 'Tools Selection' },
                { id: 'observability', icon: '👁️', title: 'Observability' },
                { id: 'classification-api', icon: '🔌', title: 'Classification API' }
              ].map((section) => (
                <button
                  key={section.id}
                  className={`${styles.navLink} ${(section.id === 'topology' && location.pathname === '/topology') ||
                      (isConfigPage && configSection === section.id)
                      ? styles.navLinkActive
                      : ''
                    }`}
                  onClick={() => {
                    if (section.id === 'topology') {
                      navigate('/topology')
                    } else {
                      onConfigSectionChange(section.id)
                      navigate('/config')
                    }
                  }}
                  title={section.title}
                >
                  <span className={styles.navIcon}>{section.icon}</span>
                  {!sidebarCollapsed && <span className={styles.navText}>{section.title}</span>}
                </button>
              ))}
            </>
          )}

          <NavLink
            to="/monitoring"
            className={({ isActive }) =>
              isActive ? `${styles.navLink} ${styles.navLinkActive}` : styles.navLink
            }
            title="Monitoring"
          >
            <span className={styles.navIcon}>📊</span>
            {!sidebarCollapsed && <span className={styles.navText}>Monitoring</span>}
          </NavLink>

          <NavLink
            to="/tracing"
            className={({ isActive }) =>
              isActive ? `${styles.navLink} ${styles.navLinkActive}` : styles.navLink
            }
            title="Tracing"
          >
            <span className={styles.navIcon}>🔎</span>
            {!sidebarCollapsed && <span className={styles.navText}>Tracing</span>}
          </NavLink>
        </nav>
        <div className={styles.sidebarFooter}>
          <button
            className={styles.collapseButton}
            onClick={toggleSidebar}
            aria-label={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {sidebarCollapsed ? (
              // Collapsed state: arrow pointing right
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M14 10L18 10M18 10L16 8M18 10L16 12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                <path d="M2 5H10M2 10H10M2 15H10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
              </svg>
            ) : (
              // Expanded state: arrow pointing left
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M6 10L2 10M2 10L4 8M2 10L4 12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                <path d="M10 5H18M10 10H18M10 15H18" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
              </svg>
            )}
          </button>
        </div>
      </aside>
      <main className={styles.main}>
        <header className={styles.header}>
          <div className={styles.headerContent}>
            <div className={styles.headerLeft}>
              <span className={styles.headerBrand}>Intelligent Router for Mixture-of-Models 🧠</span>
            </div>
            <div className={styles.headerRight}>
              <button
                className={styles.headerIconButton}
                onClick={toggleTheme}
                aria-label="Toggle theme"
                title="Toggle theme"
              >
                {theme === 'light' ? '🌙' : '☀️'}
              </button>
              <a
                href="https://github.com/vllm-project/semantic-router"
                target="_blank"
                rel="noopener noreferrer"
                className={styles.headerIconButton}
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
                className={styles.headerIconButton}
                aria-label="Documentation"
                title="Documentation"
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
                  <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
                </svg>
              </a>
            </div>
          </div>
        </header>
        <div className={styles.mainContent}>{children}</div>
      </main>
    </div>
  )
}

export default Layout
