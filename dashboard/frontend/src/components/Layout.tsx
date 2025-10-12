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
  const location = useLocation()
  const navigate = useNavigate()
  const isConfigPage = location.pathname === '/config'

  useEffect(() => {
    // Check system preference or stored preference
    const stored = localStorage.getItem('theme') as 'light' | 'dark' | null
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
    const initialTheme = stored || (prefersDark ? 'dark' : 'light')
    setTheme(initialTheme)
    document.documentElement.setAttribute('data-theme', initialTheme)
  }, [])

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light'
    setTheme(newTheme)
    localStorage.setItem('theme', newTheme)
    document.documentElement.setAttribute('data-theme', newTheme)
  }

  return (
    <div className={styles.container}>
      <aside className={styles.sidebar}>
        <NavLink to="/" className={styles.brand}>
          <img src="/vllm.png" alt="vLLM" className={styles.logo} />
          <span className={styles.brandText}>Semantic Router</span>
        </NavLink>
        <nav className={styles.nav}>
          <NavLink
            to="/playground"
            className={({ isActive }) =>
              isActive ? `${styles.navLink} ${styles.navLinkActive}` : styles.navLink
            }
          >
            <span className={styles.navIcon}>🎮</span>
            <span className={styles.navText}>Playground</span>
          </NavLink>

          {/* Configuration sections - Same level as other nav items */}
          {onConfigSectionChange && (
            <>
              {[
                { id: 'models', icon: '🤖', title: 'Models' },
                { id: 'prompt-guard', icon: '🛡️', title: 'Prompt Guard' },
                { id: 'similarity-cache', icon: '⚡', title: 'Similarity Cache' },
                { id: 'intelligent-routing', icon: '🧠', title: 'Intelligent Routing' },
                { id: 'tools-selection', icon: '🔧', title: 'Tools Selection' },
                { id: 'observability', icon: '👁️', title: 'Observability' },
                { id: 'classification-api', icon: '🔌', title: 'Classification API' }
              ].map((section) => (
                <button
                  key={section.id}
                  className={`${styles.navLink} ${isConfigPage && configSection === section.id ? styles.navLinkActive : ''}`}
                  onClick={() => {
                    onConfigSectionChange(section.id)
                    navigate('/config')
                  }}
                >
                  <span className={styles.navIcon}>{section.icon}</span>
                  <span className={styles.navText}>{section.title}</span>
                </button>
              ))}
            </>
          )}

          <NavLink
            to="/monitoring"
            className={({ isActive }) =>
              isActive ? `${styles.navLink} ${styles.navLinkActive}` : styles.navLink
            }
          >
            <span className={styles.navIcon}>📊</span>
            <span className={styles.navText}>Monitoring</span>
          </NavLink>
        </nav>
        <div className={styles.sidebarFooter}>
          <button
            className={styles.themeToggle}
            onClick={toggleTheme}
            aria-label="Toggle theme"
            title="Toggle theme"
          >
            {theme === 'light' ? '🌙' : '☀️'}
          </button>
          <a
            href="https://github.com/vllm-project/vllm"
            target="_blank"
            rel="noopener noreferrer"
            className={styles.iconButton}
            aria-label="GitHub"
            title="GitHub Repository"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
            </svg>
          </a>
          <a
            href="https://docs.vllm.ai"
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
        </div>
      </aside>
      <main className={styles.main}>{children}</main>
    </div>
  )
}

export default Layout
