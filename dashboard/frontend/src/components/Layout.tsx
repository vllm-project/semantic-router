import React, { useState, useEffect, ReactNode } from 'react'
import { NavLink } from 'react-router-dom'
import styles from './Layout.module.css'

interface LayoutProps {
  children: ReactNode
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const [theme, setTheme] = useState<'light' | 'dark'>('dark')

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
      <header className={styles.header}>
        <div className={styles.headerLeft}>
          <img src="/vllm.png" alt="vLLM" className={styles.logo} />
          <h1 className={styles.title}>Semantic Router Dashboard</h1>
        </div>
        <nav className={styles.nav}>
          <NavLink
            to="/monitoring"
            className={({ isActive }) =>
              isActive ? `${styles.navLink} ${styles.navLinkActive}` : styles.navLink
            }
          >
            <span className={styles.navIcon}>📊</span>
            Monitoring
          </NavLink>
          <NavLink
            to="/config"
            className={({ isActive }) =>
              isActive ? `${styles.navLink} ${styles.navLinkActive}` : styles.navLink
            }
          >
            <span className={styles.navIcon}>⚙️</span>
            Config
          </NavLink>
          <NavLink
            to="/playground"
            className={({ isActive }) =>
              isActive ? `${styles.navLink} ${styles.navLinkActive}` : styles.navLink
            }
          >
            <span className={styles.navIcon}>🎮</span>
            Playground
          </NavLink>
        </nav>
        <button
          className={styles.themeToggle}
          onClick={toggleTheme}
          aria-label="Toggle theme"
        >
          {theme === 'light' ? '🌙' : '☀️'}
        </button>
      </header>
      <main className={styles.main}>{children}</main>
    </div>
  )
}

export default Layout
