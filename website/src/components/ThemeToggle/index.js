import React from 'react'
import { useColorMode } from '@docusaurus/theme-common'
import styles from './index.module.css'

export default function ThemeToggle() {
  const { colorMode, setColorMode } = useColorMode()

  const toggleTheme = () => {
    setColorMode(colorMode === 'dark' ? 'light' : 'dark')
  }

  return (
    <div className={styles.themeToggle}>
      <button
        className={`${styles.toggleButton} ${colorMode === 'dark' ? styles.dark : styles.light}`}
        onClick={toggleTheme}
        aria-label={`Switch to ${colorMode === 'dark' ? 'light' : 'dark'} theme`}
        title={`Switch to ${colorMode === 'dark' ? 'light' : 'dark'} theme`}
      >
        <div className={styles.toggleSlider}>
          <div className={`${styles.toggleIcon} ${styles.sunIcon}`}>
            ☀️
          </div>
          <div className={`${styles.toggleIcon} ${styles.moonIcon}`}>
            🌙
          </div>
        </div>
        <div className={`${styles.toggleThumb} ${colorMode === 'dark' ? styles.thumbDark : styles.thumbLight}`}>
          <span className={styles.thumbIcon}>
            {colorMode === 'dark' ? '🌙' : '☀️'}
          </span>
        </div>
      </button>
      <span className={styles.themeLabel}>
        {colorMode === 'dark' ? 'Dark' : 'Light'}
        {' '}
        Mode
      </span>
    </div>
  )
}
