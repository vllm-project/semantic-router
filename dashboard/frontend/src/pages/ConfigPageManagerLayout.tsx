import React from 'react'
import styles from './ConfigPageManagerLayout.module.css'

interface ConfigPageManagerLayoutProps {
  eyebrow?: string
  title: string
  description: string
  children: React.ReactNode
}

export default function ConfigPageManagerLayout({
  eyebrow = 'Manager',
  title,
  description,
  children,
}: ConfigPageManagerLayoutProps) {
  return (
    <section className={styles.page}>
      <header className={styles.hero}>
        <div className={styles.copy}>
          <span className={styles.eyebrow}>{eyebrow}</span>
          <h1 className={styles.title}>{title}</h1>
          <p className={styles.description}>{description}</p>
        </div>
      </header>

      <div className={styles.body}>{children}</div>
    </section>
  )
}
