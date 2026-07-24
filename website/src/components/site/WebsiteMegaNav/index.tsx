import type {
  FocusEvent,
  KeyboardEvent,
  MouseEvent,
  ReactNode,
} from 'react'
import Link from '@docusaurus/Link'
import { useLocation } from '@docusaurus/router'
import clsx from 'clsx'
import React from 'react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  isWebsiteMegaNavGroupActive,
  WEBSITE_MEGA_NAV_GROUPS,
} from './navigation'
import type {
  WebsiteMegaNavGroup,
  WebsiteMegaNavKey,
  WebsiteMegaNavLink,
} from './navigation'
import styles from './index.module.css'

function NavIcon({ linkKey }: { linkKey: string }): ReactNode {
  const icons: Record<string, ReactNode> = {
    'quick-start': (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" /></svg>
    ),
    'installation': (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
        <polyline points="7 10 12 15 17 10" />
        <line x1="12" y1="15" x2="12" y2="3" />
      </svg>
    ),
    'configuration': (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="3" />
        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
      </svg>
    ),
    'tutorials': (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
        <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
      </svg>
    ),
    'api': (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="16 18 22 12 16 6" />
        <polyline points="8 6 2 12 8 18" />
      </svg>
    ),
    'troubleshooting': (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" /></svg>
    ),
    'publications': (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
        <polyline points="14 2 14 8 20 8" />
        <line x1="16" y1="13" x2="8" y2="13" />
        <line x1="16" y1="17" x2="8" y2="17" />
        <polyline points="10 9 9 9 8 9" />
      </svg>
    ),
    'white-paper': (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
        <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
      </svg>
    ),
    'vision-paper': (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10" />
        <line x1="2" y1="12" x2="22" y2="12" />
        <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
      </svg>
    ),
    'blog': (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 20h9" />
        <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" />
      </svg>
    ),
    'team': (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
        <circle cx="9" cy="7" r="4" />
        <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
        <path d="M16 3.13a4 4 0 0 1 0 7.75" />
      </svg>
    ),
    'steering': (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="3" width="7" height="7" />
        <rect x="14" y="3" width="7" height="7" />
        <rect x="14" y="14" width="7" height="7" />
        <rect x="3" y="14" width="7" height="7" />
      </svg>
    ),
    'governance': (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" /></svg>
    ),
    'working-groups': (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10" />
        <path d="M8 14s1.5 2 4 2 4-2 4-2" />
        <line x1="9" y1="9" x2="9.01" y2="9" />
        <line x1="15" y1="9" x2="15.01" y2="9" />
      </svg>
    ),
    'contributing': (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <line x1="12" y1="5" x2="12" y2="19" />
        <polyline points="19 12 12 19 5 12" />
      </svg>
    ),
    'conduct': (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="16" x2="12" y2="12" />
        <line x1="12" y1="8" x2="12.01" y2="8" />
      </svg>
    ),
    'leaderboard': (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12" /></svg>
    ),
    'github': (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22" /></svg>
    ),
    'models': (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <rect x="4" y="4" width="16" height="16" rx="2" ry="2" />
        <rect x="9" y="9" width="6" height="6" />
        <line x1="9" y1="1" x2="9" y2="4" />
        <line x1="15" y1="1" x2="15" y2="4" />
        <line x1="9" y1="20" x2="9" y2="23" />
        <line x1="15" y1="20" x2="15" y2="23" />
        <line x1="20" y1="9" x2="23" y2="9" />
        <line x1="20" y1="14" x2="23" y2="14" />
        <line x1="1" y1="9" x2="4" y2="9" />
        <line x1="1" y1="14" x2="4" y2="14" />
      </svg>
    ),
  }

  return (
    <span className={styles.cardIcon} aria-hidden="true">
      {icons[linkKey] ?? icons.api}
    </span>
  )
}

function NavCard({
  link,
  onNavigate,
}: {
  link: WebsiteMegaNavLink
  onNavigate: () => void
}): ReactNode {
  const isExternal = Boolean(link.href)

  const body = (
    <>
      <NavIcon linkKey={link.key} />
      <span className={styles.cardBody}>
        <strong>{link.label}</strong>
        <span>{link.description}</span>
      </span>
    </>
  )

  if (isExternal) {
    return (
      <a
        className={styles.navCard}
        href={link.href}
        target="_blank"
        rel="noreferrer"
        data-mega-nav-link
        onClick={onNavigate}
      >
        {body}
      </a>
    )
  }

  return (
    <Link
      className={styles.navCard}
      to={link.to ?? '/'}
      data-mega-nav-link
      onClick={onNavigate}
    >
      {body}
    </Link>
  )
}

function getRouteActiveGroup(pathname: string): WebsiteMegaNavGroup | undefined {
  return (
    WEBSITE_MEGA_NAV_GROUPS.find(
      group =>
        group.key !== 'docs'
        && isWebsiteMegaNavGroupActive(group, pathname),
    )
    ?? WEBSITE_MEGA_NAV_GROUPS.find(group =>
      isWebsiteMegaNavGroupActive(group, pathname),
    )
  )
}

export default function WebsiteMegaNav(): ReactNode {
  const { pathname } = useLocation()
  const [openKey, setOpenKey] = useState<WebsiteMegaNavKey | null>(null)
  const manualOpenRef = useRef(false)
  const rootRef = useRef<HTMLDivElement>(null)
  const dropdownRef = useRef<HTMLDivElement>(null)
  const triggerRefs = useRef<
    Partial<Record<WebsiteMegaNavKey, HTMLAnchorElement | null>>
  >({})
  const closeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const activeGroup = useMemo(() => getRouteActiveGroup(pathname), [pathname])
  const openGroup = useMemo(
    () => WEBSITE_MEGA_NAV_GROUPS.find(group => group.key === openKey),
    [openKey],
  )

  const cancelClose = useCallback(() => {
    if (closeTimerRef.current) {
      clearTimeout(closeTimerRef.current)
      closeTimerRef.current = null
    }
  }, [])

  const closeMenu = useCallback(() => {
    cancelClose()
    manualOpenRef.current = false
    setOpenKey(null)
  }, [cancelClose])

  const scheduleClose = useCallback(() => {
    if (manualOpenRef.current) {
      return
    }
    cancelClose()
    closeTimerRef.current = setTimeout(closeMenu, 120)
  }, [cancelClose, closeMenu])

  const focusTrigger = useCallback((key: WebsiteMegaNavKey) => {
    triggerRefs.current[key]?.focus()
  }, [])

  useEffect(() => {
    closeMenu()
  }, [closeMenu, pathname])

  useEffect(() => {
    if (!openKey) {
      return undefined
    }

    const handleEscape = (event: globalThis.KeyboardEvent) => {
      if (event.key !== 'Escape') {
        return
      }

      event.preventDefault()
      const keyToFocus = openKey
      closeMenu()
      focusTrigger(keyToFocus)
    }

    const handlePointerDown = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) {
        closeMenu()
      }
    }

    document.addEventListener('keydown', handleEscape)
    document.addEventListener('pointerdown', handlePointerDown)
    return () => {
      document.removeEventListener('keydown', handleEscape)
      document.removeEventListener('pointerdown', handlePointerDown)
    }
  }, [closeMenu, focusTrigger, openKey])

  useEffect(() => cancelClose, [cancelClose])

  const handleRootBlur = (event: FocusEvent<HTMLDivElement>) => {
    const nextTarget = event.relatedTarget

    if (!nextTarget || event.currentTarget.contains(nextTarget)) {
      return
    }

    closeMenu()
  }

  const handleTriggerEnter = (key: WebsiteMegaNavKey) => {
    const group = WEBSITE_MEGA_NAV_GROUPS.find(g => g.key === key)
    if (!group || group.sections.length === 0) {
      return
    }
    cancelClose()
    setOpenKey(key)
  }

  const handleTriggerLeave = () => {
    scheduleClose()
  }

  const handleDropdownEnter = () => {
    cancelClose()
  }

  const handleDropdownLeave = () => {
    scheduleClose()
  }

  const handleChevronClick = (
    event: MouseEvent<HTMLButtonElement>,
    key: WebsiteMegaNavKey,
  ) => {
    event.preventDefault()
    event.stopPropagation()

    if (openKey === key) {
      closeMenu()
    }
    else {
      manualOpenRef.current = true
      setOpenKey(key)
    }
  }

  const handleLabelKeyDown = (
    event: KeyboardEvent<HTMLAnchorElement>,
    key: WebsiteMegaNavKey,
  ) => {
    if (event.key === 'ArrowDown') {
      event.preventDefault()
      setOpenKey(key)
      requestAnimationFrame(() => {
        dropdownRef.current
          ?.querySelector<HTMLElement>('[data-mega-nav-link]')
          ?.focus()
      })
      return
    }

    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      event.preventDefault()
      const currentIndex = WEBSITE_MEGA_NAV_GROUPS.findIndex(
        group => group.key === key,
      )
      const direction = event.key === 'ArrowLeft' ? -1 : 1
      const nextIndex
        = (currentIndex + direction + WEBSITE_MEGA_NAV_GROUPS.length)
          % WEBSITE_MEGA_NAV_GROUPS.length
      const nextGroup = WEBSITE_MEGA_NAV_GROUPS[nextIndex]
      focusTrigger(nextGroup.key)
      setOpenKey(nextGroup.sections.length > 0 ? nextGroup.key : null)
      return
    }

    if (event.key === 'Home' || event.key === 'End') {
      event.preventDefault()
      const nextGroup
        = event.key === 'Home'
          ? WEBSITE_MEGA_NAV_GROUPS[0]
          : WEBSITE_MEGA_NAV_GROUPS[WEBSITE_MEGA_NAV_GROUPS.length - 1]
      focusTrigger(nextGroup.key)
      setOpenKey(nextGroup.sections.length > 0 ? nextGroup.key : null)
    }
  }

  const handleChevronKeyDown = (
    event: KeyboardEvent<HTMLButtonElement>,
    key: WebsiteMegaNavKey,
  ) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      event.stopPropagation()
      if (openKey === key) {
        closeMenu()
      }
      else {
        manualOpenRef.current = true
        setOpenKey(key)
      }
    }
  }

  return (
    <div
      className={styles.root}
      ref={rootRef}
      onBlur={handleRootBlur}
      data-mega-nav-open={openKey ? 'true' : 'false'}
    >
      <nav className={styles.triggers} aria-label="Primary navigation">
        {WEBSITE_MEGA_NAV_GROUPS.map((group) => {
          const isOpen = group.key === openKey
          const isActive = group.key === activeGroup?.key
          const hasDropdown = group.sections.length > 0

          if (!hasDropdown) {
            return (
              <Link
                key={group.key}
                ref={(element: HTMLAnchorElement | null) => {
                  triggerRefs.current[group.key] = element
                }}
                to={group.landingTo}
                id={`website-mega-nav-trigger-${group.key}`}
                className={clsx(styles.trigger, styles.triggerPlain, {
                  [styles.triggerActive]: isActive,
                })}
              >
                {group.label}
              </Link>
            )
          }

          return (
            <div
              key={group.key}
              className={clsx(styles.trigger, {
                [styles.triggerOpen]: isOpen,
                [styles.triggerActive]: isActive,
              })}
              onMouseEnter={() => handleTriggerEnter(group.key)}
              onMouseLeave={handleTriggerLeave}
            >
              <Link
                ref={(element: HTMLAnchorElement | null) => {
                  triggerRefs.current[group.key] = element
                }}
                to={group.landingTo}
                id={`website-mega-nav-trigger-${group.key}`}
                className={styles.triggerLabel}
                onFocus={() => {
                  cancelClose()
                  setOpenKey(group.key)
                }}
                onKeyDown={event => handleLabelKeyDown(event, group.key)}
              >
                {group.label}
              </Link>
              <button
                type="button"
                className={clsx(styles.triggerChevron, {
                  [styles.triggerChevronOpen]: isOpen,
                })}
                tabIndex={-1}
                aria-expanded={isOpen}
                aria-controls={`website-mega-nav-panel-${group.key}`}
                aria-label={`${isOpen ? 'Close' : 'Open'} ${group.label} menu`}
                onClick={event => handleChevronClick(event, group.key)}
                onKeyDown={event => handleChevronKeyDown(event, group.key)}
              >
                <span className={styles.triggerCaret} aria-hidden="true" />
              </button>
            </div>
          )
        })}
      </nav>

      {openGroup && openGroup.sections.length > 0 && (
        <div
          className={styles.dropdown}
          id={`website-mega-nav-panel-${openGroup.key}`}
          ref={dropdownRef}
          role="region"
          aria-labelledby={`website-mega-nav-trigger-${openGroup.key}`}
          onMouseEnter={handleDropdownEnter}
          onMouseLeave={handleDropdownLeave}
        >
          {openGroup.sections.map(section => (
            <div className={styles.section} key={section.key}>
              <div className={styles.sectionLabel}>{section.title}</div>
              <div className={styles.sectionGrid}>
                {section.links.map(link => (
                  <NavCard
                    key={link.key}
                    link={link}
                    onNavigate={closeMenu}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
