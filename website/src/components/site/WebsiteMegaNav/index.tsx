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

function MegaNavLink({
  link,
  onNavigate,
}: {
  link: WebsiteMegaNavLink
  onNavigate: () => void
}): ReactNode {
  const content = (
    <>
      <span className={styles.linkCopy}>
        <strong>{link.label}</strong>
        <span>{link.description}</span>
      </span>
      <span className={styles.linkArrow} aria-hidden="true">
        {link.href ? '↗' : '→'}
      </span>
    </>
  )

  if (link.href) {
    return (
      <a
        className={styles.link}
        href={link.href}
        target="_blank"
        rel="noreferrer"
        data-mega-nav-link
        onClick={onNavigate}
      >
        {content}
      </a>
    )
  }

  return (
    <Link
      className={styles.link}
      to={link.to ?? '/'}
      data-mega-nav-link
      onClick={onNavigate}
    >
      {content}
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
  const rootRef = useRef<HTMLDivElement>(null)
  const panelRef = useRef<HTMLDivElement>(null)
  const triggerRefs = useRef<
    Partial<Record<WebsiteMegaNavKey, HTMLButtonElement | null>>
  >({})
  const focusFirstLinkRef = useRef(false)

  const activeGroup = useMemo(() => getRouteActiveGroup(pathname), [pathname])
  const openGroup = useMemo(
    () => WEBSITE_MEGA_NAV_GROUPS.find(group => group.key === openKey),
    [openKey],
  )

  const closeMenu = useCallback(() => {
    focusFirstLinkRef.current = false
    setOpenKey(null)
  }, [])

  const focusTrigger = useCallback((key: WebsiteMegaNavKey) => {
    triggerRefs.current[key]?.focus()
  }, [])

  useEffect(() => {
    closeMenu()
  }, [closeMenu, pathname])

  useEffect(() => {
    if (!openKey || !focusFirstLinkRef.current) {
      return
    }

    focusFirstLinkRef.current = false
    panelRef.current
      ?.querySelector<HTMLElement>('[data-mega-nav-link]')
      ?.focus()
  }, [openKey])

  useEffect(() => {
    if (!openKey) {
      return undefined
    }

    const handlePointerDown = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) {
        closeMenu()
      }
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

    document.addEventListener('pointerdown', handlePointerDown)
    document.addEventListener('keydown', handleEscape)
    return () => {
      document.removeEventListener('pointerdown', handlePointerDown)
      document.removeEventListener('keydown', handleEscape)
    }
  }, [closeMenu, focusTrigger, openKey])

  const handleRootBlur = (event: FocusEvent<HTMLDivElement>) => {
    const nextTarget = event.relatedTarget

    // Safari can report a null relatedTarget while focus is moving inside a
    // fixed-position panel. Pointer and Escape handlers still close the menu.
    if (!nextTarget || event.currentTarget.contains(nextTarget)) {
      return
    }

    closeMenu()
  }

  const handleTriggerClick = (
    event: MouseEvent<HTMLButtonElement>,
    key: WebsiteMegaNavKey,
  ) => {
    event.preventDefault()
    focusFirstLinkRef.current = false
    setOpenKey(current => (current === key ? null : key))
  }

  const moveTriggerFocus = (
    currentKey: WebsiteMegaNavKey,
    direction: -1 | 1,
  ) => {
    const currentIndex = WEBSITE_MEGA_NAV_GROUPS.findIndex(
      group => group.key === currentKey,
    )
    const nextIndex
      = (currentIndex + direction + WEBSITE_MEGA_NAV_GROUPS.length)
        % WEBSITE_MEGA_NAV_GROUPS.length
    const nextKey = WEBSITE_MEGA_NAV_GROUPS[nextIndex].key

    focusTrigger(nextKey)
    if (openKey) {
      setOpenKey(nextKey)
    }
  }

  const handleTriggerKeyDown = (
    event: KeyboardEvent<HTMLButtonElement>,
    key: WebsiteMegaNavKey,
  ) => {
    if (event.key === 'ArrowDown') {
      event.preventDefault()
      focusFirstLinkRef.current = true
      setOpenKey(key)
      return
    }

    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      event.preventDefault()
      moveTriggerFocus(key, event.key === 'ArrowLeft' ? -1 : 1)
      return
    }

    if (event.key === 'Home' || event.key === 'End') {
      event.preventDefault()
      const nextKey
        = event.key === 'Home'
          ? WEBSITE_MEGA_NAV_GROUPS[0].key
          : WEBSITE_MEGA_NAV_GROUPS[WEBSITE_MEGA_NAV_GROUPS.length - 1].key
      focusTrigger(nextKey)
      if (openKey) {
        setOpenKey(nextKey)
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

          return (
            <button
              key={group.key}
              ref={(element) => {
                triggerRefs.current[group.key] = element
              }}
              id={`website-mega-nav-trigger-${group.key}`}
              type="button"
              className={clsx(styles.trigger, {
                [styles.triggerOpen]: isOpen,
                [styles.triggerActive]: isActive,
              })}
              aria-expanded={isOpen}
              aria-controls={`website-mega-nav-panel-${group.key}`}
              onClick={event => handleTriggerClick(event, group.key)}
              onKeyDown={event => handleTriggerKeyDown(event, group.key)}
            >
              {group.label}
              <span className={styles.triggerMark} aria-hidden="true">
                {isOpen ? '−' : '+'}
              </span>
            </button>
          )
        })}
      </nav>

      {openGroup && (
        <>
          <button
            className={styles.scrim}
            type="button"
            tabIndex={-1}
            aria-label="Close navigation menu"
            onClick={closeMenu}
          />
          <div
            className={styles.panel}
            id={`website-mega-nav-panel-${openGroup.key}`}
            ref={panelRef}
            role="region"
            aria-labelledby={`website-mega-nav-trigger-${openGroup.key}`}
          >
            <aside className={styles.rail}>
              <div className={styles.railHeading}>
                <span className={styles.eyebrow}>Explore</span>
                <strong>{openGroup.label}</strong>
                <p>{openGroup.description}</p>
              </div>

              <div className={styles.railGroups} aria-label="Navigation groups">
                {WEBSITE_MEGA_NAV_GROUPS.map((group, index) => (
                  <button
                    key={group.key}
                    type="button"
                    className={clsx(styles.railGroup, {
                      [styles.railGroupActive]: group.key === openGroup.key,
                    })}
                    aria-pressed={group.key === openGroup.key}
                    onClick={() => setOpenKey(group.key)}
                  >
                    <span>{String(index + 1).padStart(2, '0')}</span>
                    <strong>{group.label}</strong>
                    <span aria-hidden="true">→</span>
                  </button>
                ))}
              </div>

              <Link
                className={styles.landingLink}
                to={openGroup.landingTo}
                onClick={closeMenu}
              >
                Explore all
                {' '}
                {openGroup.label}
                <span aria-hidden="true">↗</span>
              </Link>
            </aside>

            <div className={styles.content}>
              <div className={styles.contentHeader}>
                <span>
                  {openGroup.label}
                  {' '}
                  directory
                </span>
                <button
                  className={styles.closeButton}
                  type="button"
                  onClick={() => {
                    const keyToFocus = openGroup.key
                    closeMenu()
                    focusTrigger(keyToFocus)
                  }}
                  aria-label="Close navigation menu"
                >
                  Close
                  <span aria-hidden="true">×</span>
                </button>
              </div>

              <div
                className={styles.sections}
                data-section-count={openGroup.sections.length}
              >
                {openGroup.sections.map((section, index) => (
                  <section className={styles.section} key={section.key}>
                    <header className={styles.sectionHeader}>
                      <span>{String(index + 1).padStart(2, '0')}</span>
                      <h2>{section.title}</h2>
                      <p>{section.description}</p>
                    </header>
                    <div className={styles.sectionLinks}>
                      {section.links.map(link => (
                        <MegaNavLink
                          key={link.key}
                          link={link}
                          onNavigate={closeMenu}
                        />
                      ))}
                    </div>
                  </section>
                ))}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
