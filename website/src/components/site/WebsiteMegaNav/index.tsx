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
  WebsiteMegaNavSection,
} from './navigation'
import styles from './index.module.css'

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

function getFeaturedLink(group: WebsiteMegaNavGroup): WebsiteMegaNavLink | undefined {
  return group.sections[0]?.links[0]
}

function getSupportingSections(group: WebsiteMegaNavGroup): WebsiteMegaNavSection[] {
  const featuredKey = getFeaturedLink(group)?.key

  return group.sections
    .map(section => ({
      ...section,
      links: section.links.filter(link => link.key !== featuredKey),
    }))
    .filter(section => section.links.length > 0)
}

function DestinationCard({
  link,
  onNavigate,
  featured = false,
}: {
  link: WebsiteMegaNavLink
  onNavigate: () => void
  featured?: boolean
}): ReactNode {
  const className = clsx(styles.card, featured && styles.featuredCard)
  const arrow = link.href ? '↗' : '→'

  const body = (
    <>
      {featured && <span className={styles.featuredLabel}>Featured</span>}
      <span className={styles.cardCopy}>
        <strong>{link.label}</strong>
        <span>{link.description}</span>
      </span>
      <span className={styles.cardCta}>
        {featured ? `Open ${link.label}` : 'Open'}
        <span aria-hidden="true">{arrow}</span>
      </span>
    </>
  )

  if (link.href) {
    return (
      <a
        className={className}
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
      className={className}
      to={link.to ?? '/'}
      data-mega-nav-link
      onClick={onNavigate}
    >
      {body}
    </Link>
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
  const featuredLink = useMemo(
    () => (openGroup ? getFeaturedLink(openGroup) : undefined),
    [openGroup],
  )
  const supportingSections = useMemo(
    () => (openGroup ? getSupportingSections(openGroup) : []),
    [openGroup],
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
            <header className={styles.panelHeader}>
              <div className={styles.panelHeading}>
                <h2 className={styles.panelTitle}>{openGroup.label}</h2>
                <p className={styles.panelDescription}>{openGroup.description}</p>
              </div>
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
            </header>

            <div className={styles.panelBody}>
              {featuredLink && (
                <DestinationCard
                  link={featuredLink}
                  featured
                  onNavigate={closeMenu}
                />
              )}

              <div className={styles.sections}>
                {supportingSections.map(section => (
                  <section className={styles.section} key={section.key}>
                    <header className={styles.sectionHeader}>
                      <h3>{section.title}</h3>
                      <p>{section.description}</p>
                    </header>
                    <div className={styles.cardGrid}>
                      {section.links.map(link => (
                        <DestinationCard
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

            <footer className={styles.panelFooter}>
              <Link
                className={styles.exploreAll}
                to={openGroup.landingTo}
                onClick={closeMenu}
              >
                Explore all
                {' '}
                {openGroup.label}
                <span aria-hidden="true">↗</span>
              </Link>
            </footer>
          </div>
        </>
      )}
    </div>
  )
}
