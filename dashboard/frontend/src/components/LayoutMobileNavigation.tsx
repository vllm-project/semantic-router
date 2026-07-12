import { useEffect, useRef, type KeyboardEvent } from 'react'
import { NavLink } from 'react-router-dom'
import styles from './Layout.module.css'
import {
  isLayoutMenuItemActive,
  PRIMARY_NAV_LINKS,
  type LayoutDropdownKey,
  type LayoutMenuCategory,
  type LayoutMenuItem,
} from './LayoutNavSupport'
import { preloadDashboardRoute } from '../app/routeLoaders'

interface LayoutMobileNavigationSection {
  key: LayoutDropdownKey
  label: string
  categories: LayoutMenuCategory[]
}

interface LayoutMobileNavigationProps {
  configSection?: string
  isConfigPage: boolean
  openSection: LayoutDropdownKey | null
  pathname: string
  sections: LayoutMobileNavigationSection[]
  onConfigSelect: (item: LayoutMenuItem) => void
  onNavigate: () => void
  onSectionToggle: (section: LayoutDropdownKey) => void
}

export default function LayoutMobileNavigation({
  configSection,
  isConfigPage,
  openSection,
  pathname,
  sections,
  onConfigSelect,
  onNavigate,
  onSectionToggle,
}: LayoutMobileNavigationProps) {
  const navigationRef = useRef<HTMLElement>(null)
  const visibleSections = sections.filter((section) => section.categories.length > 0)

  useEffect(() => {
    const activeControl = navigationRef.current?.querySelector<HTMLElement>(
      '[data-mobile-nav-control][aria-current="page"]',
    )
    const firstControl = navigationRef.current?.querySelector<HTMLElement>(
      '[data-mobile-nav-control]',
    )
    ;(activeControl || firstControl)?.focus()
  }, [])

  const handleNavigationKeyDown = (event: KeyboardEvent<HTMLElement>) => {
    if (event.key === 'Escape') {
      event.preventDefault()
      onNavigate()
      window.requestAnimationFrame(() => {
        document.querySelector<HTMLElement>('[aria-controls="mobile-navigation"]')?.focus()
      })
      return
    }

    if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return
    const controls = Array.from(
      navigationRef.current?.querySelectorAll<HTMLElement>('[data-mobile-nav-control]') || [],
    ).filter((control) => !control.hasAttribute('disabled') && control.offsetParent !== null)
    if (controls.length === 0) return

    event.preventDefault()
    const currentIndex = controls.indexOf(document.activeElement as HTMLElement)
    if (event.key === 'Home') {
      controls[0].focus()
    } else if (event.key === 'End') {
      controls[controls.length - 1].focus()
    } else {
      const direction = event.key === 'ArrowDown' ? 1 : -1
      const nextIndex =
        currentIndex < 0
          ? direction > 0
            ? 0
            : controls.length - 1
          : (currentIndex + direction + controls.length) % controls.length
      controls[nextIndex].focus()
    }
  }

  const renderMenuItem = (item: LayoutMenuItem, key: string) => {
    const active = isLayoutMenuItemActive(item, pathname, isConfigPage, configSection)
    const className = `${styles.mobileNavLink} ${active ? styles.mobileNavLinkActive : ''}`

    if (item.kind === 'config') {
      return (
        <button
          key={key}
          type="button"
          className={className}
          aria-current={active ? 'page' : undefined}
          data-mobile-nav-control
          onFocus={() => void preloadDashboardRoute(`/config/${item.configSection}`)}
          onPointerDown={() => void preloadDashboardRoute(`/config/${item.configSection}`)}
          onClick={() => onConfigSelect(item)}
        >
          {item.label}
        </button>
      )
    }

    return (
      <NavLink
        key={key}
        to={item.to}
        className={className}
        data-mobile-nav-control
        onFocus={() => void preloadDashboardRoute(item.to)}
        onPointerDown={() => void preloadDashboardRoute(item.to)}
        onClick={onNavigate}
      >
        {item.label}
      </NavLink>
    )
  }

  return (
    <nav
      ref={navigationRef}
      id="mobile-navigation"
      className={styles.mobileNav}
      aria-label="Mobile navigation"
      onKeyDown={handleNavigationKeyDown}
    >
      {PRIMARY_NAV_LINKS.map((link) => (
        <NavLink
          key={`mobile-${link.to}`}
          end={link.matchMode !== 'prefix'}
          to={link.to}
          data-mobile-nav-control
          className={({ isActive }) =>
            isActive
              ? `${styles.mobileNavLink} ${styles.mobileNavLinkActive}`
              : styles.mobileNavLink
          }
          onFocus={() => void preloadDashboardRoute(link.to)}
          onPointerDown={() => void preloadDashboardRoute(link.to)}
          onClick={onNavigate}
        >
          {link.label}
        </NavLink>
      ))}

      {visibleSections.map((section) => {
        const expanded = openSection === section.key
        const panelId = `mobile-navigation-${section.key}`
        const sectionActive = section.categories.some((category) =>
          category.sections.some((menuSection) =>
            menuSection.items.some((item) =>
              isLayoutMenuItemActive(item, pathname, isConfigPage, configSection),
            ),
          ),
        )

        return (
          <div key={section.key} className={styles.mobileNavSection}>
            <button
              type="button"
              className={`${styles.mobileNavSectionToggle} ${sectionActive ? styles.mobileNavSectionToggleActive : ''}`}
              aria-controls={panelId}
              aria-expanded={expanded}
              data-mobile-nav-control
              onClick={() => onSectionToggle(section.key)}
            >
              <span className={styles.mobileNavSectionLabel}>
                {section.label}
                {sectionActive ? (
                  <span className={styles.mobileNavActiveMarker}>Current</span>
                ) : null}
              </span>
              <span aria-hidden="true">{expanded ? '−' : '+'}</span>
            </button>

            {expanded ? (
              <div id={panelId} className={styles.mobileNavSectionPanel}>
                {section.categories.map((category) => {
                  const categoryTitleId = `${panelId}-${category.key}-title`
                  return (
                    <section
                      key={`${section.label}-${category.key}`}
                      className={styles.mobileNavCategory}
                      aria-labelledby={categoryTitleId}
                    >
                      <h3 id={categoryTitleId} className={styles.mobileNavCategoryTitle}>
                        {category.label}
                      </h3>
                      <p className={styles.mobileNavCategoryDescription}>{category.description}</p>
                      {category.sections.map((menuSection, menuSectionIndex) => {
                        const subsectionTitleId = `${categoryTitleId}-${menuSectionIndex}`
                        return (
                          <div
                            key={`${section.label}-${category.key}-${menuSection.title}`}
                            role="group"
                            aria-labelledby={subsectionTitleId}
                          >
                            <h4 id={subsectionTitleId} className={styles.mobileNavSubsectionTitle}>
                              {menuSection.title}
                            </h4>
                            {menuSection.items.map((item) =>
                              renderMenuItem(
                                item,
                                `${section.label}-${category.key}-${menuSection.title}-${item.label}`,
                              ),
                            )}
                          </div>
                        )
                      })}
                    </section>
                  )
                })}
              </div>
            ) : null}
          </div>
        )
      })}
    </nav>
  )
}
