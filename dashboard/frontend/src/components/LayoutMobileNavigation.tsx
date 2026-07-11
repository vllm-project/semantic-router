import { NavLink } from 'react-router-dom'
import styles from './Layout.module.css'
import {
  isLayoutMenuItemActive,
  PRIMARY_NAV_LINKS,
  type LayoutDropdownKey,
  type LayoutMenuCategory,
  type LayoutMenuItem,
} from './LayoutNavSupport'

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
  const renderMenuItem = (item: LayoutMenuItem, key: string) => {
    const active = isLayoutMenuItemActive(item, pathname, isConfigPage, configSection)
    const className = `${styles.mobileNavLink} ${active ? styles.mobileNavLinkActive : ''}`

    if (item.kind === 'config') {
      return (
        <button key={key} type="button" className={className} onClick={() => onConfigSelect(item)}>
          {item.label}
        </button>
      )
    }

    return (
      <NavLink key={key} to={item.to} className={className} onClick={onNavigate}>
        {item.label}
      </NavLink>
    )
  }

  return (
    <div
      id="mobile-navigation"
      className={styles.mobileNav}
      role="navigation"
      aria-label="Mobile navigation"
    >
      {PRIMARY_NAV_LINKS.map((link) => (
        <NavLink
          key={`mobile-${link.to}`}
          end={link.matchMode !== 'prefix'}
          to={link.to}
          className={({ isActive }) =>
            isActive
              ? `${styles.mobileNavLink} ${styles.mobileNavLinkActive}`
              : styles.mobileNavLink
          }
          onClick={onNavigate}
        >
          {link.label}
        </NavLink>
      ))}

      {sections.map((section) => {
        const expanded = openSection === section.key
        const panelId = `mobile-navigation-${section.key}`

        return (
          <div key={section.key} className={styles.mobileNavSection}>
            <button
              type="button"
              className={styles.mobileNavSectionToggle}
              aria-controls={panelId}
              aria-expanded={expanded}
              onClick={() => onSectionToggle(section.key)}
            >
              <span>{section.label}</span>
              <span aria-hidden="true">{expanded ? '−' : '+'}</span>
            </button>

            {expanded ? (
              <div id={panelId} className={styles.mobileNavSectionPanel}>
                {section.categories.map((category) => (
                  <div
                    key={`${section.label}-${category.key}`}
                    className={styles.mobileNavCategory}
                  >
                    <div className={styles.mobileNavCategoryTitle}>{category.label}</div>
                    <p className={styles.mobileNavCategoryDescription}>{category.description}</p>
                    {category.sections.map((menuSection) => (
                      <div key={`${section.label}-${category.key}-${menuSection.title}`}>
                        <div className={styles.mobileNavSubsectionTitle}>{menuSection.title}</div>
                        {menuSection.items.map((item) =>
                          renderMenuItem(
                            item,
                            `${section.label}-${category.key}-${menuSection.title}-${item.label}`,
                          ),
                        )}
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            ) : null}
          </div>
        )
      })}
    </div>
  )
}
