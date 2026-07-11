import type { FocusEvent, KeyboardEvent } from 'react'
import { NavLink } from 'react-router-dom'
import type { LayoutMenuItem, LayoutMenuSection } from './LayoutNavSupport'
import styles from './LayoutMegaMenu.module.css'

interface LayoutMegaMenuProps {
  id: string
  triggerId: string
  label: string
  sections: LayoutMenuSection[]
  isItemActive: (item: LayoutMenuItem) => boolean
  onConfigSelect: (item: Extract<LayoutMenuItem, { kind: 'config' }>) => void
  onNavigate: () => void
}

const MENU_DESCRIPTIONS: Record<string, string> = {
  Manager: 'Configure the signals, decisions, and resources behind every route.',
  Simulator: 'Explore routing behavior before it reaches the live model fleet.',
  System: 'Inspect, evaluate, and operate the router from one control surface.',
}

function getSectionTitle(label: string, section: LayoutMenuSection, sectionIndex: number) {
  if (section.title) {
    return section.title
  }

  if (label === 'Manager') {
    return sectionIndex === 0 ? 'Workspace' : 'Routing'
  }

  return label
}

const LayoutMegaMenu = ({
  id,
  triggerId,
  label,
  sections,
  isItemActive,
  onConfigSelect,
  onNavigate,
}: LayoutMegaMenuProps) => {
  const handleMenuKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const menuItems = Array.from(
      event.currentTarget.querySelectorAll<HTMLElement>('[role="menuitem"]:not([disabled])'),
    )

    if (menuItems.length === 0) {
      return
    }

    const focusedIndex = menuItems.indexOf(document.activeElement as HTMLElement)
    let nextIndex: number | null = null

    switch (event.key) {
      case 'ArrowDown':
      case 'ArrowRight':
        nextIndex = focusedIndex < 0 ? 0 : (focusedIndex + 1) % menuItems.length
        break
      case 'ArrowUp':
      case 'ArrowLeft':
        nextIndex =
          focusedIndex < 0
            ? menuItems.length - 1
            : (focusedIndex - 1 + menuItems.length) % menuItems.length
        break
      case 'Home':
        nextIndex = 0
        break
      case 'End':
        nextIndex = menuItems.length - 1
        break
      default:
        return
    }

    event.preventDefault()
    menuItems[nextIndex].focus()
  }

  const columnsClassName = sections.length >= 3 ? styles.columnsThree : styles.columnsTwo

  const handleMenuBlur = (event: FocusEvent<HTMLDivElement>) => {
    const nextTarget = event.relatedTarget
    if (
      nextTarget instanceof Node &&
      (event.currentTarget.contains(nextTarget) || nextTarget === document.getElementById(triggerId))
    ) {
      return
    }

    onNavigate()
  }

  return (
    <div
      id={id}
      role="menu"
      aria-labelledby={triggerId}
      className={styles.menu}
      data-testid={`layout-mega-menu-${label.toLowerCase()}`}
      onKeyDown={handleMenuKeyDown}
      onBlur={handleMenuBlur}
    >
      <aside className={styles.rail} data-testid="layout-mega-menu-rail" aria-hidden="true">
        <span className={styles.railMarker} />
        <span className={styles.eyebrow}>Semantic Router</span>
        <strong className={styles.railTitle}>{label}</strong>
        <p className={styles.railDescription}>
          {MENU_DESCRIPTIONS[label] ?? 'Navigate the semantic routing control plane.'}
        </p>
      </aside>

      <div
        className={`${styles.content} ${columnsClassName}`}
        data-testid="layout-mega-menu-content"
      >
        {sections.map((section, sectionIndex) => (
          <section
            key={`${label}-${section.title || sectionIndex}`}
            className={`${styles.section} ${sections.length === 1 ? styles.singleSection : ''}`}
          >
            <h2 className={styles.sectionTitle}>{getSectionTitle(label, section, sectionIndex)}</h2>
            <div className={styles.items}>
              {section.items.map((item) => {
                const active = isItemActive(item)
                const className = `${styles.item} ${active ? styles.itemActive : ''}`
                const key = `${section.title || sectionIndex}-${item.label}`

                if (item.kind === 'config') {
                  return (
                    <button
                      key={key}
                      type="button"
                      role="menuitem"
                      className={className}
                      onClick={() => onConfigSelect(item)}
                    >
                      {item.label}
                    </button>
                  )
                }

                return (
                  <NavLink
                    key={key}
                    role="menuitem"
                    to={item.to}
                    className={className}
                    onClick={onNavigate}
                  >
                    {item.label}
                  </NavLink>
                )
              })}
            </div>
          </section>
        ))}
      </div>
    </div>
  )
}

export default LayoutMegaMenu
