import { useEffect, useMemo, useState, type FocusEvent, type KeyboardEvent } from 'react'
import { NavLink } from 'react-router-dom'
import type { LayoutMenuCategory, LayoutMenuItem } from './LayoutNavSupport'
import styles from './LayoutMegaMenu.module.css'

interface LayoutMegaMenuProps {
  id: string
  triggerId: string
  label: string
  categories: LayoutMenuCategory[]
  activeCategoryKey?: string
  isItemActive: (item: LayoutMenuItem) => boolean
  onConfigSelect: (item: Extract<LayoutMenuItem, { kind: 'config' }>) => void
  onNavigate: () => void
}

const MENU_DESCRIPTIONS: Record<string, string> = {
  Build: 'Shape signals, decisions, model paths, and the context behind them.',
  Analyze: 'Measure routing outcomes and plan the heterogeneous model fleet.',
  Operate: 'Run, observe, and administer the live routing control plane.',
}

const MENU_SEQUENCE: Record<string, string> = {
  Build: 'DESIGN / DECIDE / CONNECT',
  Analyze: 'INSPECT / EVALUATE / PLAN',
  Operate: 'RUN / OBSERVE / GOVERN',
}

const LayoutMegaMenu = ({
  id,
  triggerId,
  label,
  categories,
  activeCategoryKey,
  isItemActive,
  onConfigSelect,
  onNavigate,
}: LayoutMegaMenuProps) => {
  const initialCategoryKey = activeCategoryKey ?? categories[0]?.key ?? ''
  const [selectedCategoryKey, setSelectedCategoryKey] = useState(initialCategoryKey)

  useEffect(() => {
    setSelectedCategoryKey((currentCategoryKey) => {
      if (categories.some((category) => category.key === currentCategoryKey)) {
        return currentCategoryKey
      }

      if (activeCategoryKey && categories.some((category) => category.key === activeCategoryKey)) {
        return activeCategoryKey
      }

      return categories[0]?.key ?? ''
    })
  }, [activeCategoryKey, categories])

  const selectedCategory = useMemo(
    () => categories.find((category) => category.key === selectedCategoryKey) ?? categories[0],
    [categories, selectedCategoryKey],
  )

  const focusCategory = (categoryIndex: number) => {
    const category = categories[categoryIndex]
    if (!category) return

    setSelectedCategoryKey(category.key)
    document.getElementById(`${id}-${category.key}-tab`)?.focus()
  }

  const handleCategoryKeyDown = (
    event: KeyboardEvent<HTMLButtonElement>,
    categoryIndex: number,
  ) => {
    let nextIndex: number | null = null

    switch (event.key) {
      case 'ArrowDown':
        nextIndex = (categoryIndex + 1) % categories.length
        break
      case 'ArrowUp':
        nextIndex = (categoryIndex - 1 + categories.length) % categories.length
        break
      case 'Home':
        nextIndex = 0
        break
      case 'End':
        nextIndex = categories.length - 1
        break
      case 'ArrowRight':
        event.preventDefault()
        document
          .getElementById(`${id}-${selectedCategoryKey}-panel`)
          ?.querySelector<HTMLElement>('[data-mega-link]')
          ?.focus()
        return
      default:
        return
    }

    event.preventDefault()
    focusCategory(nextIndex)
  }

  const handlePanelBlur = (event: FocusEvent<HTMLDivElement>) => {
    const nextTarget = event.relatedTarget
    if (
      nextTarget instanceof Node &&
      (event.currentTarget.contains(nextTarget) ||
        nextTarget === document.getElementById(triggerId))
    ) {
      return
    }

    onNavigate()
  }

  if (!selectedCategory) {
    return null
  }

  const columnsClassName =
    selectedCategory.sections.length >= 3 ? styles.columnsThree : styles.columnsTwo

  return (
    <div
      id={id}
      role="dialog"
      aria-labelledby={triggerId}
      className={styles.menu}
      data-testid={`layout-mega-menu-${label.toLowerCase()}`}
      onBlur={handlePanelBlur}
    >
      <aside className={styles.rail} data-testid="layout-mega-menu-rail">
        <div className={styles.railIntro}>
          <span className={styles.railMarker} />
          <span className={styles.eyebrow}>Semantic Router control plane</span>
          <strong className={styles.railTitle}>{label}</strong>
          <p className={styles.railDescription}>
            {MENU_DESCRIPTIONS[label] ?? 'Navigate the semantic routing control plane.'}
          </p>
          <span className={styles.railSequence} aria-hidden="true">
            {MENU_SEQUENCE[label]}
          </span>
        </div>

        <div
          className={styles.categoryTabs}
          role="tablist"
          aria-label={`${label} categories`}
          aria-orientation="vertical"
        >
          {categories.map((category, categoryIndex) => {
            const selected = category.key === selectedCategory.key
            return (
              <button
                key={category.key}
                id={`${id}-${category.key}-tab`}
                type="button"
                role="tab"
                aria-selected={selected}
                aria-controls={selected ? `${id}-${category.key}-panel` : undefined}
                tabIndex={selected ? 0 : -1}
                className={`${styles.categoryTab} ${selected ? styles.categoryTabSelected : ''}`}
                onClick={() => setSelectedCategoryKey(category.key)}
                onFocus={() => setSelectedCategoryKey(category.key)}
                onKeyDown={(event) => handleCategoryKeyDown(event, categoryIndex)}
              >
                <span className={styles.categoryIndex}>
                  {String(categoryIndex + 1).padStart(2, '0')}
                </span>
                <span>{category.label}</span>
                <svg
                  width="14"
                  height="14"
                  viewBox="0 0 14 14"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  aria-hidden="true"
                >
                  <path d="M4.5 2.75L8.75 7L4.5 11.25" strokeLinecap="round" />
                </svg>
              </button>
            )
          })}
        </div>
      </aside>

      <div
        id={`${id}-${selectedCategory.key}-panel`}
        role="tabpanel"
        aria-labelledby={`${id}-${selectedCategory.key}-tab`}
        className={styles.panel}
      >
        <div className={styles.panelHeader}>
          <div>
            <span className={styles.panelEyebrow}>{label}</span>
            <h2 className={styles.panelTitle}>{selectedCategory.label}</h2>
          </div>
          <p className={styles.panelDescription}>{selectedCategory.description}</p>
        </div>

        <div
          className={`${styles.content} ${columnsClassName}`}
          data-testid="layout-mega-menu-content"
        >
          {selectedCategory.sections.map((section) => (
            <section
              key={`${selectedCategory.key}-${section.title}`}
              className={`${styles.section} ${selectedCategory.sections.length === 1 ? styles.singleSection : ''}`}
            >
              <h3 className={styles.sectionTitle}>{section.title}</h3>
              {section.description ? (
                <p className={styles.sectionDescription}>{section.description}</p>
              ) : null}
              <div className={styles.items}>
                {section.items.map((item) => {
                  const active = isItemActive(item)
                  const className = `${styles.item} ${active ? styles.itemActive : ''}`
                  const key = `${section.title}-${item.label}`

                  if (item.kind === 'config') {
                    return (
                      <button
                        key={key}
                        type="button"
                        data-mega-link
                        className={className}
                        onClick={() => onConfigSelect(item)}
                      >
                        <span>{item.label}</span>
                        <span className={styles.itemArrow} aria-hidden="true">
                          ↗
                        </span>
                      </button>
                    )
                  }

                  return (
                    <NavLink
                      key={key}
                      data-mega-link
                      to={item.to}
                      className={className}
                      onClick={onNavigate}
                    >
                      <span>{item.label}</span>
                      <span className={styles.itemArrow} aria-hidden="true">
                        ↗
                      </span>
                    </NavLink>
                  )
                })}
              </div>
            </section>
          ))}
        </div>
      </div>
    </div>
  )
}

export default LayoutMegaMenu
