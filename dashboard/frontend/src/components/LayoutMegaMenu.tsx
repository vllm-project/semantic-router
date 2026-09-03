import { useEffect, useMemo, useState, type FocusEvent, type KeyboardEvent } from 'react'
import { NavLink } from 'react-router-dom'
import type { LayoutMenuCategory, LayoutMenuItem } from './LayoutNavSupport'
import { getLayoutMegaMenuGeometry } from './LayoutMegaMenuSupport'
import ProductIcon from './ProductIcon'
import styles from './LayoutMegaMenu.module.css'

interface LayoutMegaMenuProps {
  id: string
  triggerId: string
  label: string
  categories: LayoutMenuCategory[]
  activeCategoryKey?: string
  isItemActive: (item: LayoutMenuItem) => boolean
  onConfigSelect: (item: Extract<LayoutMenuItem, { kind: 'config' }>) => void
  onItemIntent: (item: LayoutMenuItem) => void
  onNavigate: () => void
}

const LayoutMegaMenu = ({
  id,
  triggerId,
  label,
  categories,
  activeCategoryKey,
  isItemActive,
  onConfigSelect,
  onItemIntent,
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

  const handlePanelBlur = (event: FocusEvent<HTMLElement>) => {
    const nextTarget = event.relatedTarget
    // Safari can report no related target when pointer activation moves focus
    // away from the menu. Closing synchronously here unmounts the link before
    // its click handler runs, so let the click/outside-click paths own it.
    if (!nextTarget) {
      return
    }

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

  const geometry = getLayoutMegaMenuGeometry(selectedCategory)

  return (
    <nav
      id={id}
      aria-labelledby={triggerId}
      className={styles.menu}
      data-density={geometry.density}
      data-item-count={geometry.itemCount}
      data-section-count={geometry.sectionCount}
      data-testid={`layout-mega-menu-${label.toLowerCase()}`}
      onBlur={handlePanelBlur}
    >
      <aside className={styles.rail} data-testid="layout-mega-menu-rail">
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
                <ProductIcon name="chevron-right" className={styles.categoryArrow} />
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
        <div className={styles.content} data-testid="layout-mega-menu-content">
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
                        onFocus={() => onItemIntent(item)}
                        onPointerEnter={() => onItemIntent(item)}
                        onClick={() => onConfigSelect(item)}
                      >
                        <span className={styles.itemLabel}>
                          <ProductIcon name={item.icon} />
                          <span>{item.label}</span>
                        </span>
                        <ProductIcon name="chevron-right" className={styles.itemArrow} />
                      </button>
                    )
                  }

                  return (
                    <NavLink
                      key={key}
                      data-mega-link
                      to={item.to}
                      className={className}
                      onFocus={() => onItemIntent(item)}
                      onPointerEnter={() => onItemIntent(item)}
                      onClick={onNavigate}
                    >
                      <span className={styles.itemLabel}>
                        <ProductIcon name={item.icon} />
                        <span>{item.label}</span>
                      </span>
                      <ProductIcon name="chevron-right" className={styles.itemArrow} />
                    </NavLink>
                  )
                })}
              </div>
            </section>
          ))}
        </div>
      </div>
    </nav>
  )
}

export default LayoutMegaMenu
