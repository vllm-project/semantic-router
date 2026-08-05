import type { LayoutMenuCategory } from './LayoutNavSupport'

export type LayoutMegaMenuDensity = 'compact' | 'standard' | 'dense'

export interface LayoutMegaMenuGeometry {
  density: LayoutMegaMenuDensity
  itemCount: number
  sectionCount: number
}

function resolveDensity(sectionCount: number, itemCount: number): LayoutMegaMenuDensity {
  if (sectionCount <= 3 && itemCount <= 4) {
    return 'compact'
  }

  if (sectionCount <= 3 && itemCount <= 7) {
    return 'standard'
  }

  return 'dense'
}

export function getLayoutMegaMenuGeometry(category: LayoutMenuCategory): LayoutMegaMenuGeometry {
  const sectionCount = category.sections.length
  const itemCount = category.sections.reduce((total, section) => total + section.items.length, 0)
  const density = resolveDensity(sectionCount, itemCount)

  return {
    density,
    itemCount,
    sectionCount,
  }
}
