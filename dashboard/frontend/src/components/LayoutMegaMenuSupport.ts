import type { LayoutMenuCategory } from './LayoutNavSupport'

export type LayoutMegaMenuDensity = 'compact' | 'standard' | 'dense'

export interface LayoutMegaMenuGeometry {
  density: LayoutMegaMenuDensity
  itemCount: number
  maxWidth: number
  railWidth: number
  sectionCount: number
}

const GEOMETRY_BY_DENSITY: Record<
  LayoutMegaMenuDensity,
  Pick<LayoutMegaMenuGeometry, 'maxWidth' | 'railWidth'>
> = {
  compact: { maxWidth: 860, railWidth: 200 },
  standard: { maxWidth: 980, railWidth: 210 },
  dense: { maxWidth: 1280, railWidth: 220 },
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
    ...GEOMETRY_BY_DENSITY[density],
  }
}
