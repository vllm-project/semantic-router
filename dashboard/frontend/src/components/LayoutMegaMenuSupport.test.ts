import { describe, expect, it } from 'vitest'

import {
  ANALYZE_MENU_CATEGORIES,
  BUILD_MENU_CATEGORIES,
  type LayoutMenuCategory,
} from './LayoutNavSupport'
import { getLayoutMegaMenuGeometry } from './LayoutMegaMenuSupport'

function categoryByKey(categories: LayoutMenuCategory[], key: string): LayoutMenuCategory {
  const category = categories.find((candidate) => candidate.key === key)
  if (!category) {
    throw new Error(`Missing layout category: ${key}`)
  }
  return category
}

describe('layout mega-menu geometry', () => {
  it('keeps sparse multi-section navigation compact', () => {
    const outcomes = categoryByKey(ANALYZE_MENU_CATEGORIES, 'outcomes')

    expect(getLayoutMegaMenuGeometry(outcomes)).toEqual({
      density: 'compact',
      itemCount: 3,
      maxWidth: 860,
      railWidth: 200,
      sectionCount: 3,
    })
  })

  it('uses standard geometry for a balanced routing category', () => {
    const routing = categoryByKey(BUILD_MENU_CATEGORIES, 'routing')

    expect(getLayoutMegaMenuGeometry(routing)).toEqual({
      density: 'standard',
      itemCount: 6,
      maxWidth: 980,
      railWidth: 210,
      sectionCount: 3,
    })
  })

  it('reserves the widest geometry for genuinely dense navigation', () => {
    const denseCategory: LayoutMenuCategory = {
      key: 'dense',
      label: 'Dense',
      description: 'A synthetic high-density category.',
      sections: Array.from({ length: 4 }, (_, sectionIndex) => ({
        title: `Section ${sectionIndex + 1}`,
        items: Array.from({ length: 2 }, (_, itemIndex) => ({
          kind: 'route' as const,
          label: `Item ${sectionIndex + 1}-${itemIndex + 1}`,
          to: `/item-${sectionIndex + 1}-${itemIndex + 1}`,
        })),
      })),
    }

    expect(getLayoutMegaMenuGeometry(denseCategory)).toEqual({
      density: 'dense',
      itemCount: 8,
      maxWidth: 1280,
      railWidth: 220,
      sectionCount: 4,
    })
  })
})
