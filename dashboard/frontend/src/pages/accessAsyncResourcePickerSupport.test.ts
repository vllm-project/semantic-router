import { describe, expect, it } from 'vitest'
import {
  ACCESS_PICKER_PAGE_SIZE,
  accessPickerRequest,
  mergeAccessPickerPage,
  missingSelectedPickerIds,
} from './accessAsyncResourcePickerSupport'

type Item = { id: string; name: string }
const item = (id: number): Item => ({ id: `item-${id}`, name: `Item ${id}` })

describe('async access selector pagination', () => {
  it('loads more than one bounded page without changing the table cursor', () => {
    const firstPage = Array.from({ length: ACCESS_PICKER_PAGE_SIZE }, (_, index) => item(index))
    const secondPage = [item(19), item(20), item(21)]

    expect(accessPickerRequest('  Ada  ')).toEqual({
      q: 'Ada',
      cursor: undefined,
      limit: 20,
      status: 'active',
    })
    expect(accessPickerRequest('Ada', 'selector-page-2')).toEqual({
      q: 'Ada',
      cursor: 'selector-page-2',
      limit: 20,
      status: 'active',
    })

    const combined = mergeAccessPickerPage(firstPage, secondPage, true, (value) => value.id)
    expect(combined).toHaveLength(22)
    expect(combined[combined.length - 1]?.id).toBe('item-21')
  })

  it('hydrates selected IDs that are outside the visible search page', () => {
    expect(
      missingSelectedPickerIds(
        ['item-1', 'item-30', 'item-31'],
        [item(1)],
        { 'item-31': item(31) },
        {},
        (value) => value.id,
      ),
    ).toEqual(['item-30'])
  })
})
