import { describe, expect, it } from 'vitest'
import { findNavItem } from './navigation'

describe('findNavItem', () => {
  it('matches an item by its own path and its detail routes', () => {
    expect(findNavItem('/observe/requests')?.item.id).toBe('requests')
    expect(findNavItem('/observe/requests/abc123')?.item.id).toBe('requests')
  })

  it('matches an item by an alias outside its path', () => {
    expect(findNavItem('/observe/runs')?.item.id).toBe('requests')
    expect(findNavItem('/observe/runs')?.group.id).toBe('observe')
  })

  it('keeps an exact item exact and finds nothing for unknown paths', () => {
    expect(findNavItem('/')?.item.id).toBe('home')
    expect(findNavItem('/nowhere')).toBeUndefined()
  })
})
