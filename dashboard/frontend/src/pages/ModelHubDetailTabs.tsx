import React, { useRef } from 'react'

import styles from './ModelHubPage.module.css'

export type ModelHubDetailTab = 'overview' | 'evaluations' | 'access' | 'pool'

export interface ModelHubDetailTabItem {
  id: ModelHubDetailTab
  label: string
}

const nextTabIndex = (
  event: React.KeyboardEvent<HTMLButtonElement>,
  currentIndex: number,
  tabCount: number,
): number | null => {
  if (event.key === 'ArrowRight') return (currentIndex + 1) % tabCount
  if (event.key === 'ArrowLeft') return (currentIndex - 1 + tabCount) % tabCount
  if (event.key === 'Home') return 0
  if (event.key === 'End') return tabCount - 1
  return null
}

export const ModelHubDetailTabs: React.FC<{
  detailID: string
  tabs: ModelHubDetailTabItem[]
  selected: ModelHubDetailTab
  select: (tab: ModelHubDetailTab) => void
}> = ({ detailID, tabs, selected, select }) => {
  const tabRefs = useRef<Partial<Record<ModelHubDetailTab, HTMLButtonElement | null>>>({})

  const selectFromKeyboard = (
    event: React.KeyboardEvent<HTMLButtonElement>,
    current: ModelHubDetailTab,
  ): void => {
    const currentIndex = tabs.findIndex((item) => item.id === current)
    const nextIndex = nextTabIndex(event, currentIndex, tabs.length)
    if (nextIndex === null) return
    event.preventDefault()
    const next = tabs[nextIndex].id
    select(next)
    tabRefs.current[next]?.focus()
  }

  return (
    <div className={styles.detailTabs} role="tablist" aria-label="Model details">
      {tabs.map((item) => (
        <button
          key={item.id}
          id={`${detailID}-tab-${item.id}`}
          type="button"
          role="tab"
          aria-controls={`${detailID}-panel`}
          aria-selected={selected === item.id}
          tabIndex={selected === item.id ? 0 : -1}
          className={selected === item.id ? styles.detailTabActive : ''}
          ref={(node) => {
            tabRefs.current[item.id] = node
          }}
          onClick={() => select(item.id)}
          onKeyDown={(event) => selectFromKeyboard(event, item.id)}
        >
          {item.label}
        </button>
      ))}
    </div>
  )
}
