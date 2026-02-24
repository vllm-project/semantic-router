import React, { useEffect, useState, useCallback } from 'react'
import styles from './Ratings.module.css'
import RatingsTable from '../components/RatingsTable'
import type { RatingRow } from '../components/RatingsTable'

const RATINGS_API = '/api/router/api/v1/ratings'
const DEFAULT_CATEGORY = ''
const REFRESH_INTERVAL_MS = 15000

interface RatingsResponse {
  category: string
  ratings: RatingRow[]
  count: number
  timestamp?: string
}

const RatingsPage: React.FC = () => {
  const [ratings, setRatings] = useState<RatingRow[]>([])
  const [category, setCategory] = useState(DEFAULT_CATEGORY)
  const [customCategory, setCustomCategory] = useState('')
  const [categories, setCategories] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(true)

  const effectiveCategory = customCategory.trim() || category

  const fetchRatings = useCallback(async () => {
    const eff = customCategory.trim() || category
    try {
      const url = eff ? `${RATINGS_API}?category=${encodeURIComponent(eff)}` : RATINGS_API
      const response = await fetch(url)
      if (!response.ok) {
        const errBody = await response.text()
        let msg = response.statusText
        try {
          const j = JSON.parse(errBody)
          const err = j?.error
          if (err && typeof err.message === 'string') msg = err.message
          else if (typeof j?.message === 'string') msg = j.message
        } catch {
          if (errBody) msg = errBody.slice(0, 200)
        }
        throw new Error(msg)
      }
      const data: RatingsResponse = await response.json()
      setRatings(data.ratings || [])
      setLastUpdated(new Date())
      setError(null)
      if (data.ratings?.length && data.category && data.category !== 'global') {
        setCategories((prev) =>
          prev.includes(data.category) ? prev : [...prev, data.category].sort()
        )
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load ratings')
      setRatings([])
    } finally {
      setLoading(false)
    }
  }, [category, customCategory])

  useEffect(() => {
    fetchRatings()
    if (autoRefresh) {
      const interval = setInterval(fetchRatings, REFRESH_INTERVAL_MS)
      return () => clearInterval(interval)
    }
  }, [fetchRatings, autoRefresh])

  const categoryLabel = effectiveCategory || 'global'

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <h1 className={styles.title}>
            <span className={styles.titleIcon}>📊</span>
            Elo Leaderboard
          </h1>
          <p className={styles.subtitle}>
            Model rankings by category. Submit feedback in the Playground to update ratings.
          </p>
        </div>
        <div className={styles.headerRight}>
          <label className={styles.autoRefreshToggle}>
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            Auto-refresh
          </label>
          <button type="button" className={styles.refreshButton} onClick={() => { setLoading(true); fetchRatings() }}>
            Refresh
          </button>
        </div>
      </div>

      <div className={styles.controls}>
        <label className={styles.categoryLabel} htmlFor="ratings-category">
          Category
        </label>
        <select
          id="ratings-category"
          className={styles.categorySelect}
          value={category}
          onChange={(e) => { setCategory(e.target.value); setCustomCategory('') }}
        >
          <option value="">global</option>
          {categories.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
        <label className={styles.categoryLabel} htmlFor="ratings-custom-category">
          Or custom
        </label>
        <input
          id="ratings-custom-category"
          type="text"
          className={styles.categoryInput}
          placeholder="e.g. coding"
          value={customCategory}
          onChange={(e) => setCustomCategory(e.target.value)}
        />
      </div>

      {error && (
        <div className={styles.error}>
          <span className={styles.errorIcon}>⚠️</span>
          {error}
        </div>
      )}

      <section className={styles.section}>
        <div className={styles.sectionHeader}>
          <h2 className={styles.sectionTitle}>Leaderboard — {categoryLabel}</h2>
          {lastUpdated && (
            <span className={styles.lastUpdated}>
              Updated {lastUpdated.toLocaleTimeString()}
            </span>
          )}
        </div>
        <RatingsTable ratings={ratings} category={categoryLabel} loading={loading} />
      </section>
    </div>
  )
}

export default RatingsPage
