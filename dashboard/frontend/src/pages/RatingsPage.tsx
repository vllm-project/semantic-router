import React, { useCallback, useEffect, useState } from 'react'
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
  const categoryLabel = effectiveCategory || 'global'
  const topModel = ratings.reduce<RatingRow | null>(
    (currentTop, row) => (!currentTop || row.rating > currentTop.rating ? row : currentTop),
    null,
  )
  const totalGames = ratings.reduce((sum, row) => sum + row.wins + row.losses + row.ties, 0)
  const averageRating = ratings.length
    ? Math.round(ratings.reduce((sum, row) => sum + row.rating, 0) / ratings.length)
    : 0
  const availableCategoryCount = categories.length + 1

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

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <span className={styles.eyebrow}>Feedback Loop</span>
          <h1 className={styles.title}>Elo Leaderboard</h1>
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

      <div className={styles.summaryGrid}>
        <article className={styles.summaryCard}>
          <span className={styles.summaryLabel}>Current category</span>
          <strong className={styles.summaryValue}>{categoryLabel}</strong>
          <span className={styles.summaryHint}>
            {effectiveCategory ? 'Custom category filter is active.' : 'Global leaderboard across all feedback.'}
          </span>
        </article>
        <article className={styles.summaryCard}>
          <span className={styles.summaryLabel}>Ranked models</span>
          <strong className={styles.summaryValue}>{ratings.length}</strong>
          <span className={styles.summaryHint}>
            {topModel ? `Top model: ${topModel.model}` : 'No ratings have been recorded yet.'}
          </span>
        </article>
        <article className={styles.summaryCard}>
          <span className={styles.summaryLabel}>Games recorded</span>
          <strong className={styles.summaryValue}>{totalGames}</strong>
          <span className={styles.summaryHint}>
            {ratings.length ? `Average rating ${averageRating}` : 'Submit comparisons in Playground to populate Elo.'}
          </span>
        </article>
      </div>

      <section className={styles.filtersPanel}>
        <div className={styles.filtersHeader}>
          <div>
            <h2 className={styles.filtersTitle}>Filter leaderboard</h2>
            <p className={styles.filtersSubtitle}>Choose a saved category or type a custom label.</p>
          </div>
          <span className={styles.filtersMeta}>
            {availableCategoryCount} available {availableCategoryCount === 1 ? 'category' : 'categories'}
          </span>
        </div>

        <div className={styles.controls}>
          <div className={styles.controlField}>
            <label className={styles.categoryLabel} htmlFor="ratings-category">
              Saved category
            </label>
            <select
              id="ratings-category"
              className={styles.categorySelect}
              value={category}
              onChange={(e) => {
                setCategory(e.target.value)
                setCustomCategory('')
              }}
            >
              <option value="">global</option>
              {categories.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>

          <div className={styles.controlField}>
            <label className={styles.categoryLabel} htmlFor="ratings-custom-category">
              Custom category
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
        </div>
      </section>

      {error && (
        <div className={styles.error}>
          <span className={styles.errorLabel}>Load error</span>
          <span>{error}</span>
        </div>
      )}

      <section className={styles.section}>
        <div className={styles.sectionHeader}>
          <div>
            <h2 className={styles.sectionTitle}>Leaderboard</h2>
            <p className={styles.sectionSubtitle}>
              Sorted by Elo rating for <strong>{categoryLabel}</strong>.
            </p>
          </div>
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
