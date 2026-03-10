import React from 'react'
import styles from './RatingsTable.module.css'

export interface RatingRow {
  model: string
  rating: number
  wins: number
  losses: number
  ties: number
}

interface RatingsTableProps {
  ratings: RatingRow[]
  category: string
  loading?: boolean
}

const RatingsTable: React.FC<RatingsTableProps> = ({ ratings, category, loading }) => {
  if (loading) {
    return (
      <div className={styles.wrapper}>
        <div className={styles.loading}>Loading leaderboard…</div>
      </div>
    )
  }

  const games = (r: RatingRow) => r.wins + r.losses + r.ties
  const sorted = [...ratings].sort((a, b) => b.rating - a.rating)

  return (
    <div className={styles.wrapper}>
      <table className={styles.table}>
        <thead>
          <tr>
            <th className={styles.rankCol}>#</th>
            <th className={styles.modelCol}>Model</th>
            <th className={styles.ratingCol}>Rating</th>
            <th className={styles.gamesCol}>Games</th>
            <th className={styles.winsCol}>Wins</th>
            <th className={styles.lossesCol}>Losses</th>
            <th className={styles.tiesCol}>Ties</th>
          </tr>
        </thead>
        <tbody>
          {sorted.length === 0 ? (
            <tr>
              <td colSpan={7} className={styles.emptyCell}>
                No ratings yet for category “{category}”. Submit feedback in the Playground to build the leaderboard.
              </td>
            </tr>
          ) : (
            sorted.map((r, idx) => (
              <tr key={r.model}>
                <td className={styles.rankCol}>{idx + 1}</td>
                <td className={styles.modelCol}>{r.model}</td>
                <td className={styles.ratingCol}>{Math.round(r.rating)}</td>
                <td className={styles.gamesCol}>{games(r)}</td>
                <td className={styles.winsCol}>{r.wins}</td>
                <td className={styles.lossesCol}>{r.losses}</td>
                <td className={styles.tiesCol}>{r.ties}</td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  )
}

export default RatingsTable
