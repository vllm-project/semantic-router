import React from 'react'
import styles from './styles.module.css'

const GoalsList: string[] = [
  'How to capture the missing signals in request, response and context?',
  'How to combine the signals to make better decisions?',
  'How to collaborate more efficiently between different models?',
  'How to secure the real world and LLM system from jailbreaks, pii leaks, hallucinations?',
  'How to collect the valuable signals and build a self-learning system?',
]

const GoalItem: React.FC<{ goal: string, index: number }> = ({ goal, index }) => {
  return (
    <div className={styles.goalItem}>
      <div className={styles.goalNumber}>{index + 1}</div>
      <div className={styles.goalText}>{goal}</div>
    </div>
  )
}

const HomepageFeatures: React.FC = () => {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className={styles.featuresHeader}>
          <h2 className={styles.featuresTitle}>
            🎯 Our Goals
          </h2>
          <p className={styles.featuresSubtitle}>
            Building the System Level Intelligence for Mixture-of-Models (MoM), bringing Collective Intelligence into LLM systems
          </p>
        </div>
        <div className={styles.goalsBanner}>
          <img
            src="/img/banner.png"
            alt="vLLM Semantic Router Banner"
            className={styles.bannerImage}
          />
        </div>
        <div className={styles.goalsContainer}>
          {GoalsList.map((goal, idx) => (
            <GoalItem key={idx} goal={goal} index={idx} />
          ))}
        </div>
      </div>
    </section>
  )
}

export default HomepageFeatures
