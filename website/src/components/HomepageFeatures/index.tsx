import React from 'react'
import Translate from '@docusaurus/Translate'
import styles from './styles.module.css'

const HomepageFeatures: React.FC = () => {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className={styles.featuresHeader}>
          <h2 className={styles.featuresTitle}>
            🎯
            {' '}
            <Translate id="features.sectionTitle">Our Goals</Translate>
          </h2>
          <p className={styles.featuresSubtitle}>
            <Translate id="features.sectionSubtitle">Building the System Level Intelligence for Mixture-of-Models (MoM), bringing Collective Intelligence into LLM systems</Translate>
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
          <div className={styles.goalItem}>
            <div className={styles.goalNumber}>1</div>
            <div className={styles.goalText}>
              <Translate id="features.goal1">How to capture the missing signals in request, response and context?</Translate>
            </div>
          </div>
          <div className={styles.goalItem}>
            <div className={styles.goalNumber}>2</div>
            <div className={styles.goalText}>
              <Translate id="features.goal2">How to combine the signals to make better decisions?</Translate>
            </div>
          </div>
          <div className={styles.goalItem}>
            <div className={styles.goalNumber}>3</div>
            <div className={styles.goalText}>
              <Translate id="features.goal3">How to collaborate more efficiently between different models?</Translate>
            </div>
          </div>
          <div className={styles.goalItem}>
            <div className={styles.goalNumber}>4</div>
            <div className={styles.goalText}>
              <Translate id="features.goal4">How to secure the real world and LLM system from jailbreaks, pii leaks, hallucinations?</Translate>
            </div>
          </div>
          <div className={styles.goalItem}>
            <div className={styles.goalNumber}>5</div>
            <div className={styles.goalText}>
              <Translate id="features.goal5">How to collect the valuable signals and build a self-learning system?</Translate>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

export default HomepageFeatures
