import React from 'react'
import Layout from '@theme/Layout'
import Translate from '@docusaurus/Translate'
import styles from './promotion.module.css'

interface PromotionRule {
  role: string
  icon: string
  level: number
  requirements: string
  details: string[]
  permissions: string
  timeline: string
  application: string
  color: string
}

const promotionRules: PromotionRule[] = [
  {
    role: 'Reviewer',
    icon: '👀',
    level: 1,
    requirements: 'Active contributions within one release cycle',
    details: [
      'Review open PRs',
      'Help open GitHub Issues',
      'Engage in community meetings and slack channel discussions',
    ],
    permissions: 'Triage Permission',
    timeline: 'After each release (2-3 month intervals)',
    application: 'Nominated by a maintainer or self-nomination',
    color: '#4CAF50',
  },
  {
    role: 'Committer',
    icon: '💻',
    level: 2,
    requirements: 'Sustained contributions across two consecutive releases',
    details: [
      'Review open PRs',
      'Help open GitHub Issues',
      'Engage in community meetings and slack channel discussions',
      'Major feature development in workgroups',
      'Demonstrate technical leadership',
      'Mentor new contributors',
    ],
    permissions: 'Write Permission',
    timeline: 'After each release (2-3 month intervals)',
    application: 'Must be nominated by a maintainer, requires majority vote from maintainers',
    color: '#2196F3',
  },
  {
    role: 'Maintainer',
    icon: '🛠️',
    level: 3,
    requirements: 'Sustained contributions across three consecutive releases',
    details: [
      'Review open PRs',
      'Help open GitHub Issues',
      'Host community meetings',
      'Demonstrate long-term project commitment',
      'Lead major feature development in workgroups',
      'Shape project direction and roadmap',
    ],
    permissions: 'Maintain Permission',
    timeline: 'After each release (2-3 month intervals)',
    application: 'Must be nominated by a maintainer, requires unanimous approval from all maintainers',
    color: '#FF9800',
  },
]

interface PromotionCardProps {
  rule: PromotionRule
}

const PromotionCard: React.FC<PromotionCardProps> = ({ rule }) => {
  return (
    <div className={styles.promotionCard} style={{ borderColor: rule.color }}>
      <div className={styles.cardHeader}>
        <span className={styles.roleIcon}>{rule.icon}</span>
        <h3 className={styles.roleTitle} style={{ color: rule.color }}>{rule.role}</h3>
        <span className={styles.permissions} style={{ backgroundColor: rule.color }}>
          {rule.permissions}
        </span>
      </div>

      <div className={styles.cardContent}>
        <div className={styles.requirements}>
          <h4>📋 Requirements</h4>
          <p className={styles.mainRequirement}>{rule.requirements}</p>
          <ul className={styles.detailsList}>
            {rule.details.map((detail, index) => (
              <li key={index}>{detail}</li>
            ))}
          </ul>
        </div>

        <div className={styles.timeline}>
          <h4>⏰ Timeline</h4>
          <p>{rule.timeline}</p>
        </div>

        <div className={styles.application}>
          <h4>📝 How to Apply</h4>
          <p>{rule.application}</p>
        </div>
      </div>
    </div>
  )
}

const Promotion: React.FC = () => {
  return (
    <Layout
      title="Promotion"
      description="vLLM Semantic Router Community Promotion Rules"
    >
      <div className={styles.container}>
        <header className={styles.header}>
          <h1><Translate id="promotion.title">Community Promotion 🚀</Translate></h1>
          <p className={styles.subtitle}>
            <Translate id="promotion.subtitle">Contributor advancement rules - Recognizing your contributions and elevating your impact</Translate>
          </p>
        </header>

        <main className={styles.main}>
          <section className={styles.overview}>
            <h2>
              📖
              <Translate id="promotion.overview.title">Promotion Overview</Translate>
            </h2>
            <div className={styles.overviewContent}>
              <div className={styles.overviewCard}>
                <h3>🎯 Promotion Timing</h3>
                <p>
                  Promotions occur after each release, with
                  <strong> 2-3 month</strong>
                  {' '}
                  intervals between releases
                </p>
              </div>
              <div className={styles.overviewCard}>
                <h3>🏆 Promotion Principles</h3>
                <p>Evaluated based on sustained contributions, technical capabilities, and community engagement</p>
              </div>
              <div className={styles.overviewCard}>
                <h3>📈 Growth Path</h3>
                <div className={styles.growthPathSimple}>
                  <span className={styles.pathText}>
                    <strong>Reviewer</strong>
                    {' '}
                    →
                    <strong>Committer</strong>
                    {' '}
                    →
                    <strong>Maintainer</strong>
                  </span>
                  <p className={styles.pathDescription}>
                    Progressive advancement through sustained contributions and community engagement
                  </p>
                </div>
              </div>
            </div>
          </section>

          <section className={styles.promotionRules}>
            <h2>
              📊
              <Translate id="promotion.rules.title">Promotion Rules</Translate>
            </h2>
            <p className={styles.rulesDescription}>
              Detailed requirements and permissions for each role. Each role builds upon the previous one with increasing responsibilities and impact.
            </p>
            <div className={styles.rulesGrid}>
              {promotionRules.map((rule, index) => (
                <PromotionCard key={index} rule={rule} />
              ))}
            </div>
          </section>

          <section className={styles.applicationProcess}>
            <h2>
              📋
              <Translate id="promotion.process.title">Application Process</Translate>
            </h2>
            <div className={styles.processSteps}>
              <div className={styles.step}>
                <div className={styles.stepNumber}>1</div>
                <div className={styles.stepContent}>
                  <h3>Self-Assessment</h3>
                  <p>Confirm you meet the contribution requirements for the desired role</p>
                </div>
              </div>
              <div className={styles.step}>
                <div className={styles.stepNumber}>2</div>
                <div className={styles.stepContent}>
                  <h3>Submit Application</h3>
                  <p>After a release, create a GitHub Issue to apply for the corresponding role</p>
                </div>
              </div>
              <div className={styles.step}>
                <div className={styles.stepNumber}>3</div>
                <div className={styles.stepContent}>
                  <h3>Community Review</h3>
                  <p>Existing maintainer team will evaluate your contributions</p>
                </div>
              </div>
              <div className={styles.step}>
                <div className={styles.stepNumber}>4</div>
                <div className={styles.stepContent}>
                  <h3>Permission Grant</h3>
                  <p>Upon approval, you'll receive the corresponding GitHub permissions</p>
                </div>
              </div>
            </div>
          </section>

          <section className={styles.getStarted}>
            <h2>
              🚀
              <Translate id="promotion.getStarted.title">Get Started</Translate>
            </h2>
            <p>Ready to begin your contribution journey?</p>
            <div className={styles.actionButtons}>
              <a href="/community/work-groups" className={styles.actionButton}>
                🏷️ View Work Groups
              </a>
              <a href="/community/contributing" className={styles.actionButton}>
                📖 Contributing Guide
              </a>
              <a href="https://github.com/vllm-project/semantic-router/issues" target="_blank" rel="noopener noreferrer" className={styles.actionButton}>
                📝 Submit Application
              </a>
            </div>
          </section>
        </main>
      </div>
    </Layout>
  )
}

export default Promotion
