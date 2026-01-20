import React from 'react'
import Layout from '@theme/Layout'
import Translate from '@docusaurus/Translate'
import Link from '@docusaurus/Link'
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
  const ruleId = rule.role.toLowerCase()

  return (
    <div className={styles.promotionCard} style={{ borderColor: rule.color }}>
      <div className={styles.cardHeader}>
        <span className={styles.roleIcon}>{rule.icon}</span>
        <h3 className={styles.roleTitle} style={{ color: rule.color }}>
          <Translate id={`promotion.role.${ruleId}.title`}>{rule.role}</Translate>
        </h3>
        <span className={styles.permissions} style={{ backgroundColor: rule.color }}>
          <Translate id={`promotion.role.${ruleId}.permissions`}>{rule.permissions}</Translate>
        </span>
      </div>

      <div className={styles.cardContent}>
        <div className={styles.requirements}>
          <h4><Translate id="promotion.card.requirements">📋 Requirements</Translate></h4>
          <p className={styles.mainRequirement}>
            <Translate id={`promotion.role.${ruleId}.requirements`}>{rule.requirements}</Translate>
          </p>
          <ul className={styles.detailsList}>
            {rule.details.map((detail, index) => (
              <li key={index}>
                <Translate id={`promotion.role.${ruleId}.details.${index}`}>{detail}</Translate>
              </li>
            ))}
          </ul>
        </div>

        <div className={styles.timeline}>
          <h4><Translate id="promotion.card.timeline">⏰ Timeline</Translate></h4>
          <p>
            <Translate id={`promotion.role.${ruleId}.timeline`}>{rule.timeline}</Translate>
          </p>
        </div>

        <div className={styles.application}>
          <h4><Translate id="promotion.card.application">📝 How to Apply</Translate></h4>
          <p>
            <Translate id={`promotion.role.${ruleId}.application`}>{rule.application}</Translate>
          </p>
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
                <h3><Translate id="promotion.timing.title">🎯 Promotion Timing</Translate></h3>
                <p>
                  <Translate id="promotion.timing.desc.prefix">Promotions occur after each release, with</Translate>
                  <strong>
                    {' '}
                    <Translate id="promotion.timing.desc.strong">2-3 month</Translate>
                  </strong>
                  {' '}
                  <Translate id="promotion.timing.desc.suffix">intervals between releases</Translate>
                </p>
              </div>
              <div className={styles.overviewCard}>
                <h3><Translate id="promotion.principles.title">🏆 Promotion Principles</Translate></h3>
                <p><Translate id="promotion.principles.desc">Evaluated based on sustained contributions, technical capabilities, and community engagement</Translate></p>
              </div>
              <div className={styles.overviewCard}>
                <h3><Translate id="promotion.growth.title">📈 Growth Path</Translate></h3>
                <div className={styles.growthPathSimple}>
                  <span className={styles.pathText}>
                    <strong><Translate id="promotion.role.reviewer.title">Reviewer</Translate></strong>
                    {' '}
                    →
                    <strong><Translate id="promotion.role.committer.title">Committer</Translate></strong>
                    {' '}
                    →
                    <strong><Translate id="promotion.role.maintainer.title">Maintainer</Translate></strong>
                  </span>
                  <p className={styles.pathDescription}>
                    <Translate id="promotion.growth.desc">Progressive advancement through sustained contributions and community engagement</Translate>
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
              <Translate id="promotion.rules.description">
                Detailed requirements and permissions for each role. Each role builds upon the previous one with increasing responsibilities and impact.
              </Translate>
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
                  <h3><Translate id="promotion.process.step1.title">Self-Assessment</Translate></h3>
                  <p><Translate id="promotion.process.step1.desc">Confirm you meet the contribution requirements for the desired role</Translate></p>
                </div>
              </div>
              <div className={styles.step}>
                <div className={styles.stepNumber}>2</div>
                <div className={styles.stepContent}>
                  <h3><Translate id="promotion.process.step2.title">Submit Application</Translate></h3>
                  <p><Translate id="promotion.process.step2.desc">After a release, create a GitHub Issue to apply for the corresponding role</Translate></p>
                </div>
              </div>
              <div className={styles.step}>
                <div className={styles.stepNumber}>3</div>
                <div className={styles.stepContent}>
                  <h3><Translate id="promotion.process.step3.title">Community Review</Translate></h3>
                  <p><Translate id="promotion.process.step3.desc">Existing maintainer team will evaluate your contributions</Translate></p>
                </div>
              </div>
              <div className={styles.step}>
                <div className={styles.stepNumber}>4</div>
                <div className={styles.stepContent}>
                  <h3><Translate id="promotion.process.step4.title">Permission Grant</Translate></h3>
                  <p><Translate id="promotion.process.step4.desc">Upon approval, you'll receive the corresponding GitHub permissions</Translate></p>
                </div>
              </div>
            </div>
          </section>

          <section className={styles.getStarted}>
            <h2>
              🚀
              <Translate id="promotion.getStarted.title">Get Started</Translate>
            </h2>
            <p><Translate id="promotion.getStarted.desc">Ready to begin your contribution journey?</Translate></p>
            <div className={styles.actionButtons}>
              <Link to="/community/work-groups" className={styles.actionButton}>
                <Translate id="promotion.getStarted.workGroups">🏷️ View Work Groups</Translate>
              </Link>
              <Link to="/community/contributing" className={styles.actionButton}>
                <Translate id="promotion.getStarted.contributing">📖 Contributing Guide</Translate>
              </Link>
              <a href="https://github.com/vllm-project/semantic-router/issues" target="_blank" rel="noopener noreferrer" className={styles.actionButton}>
                <Translate id="promotion.getStarted.apply">📝 Submit Application</Translate>
              </a>
            </div>
          </section>
        </main>
      </div>
    </Layout>
  )
}

export default Promotion
