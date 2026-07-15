import type { ReactNode } from 'react'
import Link from '@docusaurus/Link'
import Layout from '@theme/Layout'
import React from 'react'
import CommunityMemberCard from '@site/src/components/community/CommunityMemberCard'
import { committerActivityWindow } from '@site/src/data/committerActivity.generated'
import {
  committerMembers,
  emeritusCommitterMembers,
  maintainerMembers,
  type TeamMember,
} from '@site/src/data/teamMembers'
import styles from './team.module.css'

export default function Team(): ReactNode {
  return (
    <Layout
      title="Project Team"
      description="Maintainers, committers, and emeritus committers of vLLM Semantic Router"
    >
      <div className={styles.page}>
        <div className={styles.container}>
          <header className={styles.hero}>
            <span className={styles.eyebrow}>Community / Project Team</span>
            <div className={styles.heroGrid}>
              <h1>Project Team</h1>
              <div className={styles.heroCopy}>
                <p>
                  The people responsible for the health, quality, and continuity
                  of vLLM Semantic Router.
                </p>
                <div className={styles.heroLinks}>
                  <Link to="/community/steering-committee">
                    Steering Committee
                    <span aria-hidden="true">→</span>
                  </Link>
                  <Link to="/community/governance">
                    Roles & governance
                    <span aria-hidden="true">→</span>
                  </Link>
                </div>
              </div>
            </div>
          </header>

          <main className={styles.roster}>
            <RosterSection
              index="01"
              title="Maintainers"
              description="Project-wide owners who set engineering direction, approve releases, resolve technical escalations, and steward repository access."
              members={maintainerMembers}
              variant="featured"
              columns="two"
            />

            <RosterSection
              index="02"
              title="Committers"
              description="Active trusted contributors who review changes, own implementation areas, triage issues, and keep the project moving."
              members={committerMembers}
            />

            <RosterSection
              index="03"
              title="Emeritus Committers"
              description={(
                <>
                  Former active committers recognized for their contributions.
                  This roster is refreshed from GitHub activity over the rolling
                  three-month window from
                  {' '}
                  <time dateTime={committerActivityWindow.cutoffDate}>
                    {committerActivityWindow.cutoffDate}
                  </time>
                  {' '}
                  to
                  {' '}
                  <time dateTime={committerActivityWindow.generatedAt}>
                    {committerActivityWindow.generatedAt}
                  </time>
                  .
                </>
              )}
              members={emeritusCommitterMembers}
              variant="muted"
            />
          </main>
        </div>
      </div>
    </Layout>
  )
}

function RosterSection({
  index,
  title,
  description,
  members,
  variant = 'default',
  columns = 'three',
}: {
  index: string
  title: string
  description: ReactNode
  members: TeamMember[]
  variant?: 'default' | 'featured' | 'muted'
  columns?: 'two' | 'three'
}): ReactNode {
  return (
    <section className={styles.section} aria-labelledby={`roster-${index}`}>
      <header className={styles.sectionHeader}>
        <span>{index}</span>
        <div>
          <div className={styles.titleLine}>
            <h2 id={`roster-${index}`}>{title}</h2>
            <span className={styles.count}>{members.length}</span>
          </div>
          <p>{description}</p>
        </div>
      </header>
      <div
        className={columns === 'two' ? styles.gridTwo : styles.gridThree}
      >
        {members.map(member => (
          <CommunityMemberCard
            key={member.name}
            member={member}
            variant={variant}
          />
        ))}
      </div>
    </section>
  )
}
