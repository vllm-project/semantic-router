import type { ReactNode } from 'react'
import Link from '@docusaurus/Link'
import Layout from '@theme/Layout'
import React from 'react'
import CommunityMemberCard from '@site/src/components/community/CommunityMemberCard'
import {
  academicTrackMembers,
  industryTrackMembers,
  type TeamMember,
} from '@site/src/data/teamMembers'
import styles from './steering-committee.module.css'

const tracks: Array<{
  index: string
  label: string
  title: string
  description: string
  responsibilities: string[]
  members: TeamMember[]
}> = [
  {
    index: '01',
    label: 'Engineering & industry',
    title: 'Industry Track',
    description:
      'Leads engineering direction and decides where frontier research becomes reliable, production-grade infrastructure.',
    responsibilities: [
      'Project architecture and engineering evolution',
      'Research-to-production priorities',
      'Ecosystem, release, and adoption strategy',
    ],
    members: industryTrackMembers,
  },
  {
    index: '02',
    label: 'Research & academia',
    title: 'Academic Track',
    description:
      'Advances frontier research around semantic routing and works with engineering to turn evidence into shared project direction.',
    responsibilities: [
      'Research agenda and scientific quality',
      'Evaluation, publication, and external collaboration',
      'Translation of research into project capabilities',
    ],
    members: academicTrackMembers,
  },
]

export default function SteeringCommittee(): ReactNode {
  return (
    <Layout
      title="Steering Committee"
      description="The Industry and Academic tracks of the vLLM Semantic Router Steering Committee"
    >
      <div className={styles.container}>
        <header className={styles.hero}>
          <span className={styles.eyebrow}>Community / Governance</span>
          <div className={styles.heroGrid}>
            <div>
              <h1>Steering Committee</h1>
              <p className={styles.lede}>
                One committee, two complementary tracks. The committee aligns
                long-range project direction across engineering practice and
                academic research.
              </p>
            </div>
            <aside className={styles.mandate}>
              <span>Mandate</span>
              <p>
                Set project-level direction, resolve cross-track questions,
                and maintain a productive boundary between research and
                engineering.
              </p>
              <Link to="/community/governance">
                Read the governance model
                <span aria-hidden="true">→</span>
              </Link>
            </aside>
          </div>
        </header>

        <main className={styles.tracks}>
          {tracks.map(track => (
            <TrackSection key={track.index} {...track} />
          ))}
        </main>
      </div>
    </Layout>
  )
}

function TrackSection({
  index,
  label,
  title,
  description,
  responsibilities,
  members,
}: (typeof tracks)[number]): ReactNode {
  return (
    <section className={styles.track} aria-labelledby={`track-${index}`}>
      <header className={styles.trackHeader}>
        <div className={styles.trackNumber}>{index}</div>
        <div className={styles.trackTitle}>
          <span>{label}</span>
          <h2 id={`track-${index}`}>{title}</h2>
          <p>{description}</p>
        </div>
        <ul className={styles.responsibilities}>
          {responsibilities.map((responsibility, itemIndex) => (
            <li key={responsibility}>
              <span>{String(itemIndex + 1).padStart(2, '0')}</span>
              {responsibility}
            </li>
          ))}
        </ul>
      </header>
      <div className={styles.members}>
        {members.map(member => (
          <CommunityMemberCard
            key={member.name}
            member={member}
            badgeContext="steering"
          />
        ))}
      </div>
    </section>
  )
}
