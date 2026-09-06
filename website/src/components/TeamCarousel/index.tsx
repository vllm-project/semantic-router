import React from 'react'
import Translate from '@docusaurus/Translate'
import { PillLink, SectionLabel } from '@site/src/components/site/Chrome'
import {
  committerMembers,
  getTeamMemberBadge,
  maintainerMembers,
  type TeamMember,
} from '@site/src/data/teamMembers'
import styles from './styles.module.css'

const teamMembers = [...maintainerMembers, ...committerMembers]

function revealFocusedCard(event: React.FocusEvent<HTMLElement>): void {
  const card = event.currentTarget.closest('article')
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches

  card?.scrollIntoView({
    behavior: prefersReducedMotion ? 'auto' : 'smooth',
    block: 'nearest',
    inline: 'center',
  })
}

function MemberCard({
  member,
}: {
  member: TeamMember
}): JSX.Element {
  return (
    <article
      className={styles.memberCard}
      aria-label={member.name}
      tabIndex={0}
      onFocus={revealFocusedCard}
    >
      <div className={styles.avatarWrapper}>
        <img
          src={member.avatar}
          alt=""
          className={styles.avatar}
          loading="lazy"
        />
        <span className={`${styles.badge} ${styles[member.memberType]}`}>
          {getTeamMemberBadge(member)}
        </span>
      </div>
      <h3 className={styles.memberName}>{member.name}</h3>
      <p className={styles.memberRole}>
        {member.role}
        {member.company && (
          <span className={styles.company}>
            {' '}
            @
            {member.company}
          </span>
        )}
      </p>
    </article>
  )
}

function MemberSequence(): JSX.Element {
  return (
    <div className={styles.sequence}>
      {teamMembers.map((member, index) => (
        <MemberCard
          key={`${member.name}-${index}`}
          member={member}
        />
      ))}
    </div>
  )
}

const TeamCarousel: React.FC = () => {
  return (
    <section className={styles.teamSection} aria-labelledby="team-carousel-title">
      <div className="site-shell-container">
        <div className={styles.teamHeader}>
          <SectionLabel>
            <Translate id="teamCarousel.label">Community</Translate>
          </SectionLabel>
          <h2 className={styles.title} id="team-carousel-title">
            <Translate id="teamCarousel.title">Built in the open</Translate>
          </h2>
          <p className={styles.subtitle}>
            <Translate id="teamCarousel.subtitle">
              Maintainers across research, infrastructure, and model systems shape the project together.
            </Translate>
          </p>
        </div>

        <div className={styles.carouselShell}>
          <div className={styles.viewport}>
            <div className={styles.track}>
              <MemberSequence />
            </div>
          </div>
        </div>

        <div className={styles.teamFooter}>
          <p>
            <Translate id="teamCarousel.footer">
              Meet the people turning Mixture-of-Models into shared infrastructure.
            </Translate>
          </p>
          <PillLink to="/community/team" muted>
            <Translate id="teamCarousel.viewAll">View All Team Members</Translate>
          </PillLink>
        </div>
      </div>
    </section>
  )
}

export default TeamCarousel
