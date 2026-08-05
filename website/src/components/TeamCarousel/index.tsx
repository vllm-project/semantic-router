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

interface MemberSequenceProps {
  duplicate?: boolean
}

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
  duplicate = false,
}: {
  member: TeamMember
  duplicate?: boolean
}): JSX.Element {
  return (
    <article
      className={styles.memberCard}
      aria-label={duplicate ? undefined : `${member.name}`}
      tabIndex={duplicate ? -1 : 0}
      onFocus={duplicate ? undefined : revealFocusedCard}
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

function MemberSequence({ duplicate = false }: MemberSequenceProps): JSX.Element {
  return (
    <div className={styles.sequence} aria-hidden={duplicate || undefined}>
      {teamMembers.map((member, index) => (
        <MemberCard
          key={`${member.name}-${index}`}
          member={member}
          duplicate={duplicate}
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
          <div>
            <SectionLabel>
              <Translate id="teamCarousel.label">Community</Translate>
            </SectionLabel>
            <h2 className={styles.title} id="team-carousel-title">
              <Translate id="teamCarousel.title">Built in the open.</Translate>
            </h2>
          </div>
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
              <MemberSequence duplicate />
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
