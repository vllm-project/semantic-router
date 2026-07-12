import React from 'react'
import Translate from '@docusaurus/Translate'
import { PillLink, SectionLabel } from '@site/src/components/site/Chrome'
import { committerMembers, steeringCommitteeMembers } from '@site/src/data/teamMembers'
import styles from './styles.module.css'

const teamMembers = [...steeringCommitteeMembers, ...committerMembers]

const TeamCarousel: React.FC = () => {
  return (
    <section className={styles.teamSection}>
      <div className="site-shell-container">
        <div className={styles.teamHeader}>
          <div>
            <SectionLabel>
              <Translate id="teamCarousel.label">Community</Translate>
            </SectionLabel>
            <h2 className={styles.title}>
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
          <div className={styles.carouselContainer}>
            <div className={styles.carouselTrack}>
              {teamMembers.map((member, index) => (
                <article key={`${member.name}-${index}`} className={styles.memberCard}>
                  <div className={styles.avatarWrapper}>
                    <img
                      src={member.avatar}
                      alt={member.name}
                      className={styles.avatar}
                    />
                    <span className={`${styles.badge} ${styles[member.memberType]}`}>
                      {member.memberType === 'steering'
                        ? <Translate id="team.badge.steering">Steering Committee</Translate>
                        : <Translate id="team.badge.committer">Committer</Translate>}
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
              ))}
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
