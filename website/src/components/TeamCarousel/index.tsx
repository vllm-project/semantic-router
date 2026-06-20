import React, { useEffect, useRef } from 'react'
import Translate from '@docusaurus/Translate'
import { PillLink, SectionLabel } from '@site/src/components/site/Chrome'
import { committerMembers, steeringCommitteeMembers } from '@site/src/data/teamMembers'
import styles from './styles.module.css'

const teamMembers = [...steeringCommitteeMembers, ...committerMembers]

const TeamCarousel: React.FC = () => {
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const scrollContainer = scrollRef.current
    if (!scrollContainer) return
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return

    let animationFrameId: number
    let scrollPosition = 0
    const scrollSpeed = 0.5 // pixels per frame
    let totalWidth = 0

    const updateTotalWidth = () => {
      const cards = Array.from(scrollContainer.children).slice(0, teamMembers.length)
      const gap = parseFloat(window.getComputedStyle(scrollContainer).gap || '0')

      totalWidth = cards.reduce((total, card, index) => {
        const width = (card as HTMLElement).offsetWidth
        return total + width + (index < cards.length - 1 ? gap : 0)
      }, 0)
    }

    updateTotalWidth()
    window.addEventListener('resize', updateTotalWidth)

    const scroll = () => {
      scrollPosition += scrollSpeed

      if (scrollPosition >= totalWidth) {
        scrollPosition = 0
      }

      if (scrollContainer) {
        scrollContainer.style.transform = `translateX(-${scrollPosition}px)`
      }

      animationFrameId = requestAnimationFrame(scroll)
    }

    animationFrameId = requestAnimationFrame(scroll)

    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId)
      }
      window.removeEventListener('resize', updateTotalWidth)
    }
  }, [])

  // Duplicate members for infinite scroll effect
  const duplicatedMembers = [...teamMembers, ...teamMembers, ...teamMembers]

  return (
    <section className={styles.teamSection}>
      <div className="site-shell-container">
        <div className={styles.teamHeader}>
          <div>
            <SectionLabel>
              <Translate id="teamCarousel.label">Governance</Translate>
            </SectionLabel>
            <h2 className={styles.title}>
              <Translate id="teamCarousel.title">Meet Our Team</Translate>
            </h2>
          </div>
          <p className={styles.subtitle}>
            <Translate id="teamCarousel.subtitle">Innovation thrives when great minds come together</Translate>
          </p>
        </div>

        <div className={styles.carouselShell}>
          <div className={styles.carouselContainer}>
            <div className={styles.carouselTrack} ref={scrollRef}>
              {duplicatedMembers.map((member, index) => (
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
              Steering committee members and committers across research, infrastructure, and open-source operations.
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
