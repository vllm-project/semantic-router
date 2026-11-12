import React, { useEffect, useRef, useState } from 'react'
import clsx from 'clsx'
import Link from '@docusaurus/Link'
import Translate, { translate } from '@docusaurus/Translate'
import { SectionLabel } from '@site/src/components/site/Chrome'
import usePauseOffscreen from '@site/src/components/duo/usePauseOffscreen'
import {
  committerMembers,
  emeritusCommitterMembers,
  maintainerMembers,
  type TeamMember,
} from '@site/src/data/teamMembers'
import duo from '@site/src/components/duo/duo.module.css'
import styles from './styles.module.css'

const ROWS = 3
const people = [...maintainerMembers, ...committerMembers, ...emeritusCommitterMembers]
const rows = Array.from({ length: ROWS }, (_, row) => people.filter((_, index) => index % ROWS === row))

const counts = {
  maintainer: maintainerMembers.length,
  committer: committerMembers.length,
  emeritus: emeritusCommitterMembers.length,
}

// GitHub serves any size of its avatars; the wall shows them at 44 px.
function avatarSrc(avatar: string): string {
  return /^https:\/\/github\.com\/[^/?#]+\.png$/.test(avatar) ? `${avatar}?size=96` : avatar
}

function RoleTag({ member }: { member: TeamMember }): JSX.Element {
  return (
    <span className={clsx(styles.tag, styles[member.memberType])}>
      <i aria-hidden="true" />
      {member.memberType === 'emeritus'
        ? <Translate id="teamCarousel.tag.emeritus">Emeritus</Translate>
        : member.memberType === 'committer'
          ? <Translate id="teamCarousel.tag.committer">Committer</Translate>
          : <Translate id="teamCarousel.tag.maintainer">Maintainer</Translate>}
    </span>
  )
}

function Portrait({ member, copy, near }: { member: TeamMember, copy?: boolean, near: boolean }): JSX.Element {
  return (
    <li className={clsx(styles.person, copy && styles.copy)} aria-hidden={copy || undefined}>
      <img src={avatarSrc(member.avatar)} alt="" width={44} height={44} loading={near ? 'eager' : 'lazy'} decoding="async" className={styles.avatar} />
      <span className={styles.who}>
        <span className={styles.name}>{member.name}</span>
        <RoleTag member={member} />
      </span>
    </li>
  )
}

/* The maintainers, committers and emeritus committers as a slowly drifting wall: three rows in opposite directions that
 * pause on hover, focus, the Pause button and off screen, and stand still under reduced motion. */
const TeamCarousel: React.FC = () => {
  const sectionRef = useRef<HTMLElement>(null)
  const [paused, setPaused] = useState(false)
  const [near, setNear] = useState(false)
  usePauseOffscreen(sectionRef)

  // Portraits drift in from clipped edges, where lazy loading would leave them blank; load them all as the wall nears.
  useEffect(() => {
    const element = sectionRef.current
    if (!element) return undefined
    const observer = new IntersectionObserver((entries) => {
      if (entries.some(entry => entry.isIntersecting)) {
        setNear(true)
        observer.disconnect()
      }
    }, { rootMargin: '800px 0px' })
    observer.observe(element)
    return () => observer.disconnect()
  }, [])

  const toggleLabel = paused
    ? translate({ id: 'teamCarousel.play', message: 'Play the wall' })
    : translate({ id: 'teamCarousel.pause', message: 'Pause the wall' })

  return (
    <section
      ref={sectionRef}
      className={styles.teamSection}
      aria-labelledby="team-carousel-title"
      data-paused={paused ? '' : undefined}
    >
      <div className={styles.wrap}>
        <header className="site-section-intro">
          <SectionLabel>
            <Translate id="teamCarousel.label">Community</Translate>
          </SectionLabel>
          <h2 id="team-carousel-title">
            <Translate id="teamCarousel.title">Built in the open</Translate>
          </h2>
          <p>
            <Translate id="teamCarousel.subtitle">
              Researchers and builders, working together.
            </Translate>
          </p>
        </header>
      </div>

      <div className={styles.wall}>
        {rows.map((row, index) => (
          <div
            key={index}
            className={styles.row}
            data-dir={index % 2 ? 'right' : 'left'}
            style={{ '--dur': `${row.length * (6 + index)}s` } as React.CSSProperties}
          >
            <ul className={styles.track}>
              {row.map(member => <Portrait key={member.name} member={member} near={near} />)}
              {row.map(member => <Portrait key={`${member.name}-copy`} member={member} copy near={near} />)}
            </ul>
          </div>
        ))}
      </div>

      <div className={styles.wrap}>
        <div className={styles.teamFooter}>
          <p className={styles.legend}>
            <span className={clsx(styles.tag, styles.maintainer)}>
              <i aria-hidden="true" />
              <Translate id="teamCarousel.count.maintainers" values={{ count: counts.maintainer }}>{'{count} maintainers'}</Translate>
            </span>
            <span className={clsx(styles.tag, styles.committer)}>
              <i aria-hidden="true" />
              <Translate id="teamCarousel.count.committers" values={{ count: counts.committer }}>{'{count} committers'}</Translate>
            </span>
            <span className={clsx(styles.tag, styles.emeritus)}>
              <i aria-hidden="true" />
              <Translate id="teamCarousel.count.emeritus" values={{ count: counts.emeritus }}>{'{count} emeritus'}</Translate>
            </span>
          </p>
          <div className={styles.actions}>
            <button
              type="button"
              className={styles.toggle}
              onClick={() => setPaused(value => !value)}
              aria-label={toggleLabel}
              title={toggleLabel}
            >
              <svg viewBox="0 0 8 8" aria-hidden="true" shapeRendering="crispEdges">
                {paused
                  ? <path d="M2 1h1v6H2zM3 2h1v4H3zM4 3h1v2H4z" fill="currentColor" />
                  : <path d="M2 1h1v6H2zM5 1h1v6H5z" fill="currentColor" />}
              </svg>
              <span aria-hidden="true">
                {paused
                  ? <Translate id="teamCarousel.playShort">Play</Translate>
                  : <Translate id="teamCarousel.pauseShort">Pause</Translate>}
              </span>
            </button>
            <Link className={duo.more} to="/community/team">
              <Translate id="teamCarousel.viewAll">Meet the team</Translate>
            </Link>
          </div>
        </div>
      </div>
    </section>
  )
}

export default TeamCarousel
