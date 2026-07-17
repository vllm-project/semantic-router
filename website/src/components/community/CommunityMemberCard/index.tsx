import type { ReactNode } from 'react'
import { FaExternalLinkAlt, FaGithub, FaLinkedin } from 'react-icons/fa'
import clsx from 'clsx'
import React from 'react'
import {
  getTeamMemberBadge,
  type TeamMember,
} from '@site/src/data/teamMembers'
import styles from './styles.module.css'

interface CommunityMemberCardProps {
  member: TeamMember
  badgeContext?: 'team' | 'steering'
  variant?: 'default' | 'featured' | 'muted'
}

export default function CommunityMemberCard({
  member,
  badgeContext = 'team',
  variant = 'default',
}: CommunityMemberCardProps): ReactNode {
  return (
    <article
      className={clsx(styles.card, {
        [styles.featured]: variant === 'featured',
        [styles.muted]: variant === 'muted',
      })}
    >
      <header className={styles.header}>
        <img
          src={member.avatar}
          alt=""
          className={styles.avatar}
          loading="lazy"
        />
        <div className={styles.identity}>
          <div className={styles.nameLine}>
            <h3>{member.name}</h3>
            <span
              className={clsx(
                styles.badge,
                styles[badgeContext === 'steering' ? 'steering' : member.memberType],
              )}
            >
              {getTeamMemberBadge(member, badgeContext)}
            </span>
          </div>
          <p>
            {member.role}
            {member.company && (
              <span>
                {' @'}
                {member.company}
              </span>
            )}
          </p>
        </div>
      </header>

      <p className={styles.bio}>{member.bio}</p>

      {member.expertise && (
        <ul className={styles.expertise} aria-label="Areas of expertise">
          {member.expertise.map((expertise, index) => (
            <li key={index}>{expertise}</li>
          ))}
        </ul>
      )}

      <div className={styles.links}>
        {member.github && member.github !== '#' && (
          <MemberLink href={member.github} icon={<FaGithub />}>
            GitHub
          </MemberLink>
        )}
        {member.linkedin && (
          <MemberLink href={member.linkedin} icon={<FaLinkedin />}>
            LinkedIn
          </MemberLink>
        )}
        {member.externalLinks?.map(link => (
          <MemberLink
            key={link.href}
            href={link.href}
            icon={<FaExternalLinkAlt />}
          >
            {link.label}
          </MemberLink>
        ))}
      </div>
    </article>
  )
}

function MemberLink({
  href,
  icon,
  children,
}: {
  href: string
  icon: ReactNode
  children: ReactNode
}): ReactNode {
  return (
    <a href={href} target="_blank" rel="noopener noreferrer">
      {icon}
      <span>{children}</span>
      <span aria-hidden="true">↗</span>
    </a>
  )
}
