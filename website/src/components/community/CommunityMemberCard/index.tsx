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
      <div className={styles.row}>
        <img
          src={member.avatar}
          alt=""
          className={styles.avatar}
          loading="lazy"
        />
        <div className={styles.identity}>
          <h3>{member.name}</h3>
          <span
            className={clsx(
              styles.badge,
              styles[badgeContext === 'steering' ? 'steering' : member.memberType],
            )}
          >
            {getTeamMemberBadge(member, badgeContext)}
          </span>
          <span className={styles.role}>
            {member.role}
            {member.company && (
              <span>
                {' @'}
                {member.company}
              </span>
            )}
          </span>
        </div>

        <div className={styles.links}>
          {member.github && member.github !== '#' && (
            <MemberLink href={member.github} label="GitHub" icon={<FaGithub />} />
          )}
          {member.linkedin && (
            <MemberLink href={member.linkedin} label="LinkedIn" icon={<FaLinkedin />} />
          )}
          {member.externalLinks?.map(link => (
            <MemberLink
              key={link.href}
              href={link.href}
              label={link.label}
              icon={<FaExternalLinkAlt />}
            />
          ))}
        </div>
      </div>

      <p className={styles.bio}>{member.bio}</p>
    </article>
  )
}

function MemberLink({
  href,
  icon,
  label,
}: {
  href: string
  icon: ReactNode
  label: string
}): ReactNode {
  return (
    <a href={href} target="_blank" rel="noopener noreferrer" aria-label={label} title={label}>
      {icon}
    </a>
  )
}
