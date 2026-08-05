import React from 'react'
import Link from '@docusaurus/Link'
import Translate from '@docusaurus/Translate'
import useDocusaurusContext from '@docusaurus/useDocusaurusContext'
import { PillLink, SectionLabel } from '@site/src/components/site/Chrome'
import acknowledgementsData from '@site/src/components/AcknowledgementsSection/data.json'
import {
  committerMembers,
  getTeamMemberBadge,
  maintainerMembers,
  type TeamMember,
} from '@site/src/data/teamMembers'
import shared from './homepageShared.module.css'
import styles from './HomepageCommunity.module.css'

type AckProject = {
  id: string
  name: string
  logo: string
  url: string
}

const projects = (acknowledgementsData.projects ?? []) as AckProject[]
const allTeamMembers: TeamMember[] = [...maintainerMembers, ...committerMembers]
const marqueeCopies = [0, 1]

function toAssetUrl(baseUrl: string, path: string): string {
  if (path.startsWith('http')) {
    return path
  }
  const normalizedBase = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`
  const normalizedPath = path.startsWith('/') ? path.slice(1) : path
  return `${normalizedBase}${normalizedPath}`
}

function DependencyChip({
  project,
  assetBase,
}: {
  project: AckProject
  assetBase: string
}): JSX.Element {
  return (
    <li>
      <a
        href={project.url}
        target="_blank"
        rel="noopener noreferrer"
        className={styles.dependencyChip}
        title={project.name}
      >
        <img
          src={toAssetUrl(assetBase, project.logo)}
          alt=""
          className={styles.dependencyLogo}
        />
        <span>{project.name}</span>
      </a>
    </li>
  )
}

function MemberCard({
  member,
  assetBase,
}: {
  member: TeamMember
  assetBase: string
}): JSX.Element {
  return (
    <li className={styles.memberItem}>
      <Link className={styles.memberCard} to="/community/team">
        <div className={styles.avatarWrap}>
          <img
            src={toAssetUrl(assetBase, member.avatar)}
            alt=""
            className={styles.memberAvatar}
          />
          <span className={`${styles.memberBadge} ${styles[member.memberType]}`}>
            {getTeamMemberBadge(member)}
          </span>
        </div>
        <div className={styles.memberCopy}>
          <strong>{member.name}</strong>
          <span className={styles.memberRole}>{member.role}</span>
          {member.company && (
            <span className={styles.memberCompany}>
              @
              {member.company}
            </span>
          )}
        </div>
      </Link>
    </li>
  )
}

export default function HomepageCommunity(): JSX.Element {
  const { siteConfig } = useDocusaurusContext()
  const assetBase = siteConfig.baseUrl

  return (
    <section className={shared.lightSection} aria-labelledby="homepage-community-title">
      <div className={`site-shell-container ${shared.sectionInner}`}>
        <header className={shared.sectionHeader}>
          <SectionLabel>
            <Translate id="homepage.community.label">Community</Translate>
          </SectionLabel>
          <h2 id="homepage-community-title" className={shared.sectionTitle}>
            <Translate id="homepage.community.title">Built with the ecosystem</Translate>
          </h2>
          <p className={shared.sectionSubtitle}>
            <Translate id="homepage.community.subtitle">Open-source dependencies and the team behind Semantic Router.</Translate>
          </p>
        </header>

        <div className={styles.block}>
          <div className={styles.blockHeading}>
            <h3>
              <Translate id="homepage.community.dependencies.title">Dependencies</Translate>
            </h3>
            <p>
              <Translate id="homepage.community.dependencies.subtitle">
                Core projects the router builds on.
              </Translate>
            </p>
          </div>

          <div className={styles.dependencyMarquee}>
            <div className={styles.marqueeViewport} aria-hidden="true">
              <div className={styles.dependencyTrack}>
                {marqueeCopies.map(copyIndex => (
                  <ul key={`dependency-sequence-${copyIndex}`} className={styles.marqueeSequence}>
                    {projects.map(project => (
                      <DependencyChip
                        key={`${copyIndex}-${project.id}`}
                        project={project}
                        assetBase={assetBase}
                      />
                    ))}
                  </ul>
                ))}
              </div>
            </div>
          </div>
        </div>

        <div className={styles.block}>
          <div className={styles.blockHeading}>
            <h3>
              <Translate id="homepage.community.team.title">Team</Translate>
            </h3>
            <p>
              <Translate id="homepage.community.team.subtitle">
                Maintainers and active committers across research, infrastructure, and model systems.
              </Translate>
            </p>
          </div>

          <div className={styles.teamMarquee}>
            <div className={styles.marqueeViewport}>
              <ul className={styles.teamTrack}>
                {marqueeCopies.map(copyIndex =>
                  allTeamMembers.map(member => (
                    <MemberCard
                      key={`${copyIndex}-${member.name}`}
                      member={member}
                      assetBase={assetBase}
                    />
                  )),
                )}
              </ul>
            </div>
          </div>

          <div className={styles.blockFooter}>
            <PillLink to="/community/team" muted>
              <Translate id="homepage.community.team.cta">View all team members</Translate>
            </PillLink>
            <PillLink to="/community/contributors" muted>
              <Translate id="homepage.community.contributors.cta">Contributor leaderboard</Translate>
            </PillLink>
          </div>
        </div>
      </div>
    </section>
  )
}
