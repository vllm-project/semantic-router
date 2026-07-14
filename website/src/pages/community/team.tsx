import React from 'react'
import Layout from '@theme/Layout'
import Translate from '@docusaurus/Translate'
import Link from '@docusaurus/Link'
import { FaExternalLinkAlt, FaGithub, FaLinkedin } from 'react-icons/fa'
import {
  academicTrackMembers,
  committerMembers,
  getTeamMemberBadge,
  industryTrackMembers,
  type TeamMember,
} from '@site/src/data/teamMembers'
import styles from './team.module.css'

interface TeamMemberProps {
  member: TeamMember
}

const TeamMemberCard: React.FC<TeamMemberProps> = ({ member }) => {
  return (
    <div className={styles.memberCard}>
      <div className={styles.memberHeader}>
        <img
          src={member.avatar}
          alt={`${member.name} avatar`}
          className={styles.avatar}
        />
        <div className={styles.memberInfo}>
          <div className={styles.nameWithBadge}>
            <h3 className={styles.memberName}>{member.name}</h3>
            <span className={`${styles.badge} ${styles[member.memberType]}`}>
              {getTeamMemberBadge(member)}
            </span>
          </div>
          <p className={styles.memberRole}>
            {member.role}
            {member.company && (
              <span className={styles.company}>
                {' @'}
                {member.company}
              </span>
            )}
          </p>
        </div>
      </div>

      <p className={styles.memberBio}>{member.bio}</p>

      {member.expertise && (
        <ul className={styles.expertiseList}>
          {member.expertise.map((expertise, index) => (
            <li key={index}>{expertise}</li>
          ))}
        </ul>
      )}

      <div className={styles.memberActions}>
        {member.github && member.github !== '#' && (
          <a
            href={member.github}
            target="_blank"
            rel="noopener noreferrer"
            className={styles.actionLink}
          >
            <FaGithub />
            GitHub
          </a>
        )}

        {member.linkedin && (
          <a
            href={member.linkedin}
            target="_blank"
            rel="noopener noreferrer"
            className={styles.actionLink}
          >
            <FaLinkedin />
            LinkedIn
          </a>
        )}

        {member.externalLinks?.map(link => (
          <a
            key={link.href}
            href={link.href}
            target="_blank"
            rel="noopener noreferrer"
            className={styles.actionLink}
          >
            <FaExternalLinkAlt />
            {link.label}
          </a>
        ))}
      </div>
    </div>
  )
}

const Team: React.FC = () => {
  return (
    <Layout
      title="Team"
      description="Meet the team behind vLLM Semantic Router"
    >
      <div className={styles.container}>
        <header className={styles.header}>
          <h1><Translate id="team.title">Meet Our Team</Translate></h1>
          <p className={styles.subtitle}>
            <Translate id="team.subtitle">Innovation thrives when great minds come together</Translate>
          </p>
        </header>

        <main className={styles.main}>
          <section className={styles.section}>
            <h2>
              <Translate id="team.steering.title">Leadership Tracks</Translate>
            </h2>
            <p className={styles.sectionDescription}>
              <Translate id="team.steering.description">
                Project leadership is organized across industry and academic tracks, with shared direction shaped at the boundary between engineering and research.
              </Translate>
            </p>
            <div className={styles.trackList}>
              <section className={styles.trackSection} aria-labelledby="industry-track-title">
                <header className={styles.trackIntro}>
                  <span className={styles.trackIndex}>01 / Industry</span>
                  <h3 id="industry-track-title">
                    <Translate id="team.track.industry.title">Industry Track</Translate>
                  </h3>
                  <p className={styles.trackDescription}>
                    <Translate id="team.track.industry.description">
                      Leads the project's engineering direction and works with the research side to decide where frontier research and industrial practice meet, including the project's overall engineering evolution.
                    </Translate>
                  </p>
                </header>
                <div className={styles.trackMembers}>
                  {industryTrackMembers.map(member => (
                    <TeamMemberCard key={member.name} member={member} />
                  ))}
                </div>
              </section>

              <section className={styles.trackSection} aria-labelledby="academic-track-title">
                <header className={styles.trackIntro}>
                  <span className={styles.trackIndex}>02 / Academic</span>
                  <h3 id="academic-track-title">
                    <Translate id="team.track.academic.title">Academic Track</Translate>
                  </h3>
                  <p className={styles.trackDescription}>
                    <Translate id="team.track.academic.description">
                      Advances frontier academic research around the project and works closely with the engineering side to turn research into shared project progress.
                    </Translate>
                  </p>
                </header>
                <div className={styles.trackMembers}>
                  {academicTrackMembers.map(member => (
                    <TeamMemberCard key={member.name} member={member} />
                  ))}
                </div>
              </section>
            </div>
          </section>

          <section className={styles.section}>
            <h2>
              <Translate id="team.committers.title">Committers</Translate>
            </h2>
            <p className={styles.sectionDescription}>
              <Translate id="team.committers.description">
                Committers own implementation areas, review changes, answer community questions, and keep the project healthy across releases.
              </Translate>
            </p>
            <div className={styles.teamGrid}>
              {committerMembers.map((member, index) => (
                <TeamMemberCard key={index} member={member} />
              ))}
            </div>
          </section>

          <section className={styles.section}>
            <h2>
              <Translate id="team.getInvolved.title">Get Involved</Translate>
            </h2>
            <div className={styles.involvementGrid}>
              <div className={styles.involvementCard}>
                <h3>
                  <Translate id="team.getInvolved.contribute.title">Start Contributing</Translate>
                </h3>
                <p><Translate id="team.getInvolved.contribute.desc">Ready to make your first contribution?</Translate></p>
                <Link to="/community/contributing" className={styles.actionButton}>
                  <Translate id="team.getInvolved.contribute.link">Contributing Guide</Translate>
                </Link>
              </div>

              <div className={styles.involvementCard}>
                <h3>
                  <Translate id="team.getInvolved.workGroups.title">Join Working Groups</Translate>
                </h3>
                <p><Translate id="team.getInvolved.workGroups.desc">Find your area of expertise and connect with like-minded contributors.</Translate></p>
                <Link to="/community/work-groups" className={styles.actionButton}>
                  <Translate id="team.getInvolved.workGroups.link">View Work Groups</Translate>
                </Link>
              </div>

              <div className={styles.involvementCard}>
                <h3>
                  <Translate id="team.getInvolved.discussions.title">Join Discussions</Translate>
                </h3>
                <p><Translate id="team.getInvolved.discussions.desc">Participate in community discussions and share your ideas.</Translate></p>
                <a href="https://github.com/vllm-project/semantic-router/discussions" target="_blank" rel="noopener noreferrer" className={styles.actionButton}>
                  <Translate id="team.getInvolved.discussions.link">GitHub Discussions</Translate>
                </a>
              </div>
            </div>
          </section>
        </main>
      </div>
    </Layout>
  )
}

export default Team
