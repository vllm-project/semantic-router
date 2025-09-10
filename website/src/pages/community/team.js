import React from 'react'
import Layout from '@theme/Layout'
import styles from './team.module.css'

const coreTeam = [
  {
    name: 'Huamin Chen',
    role: 'Distinguished Engineer',
    company: 'Red Hat',
    avatar: '/img/team/huamin.png',
    github: 'https://github.com/rootfs',
    linkedin: 'https://www.linkedin.com/in/huaminchen',
    bio: 'Distinguished Engineer at Red Hat, driving innovation in cloud-native and AI/LLM Inference technologies.',
    expertise: ['Cloud Native', 'Kubernetes', 'Container Technologies', 'System Architecture'],
  },
  {
    name: 'Chen Wang',
    role: 'Senior Staff Research Scientist',
    company: 'IBM',
    avatar: '/img/team/chen.png',
    github: 'https://github.com/wangchen615',
    linkedin: 'https://www.linkedin.com/in/chenw615/',
    bio: 'Senior Staff Research Scientist at IBM, focusing on advanced AI systems and research.',
    expertise: ['AI Systems', 'Research Leadership', 'Machine Learning', 'Innovation'],
  },
  {
    name: 'Yue Zhu',
    role: 'Staff Research Scientist',
    company: 'IBM',
    avatar: '/img/team/yue.png',
    github: 'https://github.com/yuezhu1',
    linkedin: 'https://www.linkedin.com/in/yue-zhu-b26526a3/',
    bio: 'Staff Research Scientist at IBM, specializing in AI research and LLM Inference.',
    expertise: ['Machine Learning', 'AI Research', 'Data Science', 'Research & Development'],
  },
  {
    name: 'Xunzhuo Liu',
    role: 'Software Engineer',
    company: 'Tencent',
    avatar: '/img/team/xunzhuo.png',
    github: 'https://github.com/Xunzhuo',
    linkedin: 'https://www.linkedin.com/in/bitliu/',
    bio: 'Software Engineer at Tencent, leading the development of vLLM Semantic Router and driving the project vision.',
    expertise: ['System Architecture', 'ML Infrastructure', 'Open Source', 'Software Engineering'],
  },
]

const contributors = [
  {
    name: 'You?',
    role: 'Future Contributor',
    avatar: 'https://github.com/github.png',
    github: '/community/contributing',
    bio: 'Join our community and help make vLLM Semantic Router even better!',
    expertise: ['Your Skills Here'],
  },
]

function TeamMember({ member, isContributor = false }) {
  return (
    <div className={`${styles.memberCard} ${isContributor ? styles.contributorCard : ''}`}>
      <div className={styles.memberHeader}>
        <img
          src={member.avatar}
          alt={`${member.name} avatar`}
          className={styles.avatar}
        />
        <div className={styles.memberInfo}>
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
        </div>
      </div>

      <p className={styles.memberBio}>{member.bio}</p>

      <div className={styles.expertise}>
        {member.expertise.map((skill, index) => (
          <span key={index} className={styles.skillTag}>{skill}</span>
        ))}
      </div>

      <div className={styles.memberActions}>
        {!isContributor && member.github && member.github !== '#' && (
          <a
            href={member.github}
            target="_blank"
            rel="noopener noreferrer"
            className={styles.actionLink}
          >
            📧 GitHub
          </a>
        )}

        {!isContributor && member.linkedin && (
          <a
            href={member.linkedin}
            target="_blank"
            rel="noopener noreferrer"
            className={styles.actionLink}
          >
            💼 LinkedIn
          </a>
        )}

        {isContributor && (
          <a
            href={member.github}
            target="_self"
            className={styles.joinButton}
          >
            🚀 Join Us
          </a>
        )}
      </div>
    </div>
  )
}

export default function Team() {
  return (
    <Layout
      title="Team"
      description="Meet the team behind vLLM Semantic Router"
    >
      <div className={styles.container}>
        <header className={styles.header}>
          <h1>Meet Our Team 👥</h1>
          <p className={styles.subtitle}>
            The passionate individuals building the future of intelligent LLM routing
          </p>
        </header>

        <main className={styles.main}>
          <section className={styles.section}>
            <h2>🌟 Core Team</h2>
            <p className={styles.sectionDescription}>
              The core maintainers who drive the project forward and make key decisions.
            </p>
            <div className={styles.teamGrid}>
              {coreTeam.map((member, index) => (
                <TeamMember key={index} member={member} />
              ))}
            </div>
          </section>

          <section className={styles.section}>
            <h2>🤝 Join Our Team</h2>
            <p className={styles.sectionDescription}>
              We're always looking for passionate contributors to join our community!
            </p>
            <div className={styles.joinTeamGrid}>
              {contributors.map((member, index) => (
                <TeamMember key={index} member={member} isContributor={true} />
              ))}
            </div>
          </section>

          <section className={styles.section}>
            <h2>🏆 Recognition</h2>
            <div className={styles.recognitionCard}>
              <h3>Contributor Recognition</h3>
              <p>
                We believe in recognizing the valuable contributions of our community members.
                Contributors who show consistent dedication and quality work in specific areas
                may be invited to become maintainers with write access to the repository.
              </p>

              <div className={styles.pathToMaintainer}>
                <h4>Path to Maintainership:</h4>
                <div className={styles.steps}>
                  <div className={styles.step}>
                    <span className={styles.stepNumber}>1</span>
                    <div>
                      <h5>Contribute Regularly</h5>
                      <p>Make consistent, quality contributions to your area of interest</p>
                    </div>
                  </div>

                  <div className={styles.step}>
                    <span className={styles.stepNumber}>2</span>
                    <div>
                      <h5>Join a Working Group</h5>
                      <p>
                        Participate actively in one of our
                        <a href="/community/work-groups">Working Groups</a>
                      </p>
                    </div>
                  </div>

                  <div className={styles.step}>
                    <span className={styles.stepNumber}>3</span>
                    <div>
                      <h5>Community Vote</h5>
                      <p>Receive nomination and approval from the maintainer team</p>
                    </div>
                  </div>

                  <div className={styles.step}>
                    <span className={styles.stepNumber}>4</span>
                    <div>
                      <h5>Maintainer Access</h5>
                      <p>Get invited to the maintainer group with write access</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>

          <section className={styles.section}>
            <h2>📞 Get Involved</h2>
            <div className={styles.involvementGrid}>
              <div className={styles.involvementCard}>
                <h3>🚀 Start Contributing</h3>
                <p>Ready to make your first contribution?</p>
                <a href="/community/contributing" className={styles.actionButton}>
                  Contributing Guide
                </a>
              </div>

              <div className={styles.involvementCard}>
                <h3>👥 Join Working Groups</h3>
                <p>Find your area of expertise and connect with like-minded contributors.</p>
                <a href="/community/work-groups" className={styles.actionButton}>
                  View Work Groups
                </a>
              </div>

              <div className={styles.involvementCard}>
                <h3>💬 Join Discussions</h3>
                <p>Participate in community discussions and share your ideas.</p>
                <a href="https://github.com/vllm-project/semantic-router/discussions" target="_blank" rel="noopener noreferrer" className={styles.actionButton}>
                  GitHub Discussions
                </a>
              </div>
            </div>
          </section>
        </main>
      </div>
    </Layout>
  )
}
