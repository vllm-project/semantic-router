import React from 'react'
import Layout from '@theme/Layout'
import styles from './team.module.css'
import { FaGithub, FaLinkedin } from 'react-icons/fa'

interface TeamMember {
  name: string
  role: string
  company?: string
  avatar: string
  github?: string
  linkedin?: string
  bio: string
  memberType: 'maintainer' | 'committer' | 'committer'
}

interface TeamMemberProps {
  member: TeamMember
}

const allTeamMembers: TeamMember[] = [
  {
    name: 'Huamin Chen',
    role: 'Distinguished Engineer',
    company: 'Red Hat',
    avatar: '/img/team/huamin.png',
    github: 'https://github.com/rootfs',
    linkedin: 'https://www.linkedin.com/in/huaminchen',
    bio: 'Distinguished Engineer at Red Hat, driving innovation in cloud-native and AI/LLM Inference technologies.',
    memberType: 'maintainer',
  },
  {
    name: 'Chen Wang',
    role: 'Senior Staff Research Scientist',
    company: 'IBM',
    avatar: '/img/team/chen.png',
    github: 'https://github.com/wangchen615',
    linkedin: 'https://www.linkedin.com/in/chenw615/',
    bio: 'Senior Staff Research Scientist at IBM, focusing on advanced AI systems and research.',
    memberType: 'maintainer',
  },
  {
    name: 'Yue Zhu',
    role: 'Staff Research Scientist',
    company: 'IBM',
    avatar: '/img/team/yue.png',
    github: 'https://github.com/yuezhu1',
    linkedin: 'https://www.linkedin.com/in/yue-zhu-b26526a3/',
    bio: 'Staff Research Scientist at IBM, specializing in AI research and LLM Inference.',
    memberType: 'maintainer',
  },
  {
    name: 'Xunzhuo Liu',
    role: 'AI Networking',
    company: 'Tencent',
    avatar: '/img/team/xunzhuo.png',
    github: 'https://github.com/Xunzhuo',
    linkedin: 'https://www.linkedin.com/in/bitliu/',
    bio: 'AI Networking at Tencent, leading the development of vLLM Semantic Router and driving the project vision.',
    memberType: 'maintainer',
  },
  {
    name: 'Senan Zedan',
    company: 'Red Hat',
    role: 'R&D Manager',
    linkedin: 'https://www.linkedin.com/in/senan-zedan-2041855b/',
    avatar: 'https://github.com/szedan-rh.png',
    github: 'https://github.com/szedan-rh',
    bio: 'A dynamic and hands-on Engineering Manager who thrives on building elite engineering teams and driving them to deliver exceptional results.',
    memberType: 'committer',
  },
  {
    name: 'samzong',
    role: 'AI Infrastructure / Cloud-Native PM',
    company: 'DaoCloud',
    avatar: 'https://github.com/samzong.png',
    github: 'https://github.com/samzong',
    linkedin: 'https://www.linkedin.com/in/samzong',
    bio: 'Cloud-native AI infrastructure product leader. Focused on Kubernetes, GPU resource scheduling, and large-scale LLM serving platforms.',
    memberType: 'committer',
  },
  {
    name: 'Liav Weiss',
    role: 'Software Engineer',
    company: 'Red Hat',
    avatar: 'https://avatars.githubusercontent.com/u/74174727?v=4',
    github: 'https://github.com/liavweiss',
    linkedin: 'https://www.linkedin.com/in/liav-weiss-2a0428208',
    bio: 'Software engineer, focused on backend and cloud-native systems, with hands-on experience exploring AI infrastructure, LLM-based systems, and RAG architectures.',
    memberType: 'committer',
  },
  {
    name: 'Asaad Balum',
    role: 'Senior Software Engineer',
    company: 'Red Hat',
    avatar: 'https://avatars.githubusercontent.com/u/154635253?s=400&u=6e7e87cce16b88346a3e54e96aad263318a1901a&v=4',
    github: 'https://github.com/asaadbalum',
    linkedin: 'https://www.linkedin.com/in/asaad-balum-0928771a9/',
    bio: 'Senior software engineer with a research-driven mindset, specializing in cloud-native platforms, Kubernetes-based infrastructure, and AI enablement.',
    memberType: 'committer',
  },
  {
    name: 'Yehudit',
    role: 'Software Engineer',
    company: 'Red Hat',
    avatar: 'https://avatars.githubusercontent.com/u/34643974?s=400&v=4',
    github: 'https://github.com/yehudit1987',
    linkedin: 'https://www.linkedin.com/in/yehuditkerido/',
    bio: 'Software engineer with a research-driven mindset, focused on cloud-native platforms and AI infrastructure. Open-source contributor.',
    memberType: 'committer',
  },
  {
    name: 'Noa Limoy',
    role: 'Software Engineer',
    company: 'Red Hat',
    avatar: 'https://avatars.githubusercontent.com/noalimoy',
    github: 'https://github.com/noalimoy',
    linkedin: 'https://www.linkedin.com/in/noalimoy/',
    bio: 'Software engineer with a research-driven mindset, focused on cloud-native platforms and AI infrastructure. Open-source contributor.',
    memberType: 'committer',
  },
  {
    name: 'JaredforReal',
    company: 'Z.ai',
    role: 'Software Engineer',
    avatar: 'https://github.com/JaredforReal.png',
    github: 'https://github.com/JaredforReal',
    bio: 'Open source contributor to vLLM Semantic Router.',
    memberType: 'committer',
  },
  {
    name: 'Srinivas A',
    role: 'Software Engineer',
    company: 'Yokogawa',
    avatar: 'https://avatars.githubusercontent.com/srini-abhiram',
    github: 'https://github.com/srini-abhiram',
    linkedin: 'https://www.linkedin.com/in/sriniabhiram',
    bio: 'Application software engineer with experience in Distributed Control Systems and Big data.',
    memberType: 'committer',
  },
  {
    name: 'carlory',
    role: 'Open Source Engineer',
    company: 'DaoCloud',
    avatar: 'https://avatars.githubusercontent.com/u/28390961?v=4',
    github: 'https://github.com/carlory',
    bio: 'Open Source Engineer at DaoCloud, focusing on container technology and cloud-native solutions. Passionate about contributing to vllm and other open source projects.',
    memberType: 'committer',
  },
  {
    name: 'Yossi Ovadia',
    company: 'Red Hat',
    role: 'Senior Principal Engineer',
    avatar: 'https://github.com/yossiovadia.png',
    github: 'https://github.com/yossiovadia',
    linkedin: 'https://www.linkedin.com/in/yossi-ovadia-336b314/',
    bio: 'Making life easier for developers and customers through innovative tooling. From the Red Hat Office of the CTO.',
    memberType: 'committer',
  },
  {
    name: 'Jintao Zhang',
    company: 'Kong',
    role: 'Senior Software Engineer',
    avatar: 'https://github.com/tao12345666333.png',
    github: 'https://github.com/tao12345666333',
    linkedin: 'https://www.linkedin.com/in/jintao-zhang-402645193/',
    bio: 'Senior Software Engineer @ Kong Inc. | Microsoft MVP | CNCF Ambassador | Kubernetes Ingress-NGINX maintainer | PyCon China & KCD Beijing organizer.',
    memberType: 'committer',
  },
  {
    name: 'yuluo-yx',
    role: 'Individual Contributor',
    avatar: 'https://github.com/yuluo-yx.png',
    github: 'https://github.com/yuluo-yx',
    bio: 'Open source contributor to vLLM Semantic Router.',
    memberType: 'committer',
  },
  {
    name: 'cryo-zd',
    role: 'Individual Contributor',
    avatar: 'https://github.com/cryo-zd.png',
    github: 'https://github.com/cryo-zd',
    bio: 'Open source contributor to vLLM Semantic Router.',
    memberType: 'committer',
  },
  {
    name: 'OneZero-Y',
    role: 'Individual Contributor',
    avatar: 'https://github.com/OneZero-Y.png',
    github: 'https://github.com/OneZero-Y',
    bio: 'Open source contributor to vLLM Semantic Router.',
    memberType: 'committer',
  },
  {
    name: 'aeft',
    role: 'Individual Contributor',
    avatar: 'https://github.com/aeft.png',
    github: 'https://github.com/aeft',
    bio: 'Open source contributor to vLLM Semantic Router.',
    memberType: 'committer',
  },
]

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
              {member.memberType === 'maintainer'
                ? 'Maintainer'
                : member.memberType === 'committer'
                  ? 'Committer'
                  : 'Contributor'}
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
          <h1>Meet Our Team 👥</h1>
          <p className={styles.subtitle}>
            Innovation thrives when great minds come together
          </p>
        </header>

        <main className={styles.main}>
          <section className={styles.section}>
            <h2>👥 Our Team</h2>
            <p className={styles.sectionDescription}>
              Meet the talented people who make vLLM Semantic Router possible.
            </p>
            <div className={styles.teamGrid}>
              {allTeamMembers.map((member, index) => (
                <TeamMemberCard key={index} member={member} />
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

export default Team
