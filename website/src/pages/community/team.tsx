import React from 'react'
import Layout from '@theme/Layout'
import Translate from '@docusaurus/Translate'
import Link from '@docusaurus/Link'
import styles from './team.module.css'
import { FaGithub, FaLinkedin } from 'react-icons/fa'

interface TeamMember {
  name: string
  role: React.ReactNode
  company?: string
  avatar: string
  github?: string
  linkedin?: string
  bio: React.ReactNode
  memberType: 'maintainer' | 'committer' | 'committer'
}

interface TeamMemberProps {
  member: TeamMember
}

const allTeamMembers: TeamMember[] = [
  {
    name: 'Huamin Chen',
    role: <Translate id="team.members.HuaminChen.role">Distinguished Engineer</Translate>,
    company: 'Red Hat',
    avatar: '/img/team/huamin.png',
    github: 'https://github.com/rootfs',
    linkedin: 'https://www.linkedin.com/in/huaminchen',
    bio: <Translate id="team.members.HuaminChen.bio">Distinguished Engineer at Red Hat, driving innovation in cloud-native and AI/LLM Inference technologies.</Translate>,
    memberType: 'maintainer',
  },
  {
    name: 'Chen Wang',
    role: <Translate id="team.members.ChenWang.role">Senior Staff Research Scientist</Translate>,
    company: 'IBM',
    avatar: '/img/team/chen.png',
    github: 'https://github.com/wangchen615',
    linkedin: 'https://www.linkedin.com/in/chenw615/',
    bio: <Translate id="team.members.ChenWang.bio">Senior Staff Research Scientist at IBM, focusing on advanced AI systems and research.</Translate>,
    memberType: 'maintainer',
  },
  {
    name: 'Yue Zhu',
    role: <Translate id="team.members.YueZhu.role">Staff Research Scientist</Translate>,
    company: 'IBM',
    avatar: '/img/team/yue.png',
    github: 'https://github.com/yuezhu1',
    linkedin: 'https://www.linkedin.com/in/yue-zhu-b26526a3/',
    bio: <Translate id="team.members.YueZhu.bio">Staff Research Scientist at IBM, specializing in AI research and LLM Inference.</Translate>,
    memberType: 'maintainer',
  },
  {
    name: 'Xunzhuo Liu',
    role: <Translate id="team.members.XunzhuoLiu.role">AI Networking</Translate>,
    company: 'Tencent',
    avatar: '/img/team/xunzhuo.png',
    github: 'https://github.com/Xunzhuo',
    linkedin: 'https://www.linkedin.com/in/bitliu/',
    bio: <Translate id="team.members.XunzhuoLiu.bio">AI Networking at Tencent, leading the development of vLLM Semantic Router and driving the project vision.</Translate>,
    memberType: 'maintainer',
  },
  {
    name: 'Senan Zedan',
    company: 'Red Hat',
    role: <Translate id="team.members.SenanZedan.role">R&D Manager</Translate>,
    linkedin: 'https://www.linkedin.com/in/senan-zedan-2041855b/',
    avatar: 'https://github.com/szedan-rh.png',
    github: 'https://github.com/szedan-rh',
    bio: <Translate id="team.members.SenanZedan.bio">A dynamic and hands-on Engineering Manager who thrives on building elite engineering teams and driving them to deliver exceptional results.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'Yossi Ovadia',
    company: 'Red Hat',
    role: <Translate id="team.members.YossiOvadia.role">Senior Principal Engineer</Translate>,
    avatar: 'https://github.com/yossiovadia.png',
    github: 'https://github.com/yossiovadia',
    linkedin: 'https://www.linkedin.com/in/yossi-ovadia-336b314/',
    bio: <Translate id="team.members.YossiOvadia.bio">Making life easier for developers and customers through innovative tooling. From the Red Hat Office of the CTO.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'samzong',
    role: <Translate id="team.members.samzong.role">AI Infrastructure / Cloud-Native PM</Translate>,
    company: 'DaoCloud',
    avatar: 'https://github.com/samzong.png',
    github: 'https://github.com/samzong',
    linkedin: 'https://www.linkedin.com/in/samzong',
    bio: <Translate id="team.members.samzong.bio">Cloud-native AI infrastructure product leader. Focused on Kubernetes, GPU resource scheduling, and large-scale LLM serving platforms.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'Liav Weiss',
    role: <Translate id="team.members.LiavWeiss.role">Software Engineer</Translate>,
    company: 'Red Hat',
    avatar: 'https://avatars.githubusercontent.com/u/74174727?v=4',
    github: 'https://github.com/liavweiss',
    linkedin: 'https://www.linkedin.com/in/liav-weiss-2a0428208',
    bio: <Translate id="team.members.LiavWeiss.bio">Software engineer, focused on backend and cloud-native systems, with hands-on experience exploring AI infrastructure, LLM-based systems, and RAG architectures.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'Asaad Balum',
    role: <Translate id="team.members.AsaadBalum.role">Senior Software Engineer</Translate>,
    company: 'Red Hat',
    avatar: 'https://avatars.githubusercontent.com/u/154635253?s=400&u=6e7e87cce16b88346a3e54e96aad263318a1901a&v=4',
    github: 'https://github.com/asaadbalum',
    linkedin: 'https://www.linkedin.com/in/asaad-balum-0928771a9/',
    bio: <Translate id="team.members.AsaadBalum.bio">Senior software engineer with a research-driven mindset, specializing in cloud-native platforms, Kubernetes-based infrastructure, and AI enablement.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'Yehudit',
    role: <Translate id="team.members.Yehudit.role">Software Engineer</Translate>,
    company: 'Red Hat',
    avatar: 'https://avatars.githubusercontent.com/u/34643974?s=400&v=4',
    github: 'https://github.com/yehudit1987',
    linkedin: 'https://www.linkedin.com/in/yehuditkerido/',
    bio: <Translate id="team.members.Yehudit.bio">Software engineer with a research-driven mindset, focused on cloud-native platforms and AI infrastructure. Open-source contributor.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'Noa Limoy',
    role: <Translate id="team.members.NoaLimoy.role">Software Engineer</Translate>,
    company: 'Red Hat',
    avatar: 'https://avatars.githubusercontent.com/noalimoy',
    github: 'https://github.com/noalimoy',
    linkedin: 'https://www.linkedin.com/in/noalimoy/',
    bio: <Translate id="team.members.NoaLimoy.bio">Software engineer with a research-driven mindset, focused on cloud-native platforms and AI infrastructure. Open-source contributor.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'JaredforReal',
    company: 'Z.ai',
    role: <Translate id="team.members.JaredforReal.role">Software Engineer</Translate>,
    avatar: 'https://github.com/JaredforReal.png',
    github: 'https://github.com/JaredforReal',
    bio: <Translate id="team.members.JaredforReal.bio">Open source contributor to vLLM Semantic Router.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'Abdallah Samara',
    company: 'Red Hat',
    role: <Translate id="team.members.AbdallahSamara.role">Senior Software Engineer</Translate>,
    avatar: 'https://github.com/abdallahsamabd.png',
    github: 'https://github.com/abdallahsamabd',
    linkedin: 'https://www.linkedin.com/in/abdallah-samara',
    bio: <Translate id="team.members.AbdallahSamara.bio">Software engineer with a research-driven approach, focused on cloud-native platforms and AI infrastructure. Building semantic routing systems and contributing to open-source LLM orchestration projects.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'Hen Schwartz',
    company: 'Red Hat',
    role: <Translate id="team.members.HenSchwartz.role">Software Engineer</Translate>,
    avatar: 'https://github.com/henschwartz.png',
    github: 'https://github.com/henschwartz',
    linkedin: 'https://www.linkedin.com/in/henschwartz',
    bio: <Translate id="team.members.HenSchwartz.bio">Software engineer with a research-driven approach, focused on cloud-native platforms and AI infrastructure. Building semantic routing systems and contributing to open-source LLM orchestration projects.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'Srinivas A',
    role: <Translate id="team.members.SrinivasA.role">Software Engineer</Translate>,
    company: 'Yokogawa',
    avatar: 'https://avatars.githubusercontent.com/srini-abhiram',
    github: 'https://github.com/srini-abhiram',
    linkedin: 'https://www.linkedin.com/in/sriniabhiram',
    bio: <Translate id="team.members.SrinivasA.bio">Application software engineer with experience in Distributed Control Systems and Big data.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'carlory',
    role: <Translate id="team.members.carlory.role">Open Source Engineer</Translate>,
    company: 'DaoCloud',
    avatar: 'https://avatars.githubusercontent.com/u/28390961?v=4',
    github: 'https://github.com/carlory',
    bio: <Translate id="team.members.carlory.bio">Open Source Engineer at DaoCloud, focusing on container technology and cloud-native solutions. Passionate about contributing to vllm and other open source projects.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'Jintao Zhang',
    company: 'Kong',
    role: <Translate id="team.members.JintaoZhang.role">Senior Software Engineer</Translate>,
    avatar: 'https://github.com/tao12345666333.png',
    github: 'https://github.com/tao12345666333',
    linkedin: 'https://www.linkedin.com/in/jintao-zhang-402645193/',
    bio: <Translate id="team.members.JintaoZhang.bio">Senior Software Engineer @ Kong Inc. | Microsoft MVP | CNCF Ambassador | Kubernetes Ingress-NGINX maintainer | PyCon China & KCD Beijing organizer.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'yuluo-yx',
    role: <Translate id="team.members.yuluo-yx.role">Individual Contributor</Translate>,
    avatar: 'https://github.com/yuluo-yx.png',
    github: 'https://github.com/yuluo-yx',
    bio: <Translate id="team.members.yuluo-yx.bio">Open source contributor to vLLM Semantic Router.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'cryo-zd',
    role: <Translate id="team.members.cryo-zd.role">Individual Contributor</Translate>,
    avatar: 'https://github.com/cryo-zd.png',
    github: 'https://github.com/cryo-zd',
    bio: <Translate id="team.members.cryo-zd.bio">Open source contributor to vLLM Semantic Router.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'OneZero-Y',
    role: <Translate id="team.members.OneZero-Y.role">Individual Contributor</Translate>,
    avatar: 'https://github.com/OneZero-Y.png',
    github: 'https://github.com/OneZero-Y',
    bio: <Translate id="team.members.OneZero-Y.bio">Open source contributor to vLLM Semantic Router.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'aeft',
    role: <Translate id="team.members.aeft.role">Individual Contributor</Translate>,
    avatar: 'https://github.com/aeft.png',
    github: 'https://github.com/aeft',
    bio: <Translate id="team.members.aeft.bio">Open source contributor to vLLM Semantic Router.</Translate>,
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
                ? <Translate id="team.badge.maintainer">Maintainer</Translate>
                : member.memberType === 'committer'
                  ? <Translate id="team.badge.committer">Committer</Translate>
                  : <Translate id="team.badge.contributor">Contributor</Translate>}
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
          <h1><Translate id="team.title">Meet Our Team 👥</Translate></h1>
          <p className={styles.subtitle}>
            <Translate id="team.subtitle">Innovation thrives when great minds come together</Translate>
          </p>
        </header>

        <main className={styles.main}>
          <section className={styles.section}>
            <h2>
              👥
              <Translate id="team.coreTeam.title">Our Team</Translate>
            </h2>
            <p className={styles.sectionDescription}>
              <Translate id="team.coreTeam.description">
                Meet the talented people who make vLLM Semantic Router possible.
              </Translate>
            </p>
            <div className={styles.teamGrid}>
              {allTeamMembers.map((member, index) => (
                <TeamMemberCard key={index} member={member} />
              ))}
            </div>
          </section>

          <section className={styles.section}>
            <h2>
              🏆
              <Translate id="team.recognition.title">Recognition</Translate>
            </h2>
            <div className={styles.recognitionCard}>
              <h3><Translate id="team.recognition.subtitle">Contributor Recognition</Translate></h3>
              <p>
                <Translate id="team.recognition.description">
                  We believe in recognizing the valuable contributions of our community members.
                  Contributors who show consistent dedication and quality work in specific areas
                  may be invited to become maintainers with write access to the repository.
                </Translate>
              </p>

              <div className={styles.pathToMaintainer}>
                <h4><Translate id="team.recognition.pathTitle">Path to Maintainership:</Translate></h4>
                <div className={styles.steps}>
                  <div className={styles.step}>
                    <span className={styles.stepNumber}>1</span>
                    <div>
                      <h5><Translate id="team.recognition.step1.title">Contribute Regularly</Translate></h5>
                      <p><Translate id="team.recognition.step1.desc">Make consistent, quality contributions to your area of interest</Translate></p>
                    </div>
                  </div>

                  <div className={styles.step}>
                    <span className={styles.stepNumber}>2</span>
                    <div>
                      <h5><Translate id="team.recognition.step2.title">Join a Working Group</Translate></h5>
                      <p>
                        <Translate id="team.recognition.step2.desc">Participate actively in one of our</Translate>
                        {' '}
                        <Link to="/community/work-groups"><Translate id="team.recognition.step2.link">Working Groups</Translate></Link>
                      </p>
                    </div>
                  </div>

                  <div className={styles.step}>
                    <span className={styles.stepNumber}>3</span>
                    <div>
                      <h5><Translate id="team.recognition.step3.title">Community Vote</Translate></h5>
                      <p><Translate id="team.recognition.step3.desc">Receive nomination and approval from the maintainer team</Translate></p>
                    </div>
                  </div>

                  <div className={styles.step}>
                    <span className={styles.stepNumber}>4</span>
                    <div>
                      <h5><Translate id="team.recognition.step4.title">Maintainer Access</Translate></h5>
                      <p><Translate id="team.recognition.step4.desc">Get invited to the maintainer group with write access</Translate></p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>

          <section className={styles.section}>
            <h2>
              📞
              <Translate id="team.getInvolved.title">Get Involved</Translate>
            </h2>
            <div className={styles.involvementGrid}>
              <div className={styles.involvementCard}>
                <h3>
                  🚀
                  <Translate id="team.getInvolved.contribute.title">Start Contributing</Translate>
                </h3>
                <p><Translate id="team.getInvolved.contribute.desc">Ready to make your first contribution?</Translate></p>
                <Link to="/community/contributing" className={styles.actionButton}>
                  <Translate id="team.getInvolved.contribute.link">Contributing Guide</Translate>
                </Link>
              </div>

              <div className={styles.involvementCard}>
                <h3>
                  👥
                  <Translate id="team.getInvolved.workGroups.title">Join Working Groups</Translate>
                </h3>
                <p><Translate id="team.getInvolved.workGroups.desc">Find your area of expertise and connect with like-minded contributors.</Translate></p>
                <Link to="/community/work-groups" className={styles.actionButton}>
                  <Translate id="team.getInvolved.workGroups.link">View Work Groups</Translate>
                </Link>
              </div>

              <div className={styles.involvementCard}>
                <h3>
                  💬
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
