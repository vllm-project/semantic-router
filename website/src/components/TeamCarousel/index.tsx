import React, { useEffect, useRef } from 'react'
import Translate from '@docusaurus/Translate'
import { PillLink, SectionLabel } from '@site/src/components/site/Chrome'
import styles from './styles.module.css'

interface TeamMember {
  name: string
  role: string
  company?: string
  avatar: string
  memberType: 'maintainer' | 'committer' | 'contributor'
}

// Complete team members data
const teamMembers: TeamMember[] = [
  {
    name: 'Huamin Chen',
    role: 'Distinguished Engineer',
    company: 'Red Hat',
    avatar: '/img/team/huamin.png',
    memberType: 'maintainer',
  },
  {
    name: 'Chen Wang',
    role: 'Senior Staff Research Scientist',
    company: 'IBM',
    avatar: '/img/team/chen.png',
    memberType: 'maintainer',
  },
  {
    name: 'Yue Zhu',
    role: 'Staff Research Scientist',
    company: 'IBM',
    avatar: '/img/team/yue.png',
    memberType: 'maintainer',
  },
  {
    name: 'Xunzhuo Liu',
    role: 'Intelligent Routing',
    company: 'vLLM',
    avatar: '/img/team/xunzhuo.png',
    memberType: 'maintainer',
  },
  {
    name: 'Senan Zedan',
    company: 'Red Hat',
    role: 'R&D Manager',
    avatar: 'https://github.com/szedan-rh.png',
    memberType: 'committer',
  },
  {
    name: 'samzong',
    role: 'AI Infrastructure / Cloud-Native PM',
    company: 'DaoCloud',
    avatar: 'https://github.com/samzong.png',
    memberType: 'committer',
  },
  {
    name: 'Liav Weiss',
    role: 'Software Engineer',
    company: 'Red Hat',
    avatar: 'https://avatars.githubusercontent.com/u/74174727?v=4',
    memberType: 'committer',
  },
  {
    name: 'Asaad Balum',
    role: 'Senior Software Engineer',
    company: 'Red Hat',
    avatar: 'https://avatars.githubusercontent.com/u/154635253?s=400&u=6e7e87cce16b88346a3e54e96aad263318a1901a&v=4',
    memberType: 'committer',
  },
  {
    name: 'Yehudit',
    role: 'Software Engineer',
    company: 'Red Hat',
    avatar: 'https://avatars.githubusercontent.com/u/34643974?s=400&v=4',
    memberType: 'committer',
  },
  {
    name: 'Noa Limoy',
    role: 'Software Engineer',
    company: 'Red Hat',
    avatar: 'https://avatars.githubusercontent.com/noalimoy',
    memberType: 'committer',
  },
  {
    name: 'JaredforReal',
    company: 'Z.ai',
    role: 'Software Engineer',
    avatar: 'https://github.com/JaredforReal.png',
    memberType: 'committer',
  },
  {
    name: 'Srinivas A',
    role: 'Software Engineer',
    company: 'Yokogawa',
    avatar: 'https://avatars.githubusercontent.com/srini-abhiram',
    memberType: 'committer',
  },
  {
    name: 'carlory',
    role: 'Open Source Engineer',
    company: 'DaoCloud',
    avatar: 'https://avatars.githubusercontent.com/u/28390961?v=4',
    memberType: 'committer',
  },
  {
    name: 'Yossi Ovadia',
    company: 'Red Hat',
    role: 'Senior Principal Engineer',
    avatar: 'https://github.com/yossiovadia.png',
    memberType: 'committer',
  },
  {
    name: 'Jintao Zhang',
    company: 'Kong',
    role: 'Senior Software Engineer',
    avatar: 'https://github.com/tao12345666333.png',
    memberType: 'committer',
  },
  {
    name: 'yuluo-yx',
    role: 'Individual Contributor',
    avatar: 'https://github.com/yuluo-yx.png',
    memberType: 'committer',
  },
  {
    name: 'cryo-zd',
    role: 'Individual Contributor',
    avatar: 'https://github.com/cryo-zd.png',
    memberType: 'committer',
  },
  {
    name: 'OneZero-Y',
    role: 'Individual Contributor',
    avatar: 'https://github.com/OneZero-Y.png',
    memberType: 'committer',
  },
  {
    name: 'aeft',
    role: 'Individual Contributor',
    avatar: 'https://github.com/aeft.png',
    memberType: 'committer',
  },
  {
    name: 'Hao Wu',
    role: 'Individual Contributor',
    avatar: 'https://github.com/haowu1234.png',
    memberType: 'committer',
  },
  {
    name: 'Qiping Pan',
    role: 'Individual Contributor',
    avatar: 'https://github.com/ppppqp.png',
    memberType: 'committer',
  },
]

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
              <Translate id="teamCarousel.label">Contributors</Translate>
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
                      {member.memberType === 'maintainer'
                        ? <Translate id="team.badge.maintainer">Maintainer</Translate>
                        : member.memberType === 'committer'
                          ? <Translate id="team.badge.committer">Committer</Translate>
                          : <Translate id="team.badge.contributor">Contributor</Translate>}
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
              Maintainers, committers, and contributors across research, infrastructure, and open-source operations.
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
