import React, { useEffect, useRef } from 'react'
import Translate from '@docusaurus/Translate'
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
    role: 'AI Networking',
    company: 'Tencent',
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
]

const TeamCarousel: React.FC = () => {
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const scrollContainer = scrollRef.current
    if (!scrollContainer) return

    let animationFrameId: number
    let scrollPosition = 0
    const scrollSpeed = 0.5 // pixels per frame

    const scroll = () => {
      scrollPosition += scrollSpeed

      // Reset scroll position when we've scrolled past one set of items
      const cardWidth = 250 // approximate card width + gap
      const totalWidth = cardWidth * teamMembers.length

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
    }
  }, [])

  // Duplicate members for infinite scroll effect
  const duplicatedMembers = [...teamMembers, ...teamMembers, ...teamMembers]

  return (
    <section className={styles.teamCarousel}>
      <div className="container">
        <h2 className={styles.title}>
          👥
          {' '}
          <Translate id="teamCarousel.title">Meet Our Team</Translate>
        </h2>
        <p className={styles.subtitle}>
          <Translate id="teamCarousel.subtitle">The amazing people behind vLLM Semantic Router</Translate>
        </p>

        <div className={styles.carouselContainer}>
          <div className={styles.carouselTrack} ref={scrollRef}>
            {duplicatedMembers.map((member, index) => (
              <div key={`${member.name}-${index}`} className={styles.memberCard}>
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
              </div>
            ))}
          </div>
        </div>

        <div className={styles.viewAllLink}>
          <a href="/community/team" className={styles.viewAllButton}>
            <Translate id="teamCarousel.viewAll">View All Team Members</Translate>
            {' '}
            →
          </a>
        </div>
      </div>
    </section>
  )
}

export default TeamCarousel
