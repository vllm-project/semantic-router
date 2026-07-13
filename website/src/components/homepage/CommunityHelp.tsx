import React from 'react'
import Translate, { translate } from '@docusaurus/Translate'
import { PillLink } from '@site/src/components/site/Chrome'
import ScrollReveal from '@site/src/components/site/ScrollReveal'
import shared from './homepageShared.module.css'
import styles from './CommunityHelp.module.css'

type HelpCard = {
  title: string
  description: string
  cta: string
  href?: string
  to?: string
}

const helpCards: HelpCard[] = [
  {
    title: translate({ id: 'homepage.help.github.title', message: 'GitHub Issues' }),
    description: translate({
      id: 'homepage.help.github.description',
      message: 'Bug reports, feature requests, and roadmap discussion.',
    }),
    cta: translate({ id: 'homepage.help.github.cta', message: 'Open an issue' }),
    href: 'https://github.com/vllm-project/semantic-router/issues',
  },
  {
    title: translate({ id: 'homepage.help.discussions.title', message: 'GitHub Discussions' }),
    description: translate({
      id: 'homepage.help.discussions.description',
      message: 'Ask questions, share ideas, and connect with maintainers.',
    }),
    cta: translate({ id: 'homepage.help.discussions.cta', message: 'Join discussions' }),
    href: 'https://github.com/vllm-project/semantic-router/discussions',
  },
  {
    title: translate({ id: 'homepage.help.docs.title', message: 'Documentation' }),
    description: translate({
      id: 'homepage.help.docs.description',
      message: 'Tutorials, install guides, and API references to get productive fast.',
    }),
    cta: translate({ id: 'homepage.help.docs.cta', message: 'Read the docs' }),
    to: '/docs/intro',
  },
]

export default function CommunityHelp(): JSX.Element {
  return (
    <section className={shared.lightSection} aria-labelledby="community-help-title">
      <div className={`site-shell-container ${shared.sectionInner}`}>
        <ScrollReveal>
          <header className={shared.sectionHeader}>
            <h2 id="community-help-title" className={shared.sectionTitle}>
              <Translate id="homepage.help.title">Get help from the community</Translate>
            </h2>
            <p className={shared.sectionSubtitle}>
              <Translate id="homepage.help.subtitle">New to routing or debugging production? The community is open to everyone.</Translate>
            </p>
          </header>
        </ScrollReveal>

        <div className={styles.grid}>
          {helpCards.map((card, index) => (
            <ScrollReveal key={card.title} delay={index * 70}>
              <article className={styles.card}>
                <h3>{card.title}</h3>
                <p>{card.description}</p>
                {card.href
                  ? (
                      <PillLink href={card.href} rel="noreferrer" target="_blank" muted>
                        {card.cta}
                      </PillLink>
                    )
                  : (
                      <PillLink to={card.to ?? '/docs/intro'} muted>
                        {card.cta}
                      </PillLink>
                    )}
              </article>
            </ScrollReveal>
          ))}
        </div>
      </div>
    </section>
  )
}
