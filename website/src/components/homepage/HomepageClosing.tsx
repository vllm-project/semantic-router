import React from 'react'
import Link from '@docusaurus/Link'
import Translate from '@docusaurus/Translate'
import { PillLink } from '@site/src/components/site/Chrome'
import styles from './HomepageClosing.module.css'

const routes = [
  { to: '/docs/intro', label: <Translate id="homepage.closing.docs">Documentation</Translate> },
  { to: '/publications', label: <Translate id="homepage.closing.papers">Research</Translate> },
  { to: '/community/team', label: <Translate id="homepage.closing.team">Team</Translate> },
  { to: '/community/contributors', label: <Translate id="homepage.closing.contributors">Contributors</Translate> },
]

export default function HomepageClosing(): JSX.Element {
  return (
    <section className={styles.section} aria-label="Explore more">
      <div className="site-shell-container">
        <div className={styles.shell}>
          <div className={styles.copy}>
            <h2>
              <Translate id="homepage.closing.title">Ready to route smarter?</Translate>
            </h2>
            <p>
              <Translate id="homepage.closing.description">
                Install locally, explore the docs, or join the community building
                open Mixture-of-Models routing.
              </Translate>
            </p>
          </div>

          <nav className={styles.routeList} aria-label="Site sections">
            {routes.map(route => (
              <Link key={route.to} className={styles.routeLink} to={route.to}>
                {route.label}
              </Link>
            ))}
          </nav>

          <div className={styles.actions}>
            <PillLink href="https://app.vllm-sr.ai/playground" rel="noreferrer" target="_blank">
              <Translate id="homepage.closing.playground">Try playground</Translate>
            </PillLink>
            <PillLink
              href="https://github.com/vllm-project/semantic-router"
              muted
              rel="noreferrer"
              target="_blank"
            >
              <Translate id="homepage.closing.github">GitHub</Translate>
            </PillLink>
          </div>
        </div>
      </div>
    </section>
  )
}
