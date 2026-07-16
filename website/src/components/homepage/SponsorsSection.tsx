import React from 'react'
import Translate, { translate } from '@docusaurus/Translate'
import useDocusaurusContext from '@docusaurus/useDocusaurusContext'
import clsx from 'clsx'
import { SectionLabel } from '@site/src/components/site/Chrome'
import ScrollReveal from '@site/src/components/site/ScrollReveal'
import {
  sponsorCategories,
  type Sponsor,
  type SponsorCategoryId,
  type SponsorLogoStyle,
} from '@site/src/data/sponsors'
import styles from './SponsorsSection.module.css'

function categoryLabel(id: SponsorCategoryId): string {
  switch (id) {
    case 'cash':
      return translate({
        id: 'homepage.sponsors.category.cash',
        message: 'Cash donations',
      })
    case 'compute':
      return translate({
        id: 'homepage.sponsors.category.compute',
        message: 'Compute resources',
      })
    case 'slack':
      return translate({
        id: 'homepage.sponsors.category.slack',
        message: 'Slack sponsor',
      })
    default:
      return id
  }
}

function CategoryIcon({ category }: { category: SponsorCategoryId }): JSX.Element {
  if (category === 'cash') {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true" className={styles.categoryIcon}>
        <path
          d="M12 3c-4.4 0-8 1.8-8 4s3.6 4 8 4 8-1.8 8-4-3.6-4-8-4Zm0 10c-4.4 0-8 1.8-8 4v1h16v-1c0-2.2-3.6-4-8-4Z"
          fill="currentColor"
        />
      </svg>
    )
  }
  if (category === 'compute') {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true" className={styles.categoryIcon}>
        <path
          d="M4 5h16a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1Zm2 3v2h2V8H6Zm4 0v2h2V8h-2Zm4 0v2h2V8h-2ZM6 13v2h2v-2H6Zm4 0v2h2v-2h-2Zm4 0v2h2v-2h-2Z"
          fill="currentColor"
        />
      </svg>
    )
  }
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className={styles.categoryIcon}>
      <path
        d="M5 5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v9.5a2 2 0 0 1-2 2H9.8L5 20.3V5Zm3.5 4.5h7v1.5h-7V9.5Zm0 3h5v1.5h-5V12.5Z"
        fill="currentColor"
      />
    </svg>
  )
}

function resolveLogoSrc(logo: string, assetBase: string): string {
  if (logo.startsWith('http')) {
    return logo
  }
  return `${assetBase}${logo.replace(/^\//, '')}`
}

function logoClassName(style: SponsorLogoStyle | undefined): string {
  return style === 'mono' ? styles.sponsorLogoMono : styles.sponsorLogoBrand
}

function SponsorChip({
  sponsor,
  assetBase,
}: {
  sponsor: Sponsor
  assetBase: string
}): JSX.Element {
  const logoSrc = resolveLogoSrc(sponsor.logo, assetBase)

  return (
    <a
      href={sponsor.url}
      className={styles.sponsorChip}
      target="_blank"
      rel="noopener noreferrer"
      title={sponsor.name}
    >
      <span className={styles.sponsorMark} aria-hidden="true">
        <img
          src={logoSrc}
          alt=""
          className={clsx(styles.sponsorLogo, logoClassName(sponsor.logoStyle))}
        />
      </span>
      <span className={styles.sponsorName}>{sponsor.name}</span>
    </a>
  )
}

export default function SponsorsSection(): JSX.Element {
  const { siteConfig } = useDocusaurusContext()
  const assetBase = siteConfig.baseUrl.endsWith('/')
    ? siteConfig.baseUrl
    : `${siteConfig.baseUrl}/`

  return (
    <section className={styles.section} aria-labelledby="homepage-sponsors-title">
      <div className="site-shell-container">
        <ScrollReveal>
          <header className={styles.header}>
            <SectionLabel>
              <Translate id="homepage.sponsors.label">Community</Translate>
            </SectionLabel>
            <h2 id="homepage-sponsors-title" className={styles.title}>
              <Translate id="homepage.sponsors.title">Sponsors</Translate>
            </h2>
            <p className={styles.subtitle}>
              <Translate id="homepage.sponsors.subtitle">
                vLLM Semantic Router is a community project. Development and testing
                compute are supported by the organizations below. Thank you for your
                support.
              </Translate>
            </p>
          </header>
        </ScrollReveal>

        <div className={styles.categories}>
          {sponsorCategories.map((category, categoryIndex) => (
            <ScrollReveal key={category.id} delay={categoryIndex * 60}>
              <div className={styles.categoryBlock}>
                <div className={styles.categoryHeading}>
                  <CategoryIcon category={category.id} />
                  <h3>{categoryLabel(category.id)}</h3>
                </div>
                <ul className={styles.sponsorGrid}>
                  {category.sponsors.map(sponsor => (
                    <li key={sponsor.id}>
                      <SponsorChip sponsor={sponsor} assetBase={assetBase} />
                    </li>
                  ))}
                </ul>
              </div>
            </ScrollReveal>
          ))}
        </div>

        <ScrollReveal delay={180}>
          <p className={styles.footnote}>
            <Translate id="homepage.sponsors.footnote">
              Donations are collected through the vLLM project to support development,
              maintenance, and adoption across the ecosystem.
            </Translate>
            {' '}
            <a
              href="https://vllm.ai/"
              target="_blank"
              rel="noopener noreferrer"
              className={styles.footnoteLink}
            >
              <Translate id="homepage.sponsors.footnoteLink">Learn more on vllm.ai</Translate>
            </a>
          </p>
        </ScrollReveal>
      </div>
    </section>
  )
}
