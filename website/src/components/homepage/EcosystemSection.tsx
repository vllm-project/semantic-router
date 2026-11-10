import React from 'react'
import Link from '@docusaurus/Link'
import Translate from '@docusaurus/Translate'
import useBaseUrl from '@docusaurus/useBaseUrl'
import ThemedImage from '@theme/ThemedImage'
import { SectionLabel } from '@site/src/components/site/Chrome'
import duo from '@site/src/components/duo/duo.module.css'
import foundations from './openSourceFoundations.json'
import styles from './EcosystemSection.module.css'

type Organization = {
  id: string
  name: string
  width: number
  caption?: boolean
  light?: boolean
}

type Project = {
  id: string
  name: string
  logo: string
  url: string
}

const organizations: Organization[] = [
  { id: 'amd', name: 'AMD', width: 148, light: true },
  { id: 'hugging-face', name: 'Hugging Face', width: 176 },
  { id: 'microsoft', name: 'Microsoft', width: 166 },
  { id: 'intel', name: 'Intel', width: 106, light: true },
  { id: 'nvidia', name: 'NVIDIA', width: 154, light: true },
  { id: 'red-hat', name: 'Red Hat', width: 142 },
  { id: 'ibm', name: 'IBM', width: 104 },
  { id: 'liquid', name: 'Liquid', width: 146, light: true },
  { id: 'daocloud', name: 'DaoCloud', width: 144, light: true },
  { id: 'delta', name: 'Delta', width: 136 },
  { id: 'mbzuai', name: 'MBZUAI', width: 160, caption: true, light: true },
  { id: 'mcgill', name: 'McGill University', width: 146 },
  { id: 'kr-labs', name: '[KR] Labs', width: 138, light: true },
  { id: 'university-of-chicago', name: 'University of Chicago', width: 144, light: true },
  { id: 'uc-berkeley', name: 'UC Berkeley', width: 146, light: true },
  { id: 'umass-boston', name: 'UMass Boston', width: 145, caption: true },
  { id: 'uic', name: 'University of Illinois Chicago', width: 185, light: true },
  { id: 'national-taiwan-university', name: 'National Taiwan University', width: 163, caption: true },
  { id: 'nyu', name: 'New York University', width: 175, light: true },
  { id: 'ubs', name: 'UBS', width: 120, light: true },
  { id: 'ai21', name: 'AI21', width: 100, light: true },
  { id: 'bayer', name: 'Bayer', width: 72 },
  { id: 'dell', name: 'Dell', width: 144 },
  { id: 'nutanix', name: 'Nutanix', width: 155, light: true },
]

const projects: Project[] = foundations.projects

export default function EcosystemSection(): React.JSX.Element {
  const assetPath = useBaseUrl('/img/ecosystem/')

  return (
    <section id="ecosystem" className={styles.section} aria-labelledby="ecosystem-title">
      <div className={styles.wrap}>
        <header className="site-section-intro">
          <SectionLabel>
            <Translate id="homepage.ecosystem.label">Industry & research</Translate>
          </SectionLabel>
          <h2 id="ecosystem-title">
            <Translate id="homepage.ecosystem.title">Ecosystem & partnerships</Translate>
          </h2>
        </header>

        <ul className={styles.organizations}>
          {organizations.map(organization => (
            <li
              className={styles.organization}
              key={organization.id}
              data-caption={organization.caption || undefined}
              style={{ '--logo-width': `${organization.width}px` } as React.CSSProperties}
            >
              <ThemedImage
                sources={{
                  light: `${assetPath}${organization.id}${organization.light ? '-light' : ''}.svg`,
                  dark: `${assetPath}${organization.id}.svg`,
                }}
                alt={organization.name}
                loading="lazy"
                decoding="async"
              />
              {organization.caption && (
                <span className={styles.logoCaption} aria-hidden="true">{organization.name}</span>
              )}
            </li>
          ))}
        </ul>

        <div className={styles.foundations}>
          <div className={styles.foundationsIntro}>
            <SectionLabel>
              <Translate id="acknowledgements.label">Open-source foundations</Translate>
            </SectionLabel>
            <h3 className={styles.foundationsTitle} id="foundations-title">
              <Translate id="acknowledgements.title">Built with open source</Translate>
            </h3>
            <p className={styles.foundationsLede}>
              <Translate id="acknowledgements.subtitle">
                The libraries and platforms beneath vLLM Semantic Router.
              </Translate>
            </p>
          </div>

          <ul className={styles.projects} aria-labelledby="foundations-title">
            {projects.map(project => (
              <li key={project.id}>
                <a
                  className={styles.project}
                  href={project.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  data-project-id={project.id}
                >
                  <span className={styles.projectLogo}>
                    <img src={project.logo} alt="" loading="lazy" decoding="async" />
                  </span>
                  <span className={styles.projectName}>{project.name}</span>
                  <span className={styles.projectArrow} aria-hidden="true">↗</span>
                </a>
              </li>
            ))}
          </ul>
        </div>

        <div className={styles.footer}>
          <p>
            <Translate id="homepage.ecosystem.growing">
              And an open community still growing.
            </Translate>
          </p>
          <Link className={duo.more} to="/docs/community/overview">
            <Translate id="homepage.ecosystem.cta">Build with us</Translate>
          </Link>
        </div>
      </div>
    </section>
  )
}
