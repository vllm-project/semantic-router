import React from 'react'
import Link from '@docusaurus/Link'
import Translate from '@docusaurus/Translate'
import useBaseUrl from '@docusaurus/useBaseUrl'
import styles from './EcosystemSection.module.css'

type Organization = {
  id: string
  name: string
  width: number
  caption?: boolean
}

const organizations: Organization[] = [
  { id: 'amd', name: 'AMD', width: 148 },
  { id: 'hugging-face', name: 'Hugging Face', width: 176 },
  { id: 'microsoft', name: 'Microsoft', width: 166 },
  { id: 'intel', name: 'Intel', width: 106 },
  { id: 'nvidia', name: 'NVIDIA', width: 154 },
  { id: 'red-hat', name: 'Red Hat', width: 142 },
  { id: 'ibm', name: 'IBM', width: 104 },
  { id: 'liquid', name: 'Liquid', width: 146 },
  { id: 'daocloud', name: 'DaoCloud', width: 144 },
  { id: 'delta', name: 'Delta', width: 136 },
  { id: 'mbzuai', name: 'MBZUAI', width: 160, caption: true },
  { id: 'mcgill', name: 'McGill University', width: 146 },
  { id: 'kr-labs', name: '[KR] Labs', width: 138 },
  { id: 'university-of-chicago', name: 'University of Chicago', width: 144 },
  { id: 'uc-berkeley', name: 'UC Berkeley', width: 146 },
  { id: 'umass-boston', name: 'UMass Boston', width: 145, caption: true },
  { id: 'uic', name: 'University of Illinois Chicago', width: 185 },
  { id: 'national-taiwan-university', name: 'National Taiwan University', width: 163, caption: true },
  { id: 'nyu', name: 'New York University', width: 175 },
  { id: 'ubs', name: 'UBS', width: 120 },
  { id: 'ai21', name: 'AI21', width: 100 },
  { id: 'bayer', name: 'Bayer', width: 72 },
  { id: 'dell', name: 'Dell', width: 144 },
  { id: 'nutanix', name: 'Nutanix', width: 155 },
]

export default function EcosystemSection(): React.JSX.Element {
  const assetPath = useBaseUrl('/img/ecosystem/')

  return (
    <section id="ecosystem" className={styles.section} aria-labelledby="ecosystem-title">
      <div
        className={styles.atmosphere}
        style={{ backgroundImage: `url(${assetPath}atmosphere.webp)` }}
        aria-hidden="true"
      />
      <div className={styles.content}>
        <header className={styles.heading}>
          <img
            className={styles.wordmark}
            src={`${assetPath}vllm-sr-wordmark-dark.png`}
            alt="vLLM Semantic Router"
            width={2160}
            height={690}
            loading="lazy"
          />
          <h2 id="ecosystem-title">
            <Translate id="homepage.ecosystem.title">Ecosystem & partnerships</Translate>
          </h2>
          <p>
            <Translate
              id="homepage.ecosystem.description"
              values={{
                decisionLayer: (
                  <strong>
                    <Translate id="homepage.ecosystem.decisionLayer">decision layer</Translate>
                  </strong>
                ),
              }}
            >
              {'An open, programmable {decisionLayer} for models and compute.'}
            </Translate>
          </p>
        </header>

        <ul className={styles.organizations}>
          {organizations.map(organization => (
            <li
              className={styles.organization}
              key={organization.id}
              data-caption={organization.caption || undefined}
              style={{ '--logo-width': `${organization.width}px` } as React.CSSProperties}
            >
              <img
                src={`${assetPath}${organization.id}.svg`}
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

        <div className={styles.footer}>
          <p>
            <Translate id="homepage.ecosystem.growing">
              And an open community still growing.
            </Translate>
          </p>
          <Link to="/docs/community/overview">
            <Translate id="homepage.ecosystem.cta">Build with us</Translate>
            <span aria-hidden="true">↗</span>
          </Link>
        </div>
      </div>
    </section>
  )
}
