import React from 'react'
import useBaseUrl from '@docusaurus/useBaseUrl'
import ThemedImage from '@theme/ThemedImage'
import styles from './EcosystemGrid.module.css'

type Organization = {
  id: string
  name: string
  width: number
  caption?: boolean
  light?: boolean
}

/**
 * The Ecosystem & partnerships logo grid: one ordered reading sequence,
 * shared by the homepage teaser and the Community ecosystem page so the
 * logo set, order, and copy live in exactly one place. Keep in sync with
 * the static export in website/static/img/ecosystem/README.md.
 */
export const organizations: Organization[] = [
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

export default function EcosystemGrid(): React.JSX.Element {
  const assetPath = useBaseUrl('/img/ecosystem/')

  return (
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
  )
}
