import React from 'react'
import Layout from '@theme-original/DocItem/Layout'
import type LayoutType from '@theme/DocItem/Layout'
import type { WrapperProps } from '@docusaurus/types'
import { useDoc } from '@docusaurus/plugin-content-docs/client'
import TranslationBanner from '@site/src/components/TranslationBanner'
import { PillLink, SectionLabel } from '@site/src/components/site/Chrome'

type Props = WrapperProps<typeof LayoutType>

function DocMasthead(): JSX.Element {
  const { metadata, frontMatter } = useDoc()
  const description = typeof frontMatter.description === 'string'
    ? frontMatter.description
    : metadata.description

  return (
    <div className="site-doc-masthead">
      <div className="site-shell-container">
        <SectionLabel>Documentation</SectionLabel>
        <div className="site-doc-masthead__row">
          <div className="site-doc-masthead__copy">
            <h1>{metadata.title}</h1>
            {description && <p>{description}</p>}
          </div>
          <div className="site-doc-masthead__actions">
            <PillLink to="/docs/intro">Docs index</PillLink>
            {metadata.editUrl && (
              <PillLink href={metadata.editUrl} muted target="_blank" rel="noreferrer">
                Edit page
              </PillLink>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default function LayoutWrapper(props: Props): React.JSX.Element {
  return (
    <div className="site-docs-shell">
      <DocMasthead />
      <TranslationBanner />
      <Layout {...props} />
    </div>
  )
}
