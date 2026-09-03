import React from 'react'
import Layout from '@theme-original/DocItem/Layout'
import type LayoutType from '@theme/DocItem/Layout'
import type { WrapperProps } from '@docusaurus/types'
import TranslationBanner from '@site/src/components/TranslationBanner'

type Props = WrapperProps<typeof LayoutType>

export default function LayoutWrapper(props: Props): React.JSX.Element {
  return (
    <div className="site-docs-shell">
      <TranslationBanner />
      <Layout {...props} />
    </div>
  )
}
