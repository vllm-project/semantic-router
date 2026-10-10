import React from 'react'
import Layout from '@theme/Layout'
import Translate from '@docusaurus/Translate'
import Link from '@docusaurus/Link'
import useBaseUrl from '@docusaurus/useBaseUrl'
import ThemedImage from '@theme/ThemedImage'
import CommunityLayout from '@site/src/components/community/CommunityLayout'
import EcosystemGrid from '@site/src/components/ecosystem/EcosystemGrid'
import styles from './ecosystem.module.css'

const Ecosystem: React.FC = () => {
  const assetPath = useBaseUrl('/img/ecosystem/')

  return (
    <Layout
      title="Ecosystem & Partnerships"
      description="The organizations building with vLLM Semantic Router: an open, programmable decision layer for models and compute."
    >
      <CommunityLayout
        activeKey="ecosystem"
        title={<Translate id="community.ecosystem.title">Ecosystem & partnerships</Translate>}
        description={(
          <Translate
            id="community.ecosystem.description"
            values={{
              decisionLayer: (
                <strong>
                  <Translate id="community.ecosystem.decisionLayer">decision layer</Translate>
                </strong>
              ),
            }}
          >
            {'An open, programmable {decisionLayer} for models and compute.'}
          </Translate>
        )}
      >
        <div className={styles.main}>
          <div className={styles.showcase}>
            <ThemedImage
              className={styles.wordmark}
              sources={{
                light: `${assetPath}vllm-sr-wordmark-light.png`,
                dark: `${assetPath}vllm-sr-wordmark-dark.png`,
              }}
              alt="vLLM Semantic Router"
              width={2160}
              height={690}
              loading="lazy"
            />
            <EcosystemGrid />
          </div>

          <div className={styles.footer}>
            <p>
              <Translate id="homepage.ecosystem.growing">
                And an open community still growing.
              </Translate>
            </p>
            <Link to="/community/contributing">
              <Translate id="homepage.ecosystem.cta">Build with us</Translate>
              <span aria-hidden="true">↗</span>
            </Link>
          </div>
        </div>
      </CommunityLayout>
    </Layout>
  )
}

export default Ecosystem
