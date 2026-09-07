import React, { useEffect, useState } from 'react'
import Translate, { translate } from '@docusaurus/Translate'
import useDocusaurusContext from '@docusaurus/useDocusaurusContext'
import { PillLink, SectionLabel } from '@site/src/components/site/Chrome'
import styles from './index.module.css'

type CopyStatus = 'idle' | 'copied' | 'error'

function buildInstallScriptUrl(siteUrl: string, baseUrl: string): string {
  const normalizedSiteUrl = siteUrl.replace(/\/$/, '')
  const normalizedBaseUrl = baseUrl === '/' ? '' : baseUrl.replace(/\/$/, '')
  return `${normalizedSiteUrl}${normalizedBaseUrl}/install.sh`
}

export default function InstallQuickStartSection(): JSX.Element {
  const { siteConfig } = useDocusaurusContext()
  const installScriptUrl = buildInstallScriptUrl(siteConfig.url, siteConfig.baseUrl)
  const installCommand = `curl -fsSL ${installScriptUrl} | bash -s -- --channel dev`
  const [copyStatus, setCopyStatus] = useState<CopyStatus>('idle')

  useEffect(() => {
    if (copyStatus === 'idle') {
      return undefined
    }

    const timeoutId = window.setTimeout(() => {
      setCopyStatus('idle')
    }, 1800)

    return () => {
      window.clearTimeout(timeoutId)
    }
  }, [copyStatus])

  async function handleCopy(): Promise<void> {
    if (typeof navigator === 'undefined' || !navigator.clipboard) {
      setCopyStatus('error')
      return
    }

    try {
      await navigator.clipboard.writeText(installCommand)
      setCopyStatus('copied')
    }
    catch {
      setCopyStatus('error')
    }
  }

  const copied = copyStatus === 'copied'
  const failed = copyStatus === 'error'
  const copyLabel = copied
    ? translate({ id: 'homepage.install.copy.copied', message: 'Copied' })
    : failed
      ? translate({ id: 'homepage.install.copy.error', message: 'Copy failed' })
      : translate({ id: 'homepage.install.copy.aria', message: 'Copy command to clipboard' })

  return (
    <section id="install-quickstart" className={styles.section}>
      <div className="site-shell-container">
        <header className={`site-section-intro ${styles.heading}`}>
          <SectionLabel>
            <Translate id="homepage.install.label">Quick start</Translate>
          </SectionLabel>
          <h2>
            <Translate id="homepage.install.title.human">
              Install locally in one line
            </Translate>
          </h2>
        </header>

        <div className={styles.commandShell}>
          <span className={styles.commandPrompt} aria-hidden="true">$</span>
          <code className={styles.command}>{installCommand}</code>
          <button
            type="button"
            className={`${styles.copyButton} ${copied ? styles.copyButtonSuccess : ''}`}
            onClick={() => {
              void handleCopy()
            }}
            title={copyLabel}
            aria-label={copyLabel}
          >
            <span aria-hidden="true">{copied ? '✓' : failed ? '!' : '⧉'}</span>
          </button>
        </div>

        <div className={styles.actions}>
          <PillLink className={styles.guideLink} to="/docs/installation">
            <Translate id="homepage.install.primaryCta">
              Full installation guide
            </Translate>
          </PillLink>
          <PillLink className={styles.docsLink} to="/docs/intro" muted>
            <Translate id="homepage.install.secondaryCta">Read the docs</Translate>
          </PillLink>
        </div>
      </div>
    </section>
  )
}
