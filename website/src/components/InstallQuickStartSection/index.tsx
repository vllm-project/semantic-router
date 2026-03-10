import React, { useEffect, useState } from 'react'
import Translate, { translate } from '@docusaurus/Translate'
import useDocusaurusContext from '@docusaurus/useDocusaurusContext'
import { PillButton, PillLink, SectionLabel } from '@site/src/components/site/Chrome'
import styles from './index.module.css'

type CopyState = 'idle' | 'copied' | 'error'

function buildInstallScriptUrl(siteUrl: string, baseUrl: string): string {
  const normalizedSiteUrl = siteUrl.replace(/\/$/, '')
  const normalizedBaseUrl = baseUrl === '/' ? '' : baseUrl.replace(/\/$/, '')
  return `${normalizedSiteUrl}${normalizedBaseUrl}/install.sh`
}

export default function InstallQuickStartSection(): JSX.Element {
  const { siteConfig } = useDocusaurusContext()
  const installScriptUrl = buildInstallScriptUrl(siteConfig.url, siteConfig.baseUrl)
  const installCommand = `curl -fsSL ${installScriptUrl} | bash`
  const [copyState, setCopyState] = useState<CopyState>('idle')

  useEffect(() => {
    if (copyState === 'idle') {
      return undefined
    }

    const timeoutId = window.setTimeout(() => {
      setCopyState('idle')
    }, 1800)

    return () => {
      window.clearTimeout(timeoutId)
    }
  }, [copyState])

  let copyLabel = translate({
    id: 'homepage.install.copy.idle',
    message: 'Copy command',
  })

  if (copyState === 'copied') {
    copyLabel = translate({
      id: 'homepage.install.copy.copied',
      message: 'Copied',
    })
  }
  else if (copyState === 'error') {
    copyLabel = translate({
      id: 'homepage.install.copy.error',
      message: 'Copy failed',
    })
  }

  async function handleCopy(): Promise<void> {
    if (typeof navigator === 'undefined' || !navigator.clipboard) {
      setCopyState('error')
      return
    }

    try {
      await navigator.clipboard.writeText(installCommand)
      setCopyState('copied')
    }
    catch {
      setCopyState('error')
    }
  }

  return (
    <section id="install-quickstart" className={styles.section}>
      <div className="site-shell-container">
        <div className={styles.heading}>
          <div className={styles.meta}>
            <SectionLabel>
              <Translate id="homepage.install.label">Quick start</Translate>
            </SectionLabel>
            <p>
              <Translate id="homepage.install.meta">
                One supported local path. Copy the installer, run it, then open the dashboard.
              </Translate>
            </p>
          </div>

          <div className={styles.copy}>
            <h2>
              <Translate id="homepage.install.title">Install locally in one line.</Translate>
            </h2>
            <p>
              <Translate id="homepage.install.description">
                The supported first-run path is a single installer that sets up the CLI and local
                serve flow on macOS and Linux.
              </Translate>
            </p>
          </div>
        </div>

        <div className={styles.frame}>
          <div className={styles.frameHeader}>
            <SectionLabel>
              <Translate id="homepage.install.frameLabel">One-liner install</Translate>
            </SectionLabel>
            <span className={styles.platform}>macOS / Linux</span>
          </div>

          <div className={styles.commandBlock}>
            <code className={styles.command}>{installCommand}</code>
          </div>

          <div className={styles.frameFooter}>
            <p className={styles.note}>
              <Translate id="homepage.install.footer">
                Installs into ~/.local/share/vllm-sr, writes ~/.local/bin/vllm-sr, and keeps
                Windows on the manual pip flow in the docs.
              </Translate>
            </p>

            <div className={styles.actions}>
              <PillButton
                onClick={() => {
                  void handleCopy()
                }}
              >
                {copyLabel}
              </PillButton>
              <PillLink to="/docs/installation" muted>
                <Translate id="homepage.install.primaryCta">Full installation guide</Translate>
              </PillLink>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
