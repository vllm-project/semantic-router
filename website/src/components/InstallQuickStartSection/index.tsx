import React, { useEffect, useState } from 'react'
import Translate, { translate } from '@docusaurus/Translate'
import useDocusaurusContext from '@docusaurus/useDocusaurusContext'
import { PillButton, PillLink, SectionLabel } from '@site/src/components/site/Chrome'
import styles from './index.module.css'

type CopyStatus = 'idle' | 'copied' | 'error'

type CopyState = {
  status: CopyStatus
  target: string | null
}

function buildInstallScriptUrl(siteUrl: string, baseUrl: string): string {
  const normalizedSiteUrl = siteUrl.replace(/\/$/, '')
  const normalizedBaseUrl = baseUrl === '/' ? '' : baseUrl.replace(/\/$/, '')
  return `${normalizedSiteUrl}${normalizedBaseUrl}/install.sh`
}

export default function InstallQuickStartSection(): JSX.Element {
  const { siteConfig } = useDocusaurusContext()
  const installScriptUrl = buildInstallScriptUrl(siteConfig.url, siteConfig.baseUrl)
  const installCommand = `curl -fsSL ${installScriptUrl} | bash`
  const [copyState, setCopyState] = useState<CopyState>({
    status: 'idle',
    target: null,
  })

  useEffect(() => {
    if (copyState.status === 'idle') {
      return undefined
    }

    const timeoutId = window.setTimeout(() => {
      setCopyState({
        status: 'idle',
        target: null,
      })
    }, 1800)

    return () => {
      window.clearTimeout(timeoutId)
    }
  }, [copyState])

  const idleCopyLabel = translate({
    id: 'homepage.install.copy.idle',
    message: 'Copy text',
  })
  const copiedLabel = translate({
    id: 'homepage.install.copy.copied',
    message: 'Copied',
  })
  const errorLabel = translate({
    id: 'homepage.install.copy.error',
    message: 'Copy failed',
  })

  function copyLabelFor(target: string): string {
    if (copyState.target !== target || copyState.status === 'idle') {
      return idleCopyLabel
    }
    if (copyState.status === 'copied') {
      return copiedLabel
    }
    return errorLabel
  }

  async function handleCopy(target: string, text: string): Promise<void> {
    if (typeof navigator === 'undefined' || !navigator.clipboard) {
      setCopyState({
        status: 'error',
        target,
      })
      return
    }

    try {
      await navigator.clipboard.writeText(text)
      setCopyState({
        status: 'copied',
        target,
      })
    }
    catch {
      setCopyState({
        status: 'error',
        target,
      })
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
              <Translate id="homepage.install.title.human">
                Install locally in one line.
              </Translate>
            </h2>
            <p>
              <Translate id="homepage.install.description.human">
                The supported first-run path is a single installer that sets up the CLI and local
                serve flow on macOS and Linux.
              </Translate>
            </p>
          </div>
        </div>

        <div className={styles.frame}>
          <div className={styles.frameHeader}>
            <SectionLabel>
              <Translate id="homepage.install.frameLabel.human">One-liner install</Translate>
            </SectionLabel>
            <span className={styles.platform}>macOS / Linux</span>
          </div>

          <div className={styles.commandBlock}>
            <code className={styles.command}>{installCommand}</code>
          </div>

          <div className={styles.frameFooter}>
            <p className={styles.note}>
              <Translate id="homepage.install.footer.human">
                Installs into ~/.local/share/vllm-sr, writes ~/.local/bin/vllm-sr, and keeps
                Windows on the manual pip flow in the docs.
              </Translate>
            </p>

            <div className={styles.actions}>
              <PillButton
                onClick={() => {
                  void handleCopy('human-install-command', installCommand)
                }}
              >
                {copyLabelFor('human-install-command')}
              </PillButton>
              <PillLink to="/docs/installation" muted>
                <Translate id="homepage.install.primaryCta">
                  Full installation guide
                </Translate>
              </PillLink>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
