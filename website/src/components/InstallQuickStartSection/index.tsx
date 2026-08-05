import React, { useEffect, useState } from 'react'
import clsx from 'clsx'
import Translate, { translate } from '@docusaurus/Translate'
import useDocusaurusContext from '@docusaurus/useDocusaurusContext'
import { PillLink, SectionLabel } from '@site/src/components/site/Chrome'
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

interface CommandShellProps {
  command: string
  copyTarget: string
  copyState: CopyState
  onCopy: (target: string, text: string) => void
  shellLabel?: string
  variant?: 'default' | 'remove'
}

function CommandShell({
  command,
  copyTarget,
  copyState,
  onCopy,
  shellLabel = 'shell',
  variant = 'default',
}: CommandShellProps): JSX.Element {
  const idleCopyLabel = translate({
    id: 'homepage.install.copy.idle',
    message: 'Copy',
  })
  const copiedLabel = translate({
    id: 'homepage.install.copy.copied',
    message: 'Copied',
  })
  const errorLabel = translate({
    id: 'homepage.install.copy.error',
    message: 'Failed',
  })

  const isActive = copyState.target === copyTarget && copyState.status !== 'idle'
  const copyLabel = !isActive
    ? idleCopyLabel
    : copyState.status === 'copied'
      ? copiedLabel
      : errorLabel

  return (
    <div className={clsx(styles.commandShell, variant === 'remove' && styles.commandShellRemove)}>
      <div className={styles.commandToolbar}>
        <div className={styles.terminalMeta}>
          <span className={styles.terminalDots} aria-hidden="true">
            <span className={styles.terminalDot} />
            <span className={styles.terminalDot} />
            <span className={styles.terminalDot} />
          </span>
          <span className={styles.shellLabel}>{shellLabel}</span>
        </div>
        <button
          type="button"
          className={`${styles.copyChip} ${isActive ? styles.copyChipActive : ''} ${copyState.target === copyTarget && copyState.status === 'copied' ? styles.copyChipSuccess : ''}`}
          onClick={() => {
            void onCopy(copyTarget, command)
          }}
          aria-label={translate({
            id: 'homepage.install.copy.aria',
            message: 'Copy command to clipboard',
          })}
        >
          <span className={styles.copyChipIcon} aria-hidden="true">
            {copyState.target === copyTarget && copyState.status === 'copied' ? '✓' : '⧉'}
          </span>
          {copyLabel}
        </button>
      </div>
      <div className={styles.commandBlock}>
        <code className={styles.command}>{command}</code>
      </div>
    </div>
  )
}

export default function InstallQuickStartSection(): JSX.Element {
  const { siteConfig } = useDocusaurusContext()
  const installScriptUrl = buildInstallScriptUrl(siteConfig.url, siteConfig.baseUrl)
  const installCommand = `curl -fsSL ${installScriptUrl} | bash`
  const removeCommand = 'rm -rf ~/.local/share/vllm-sr && rm -f ~/.local/bin/vllm-sr'
  const serveCommand = 'vllm-sr serve --image-pull-policy never'
  const [showFollowup, setShowFollowup] = useState(false)
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
      if (target === 'install-command') {
        setShowFollowup(true)
      }
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
        <header className={styles.heading}>
          <SectionLabel>
            <Translate id="homepage.install.label">Quick start</Translate>
          </SectionLabel>
          <h2>
            <Translate id="homepage.install.title.human">
              Install locally in one line.
            </Translate>
          </h2>
          <p className={styles.lede}>
            <Translate id="homepage.install.meta">
              One supported local path. Copy the installer, run it, then open the dashboard.
            </Translate>
            {' '}
            <span className={styles.ledeMuted}>
              <Translate id="homepage.install.description.human">
                The supported first-run path is a single installer that sets up the CLI and local
                serve flow on macOS and Linux.
              </Translate>
            </span>
          </p>
        </header>

        <div className={styles.frame}>
          <div className={styles.pathMeta}>
            <span className={styles.pathMetaLabel}>
              <Translate id="homepage.install.pathLabel">Local install path</Translate>
            </span>
            <span className={styles.pathMetaChip}>
              {showFollowup
                ? (
                    <Translate id="homepage.install.pathDuration">3 steps · ~2 min</Translate>
                  )
                : (
                    <Translate id="homepage.install.pathDurationShort">One command to start</Translate>
                  )}
            </span>
          </div>

          <ol className={styles.steps}>
            <li className={styles.step}>
              <div className={styles.stepRail} aria-hidden="true">
                <span className={styles.stepMarker}>01</span>
                {showFollowup && <span className={styles.stepLine} />}
              </div>
              <div className={styles.stepBody}>
                <div className={styles.stepHeader}>
                  <h3 className={styles.stepTitle}>
                    <Translate id="homepage.install.step1.title">Install the CLI</Translate>
                  </h3>
                  <span className={styles.platform}>macOS / Linux</span>
                </div>
                <p className={styles.stepHint}>
                  <Translate id="homepage.install.step1.hint">
                    Downloads the installer, prepares Docker, and writes vllm-sr to your PATH.
                  </Translate>
                </p>
                <CommandShell
                  command={installCommand}
                  copyTarget="install-command"
                  copyState={copyState}
                  onCopy={handleCopy}
                />

                <details className={styles.removeDetails}>
                  <summary className={styles.removeSummary}>
                    <Translate id="homepage.install.step1.removeLabel">Remove local install</Translate>
                  </summary>
                  <p className={styles.removeHint}>
                    <Translate id="homepage.install.step1.removeHint">
                      Removes ~/.local/share/vllm-sr and ~/.local/bin/vllm-sr. Stop any running serve session first.
                    </Translate>
                  </p>
                  <CommandShell
                    command={removeCommand}
                    copyTarget="remove-command"
                    copyState={copyState}
                    onCopy={handleCopy}
                    shellLabel="remove"
                    variant="remove"
                  />
                </details>

                {!showFollowup && (
                  <button
                    type="button"
                    className={styles.followupToggle}
                    onClick={() => setShowFollowup(true)}
                  >
                    <Translate id="homepage.install.showFollowup">
                      Show serve &amp; dashboard steps
                    </Translate>
                  </button>
                )}
              </div>
            </li>

            {showFollowup && (
              <>
                <li className={styles.step}>
                  <div className={styles.stepRail} aria-hidden="true">
                    <span className={styles.stepMarker}>02</span>
                    <span className={styles.stepLine} />
                  </div>
                  <div className={styles.stepBody}>
                    <h3 className={styles.stepTitle}>
                      <Translate id="homepage.install.step2.title">Start local serve</Translate>
                    </h3>
                    <p className={styles.stepHint}>
                      <Translate id="homepage.install.step2.hint">
                        Boots the router image and dashboard. Skip if the installer already launched it.
                      </Translate>
                    </p>
                    <CommandShell
                      command={serveCommand}
                      copyTarget="serve-command"
                      copyState={copyState}
                      onCopy={handleCopy}
                      shellLabel="vllm-sr"
                    />
                  </div>
                </li>

                <li className={styles.step}>
                  <div className={styles.stepRail} aria-hidden="true">
                    <span className={styles.stepMarker}>03</span>
                  </div>
                  <div className={styles.stepBody}>
                    <h3 className={styles.stepTitle}>
                      <Translate id="homepage.install.step3.title">Open the dashboard</Translate>
                    </h3>
                    <p className={styles.stepHint}>
                      <Translate id="homepage.install.step3.hint">
                        Setup mode appears on first run. Configure models, then send traffic through the router.
                      </Translate>
                    </p>
                    <a
                      className={styles.dashboardLink}
                      href="http://localhost:8700"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <span className={styles.dashboardLinkLabel}>
                        <Translate id="homepage.install.step3.linkLabel">Local dashboard</Translate>
                      </span>
                      <code className={styles.dashboardUrl}>localhost:8700</code>
                      <span className={styles.dashboardArrow} aria-hidden="true">↗</span>
                    </a>
                  </div>
                </li>
              </>
            )}
          </ol>

          <div className={styles.frameFooter}>
            <p className={styles.note}>
              <Translate id="homepage.install.footer.human">
                Installs into ~/.local/share/vllm-sr, writes ~/.local/bin/vllm-sr, and keeps
                Windows on the manual pip flow in the docs.
              </Translate>
            </p>

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
        </div>
      </div>
    </section>
  )
}
