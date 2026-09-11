import React, { useEffect, useState } from 'react'
import Translate, { translate } from '@docusaurus/Translate'
import { PillLink, SectionLabel } from '@site/src/components/site/Chrome'
import {
  AGENT_INSTALL_PROMPT,
  AGENT_SKILL_PATH,
  CURL_INSTALL_COMMAND,
} from '@site/src/data/installation'
import styles from './index.module.css'

type CopyStatus = 'idle' | 'copied' | 'error'
type InstallAudience = 'human' | 'agent'

export default function InstallQuickStartSection(): JSX.Element {
  const [activeAudience, setActiveAudience] = useState<InstallAudience>('human')
  const [copyStatus, setCopyStatus] = useState<CopyStatus>('idle')
  const copyText = activeAudience === 'human'
    ? CURL_INSTALL_COMMAND
    : AGENT_INSTALL_PROMPT

  function selectAudience(audience: InstallAudience): void {
    setActiveAudience(audience)
    setCopyStatus('idle')
  }

  function handleAudienceNavigation(
    event: React.KeyboardEvent<HTMLButtonElement>,
  ): void {
    let audience: InstallAudience | undefined
    if (event.key === 'ArrowLeft' || event.key === 'Home') {
      audience = 'human'
    }
    else if (event.key === 'ArrowRight' || event.key === 'End') {
      audience = 'agent'
    }

    if (!audience) {
      return
    }

    event.preventDefault()
    selectAudience(audience)
    document.getElementById(`install-tab-${audience}`)?.focus()
  }

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
      await navigator.clipboard.writeText(copyText)
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
            <Translate id="homepage.install.label">Installation</Translate>
          </SectionLabel>
          <h2>
            {activeAudience === 'human'
              ? (
                  <Translate id="homepage.install.title.human">
                    Install locally in one line
                  </Translate>
                )
              : (
                  <Translate id="homepage.install.title.agent">
                    Hand the setup to your agent
                  </Translate>
                )}
          </h2>
        </header>

        <div
          className={styles.audienceTabs}
          role="tablist"
          aria-label={translate({
            id: 'homepage.install.audience.aria',
            message: 'Choose an installation audience',
          })}
        >
          <button
            id="install-tab-human"
            type="button"
            role="tab"
            aria-selected={activeAudience === 'human'}
            aria-controls="install-panel"
            tabIndex={activeAudience === 'human' ? 0 : -1}
            className={`${styles.audienceTab} ${activeAudience === 'human' ? styles.audienceTabActive : ''}`}
            onClick={() => {
              selectAudience('human')
            }}
            onKeyDown={handleAudienceNavigation}
          >
            <Translate id="homepage.install.audience.human">For humans</Translate>
          </button>
          <button
            id="install-tab-agent"
            type="button"
            role="tab"
            aria-selected={activeAudience === 'agent'}
            aria-controls="install-panel"
            tabIndex={activeAudience === 'agent' ? 0 : -1}
            className={`${styles.audienceTab} ${activeAudience === 'agent' ? styles.audienceTabActive : ''}`}
            onClick={() => {
              selectAudience('agent')
            }}
            onKeyDown={handleAudienceNavigation}
          >
            <Translate id="homepage.install.audience.agent">For agents</Translate>
          </button>
        </div>

        <div
          id="install-panel"
          role="tabpanel"
          aria-labelledby={`install-tab-${activeAudience}`}
          className={`${styles.commandShell} ${activeAudience === 'agent' ? styles.commandShellAgent : ''}`}
        >
          <span className={styles.commandPrompt} aria-hidden="true">
            {activeAudience === 'human' ? '$' : '›'}
          </span>
          <code
            className={`${styles.command} ${activeAudience === 'agent' ? styles.agentPrompt : ''}`}
          >
            {copyText}
          </code>
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
          {activeAudience === 'agent'
            ? (
                <PillLink
                  className={styles.guideLink}
                  href={AGENT_SKILL_PATH}
                >
                  <Translate id="homepage.install.agentCta">
                    View agent skill
                  </Translate>
                </PillLink>
              )
            : (
                <PillLink className={styles.guideLink} to="/docs/installation">
                  <Translate id="homepage.install.primaryCta">
                    Installation guide
                  </Translate>
                </PillLink>
              )}
          {activeAudience === 'agent'
            ? (
                <PillLink className={styles.docsLink} to="/docs/installation" muted>
                  <Translate id="homepage.install.secondaryCta">Open quickstart</Translate>
                </PillLink>
              )
            : (
                <PillLink className={styles.docsLink} to="/docs/intro" muted>
                  <Translate id="homepage.install.docsCta">Read the docs</Translate>
                </PillLink>
              )}
        </div>
      </div>
    </section>
  )
}
