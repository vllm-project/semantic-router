import React, { useState } from 'react'
import Link from '@docusaurus/Link'
import Translate, { translate } from '@docusaurus/Translate'
import { FiTerminal, FiUser } from 'react-icons/fi'
import { SectionLabel } from '@site/src/components/site/Chrome'
import Terminal from '@site/src/components/duo/Terminal'
import duo from '@site/src/components/duo/duo.module.css'
import {
  AGENT_INSTALL_DOC_PATH,
  AGENT_INSTALL_PROMPT,
  AGENT_SKILL_PATH,
  CURL_INSTALL_COMMAND,
} from '@site/src/data/installation'
import styles from './index.module.css'

type InstallAudience = 'human' | 'agent'

export default function InstallQuickStartSection(): JSX.Element {
  const [activeAudience, setActiveAudience] = useState<InstallAudience>('human')
  const copyText = activeAudience === 'human'
    ? CURL_INSTALL_COMMAND
    : AGENT_INSTALL_PROMPT

  function selectAudience(audience: InstallAudience): void {
    setActiveAudience(audience)
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

  const audienceIcon = activeAudience === 'human' ? <FiUser aria-hidden="true" /> : <FiTerminal aria-hidden="true" />

  return (
    <section id="install-quickstart" className={styles.section}>
      <div className={styles.wrap}>
        <header className="site-section-intro">
          <SectionLabel>
            <Translate id="homepage.install.label">Installation</Translate>
          </SectionLabel>
          <h2>
            {activeAudience === 'human'
              ? (
                  <Translate id="homepage.install.title.human">
                    Get started in one line
                  </Translate>
                )
              : (
                  <Translate id="homepage.install.title.agent">
                    Let your agent set it up
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
            <FiUser className={styles.audienceIcon} aria-hidden="true" />
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
            <FiTerminal className={styles.audienceIcon} aria-hidden="true" />
            <Translate id="homepage.install.audience.agent">For agents</Translate>
          </button>
        </div>

        <div
          id="install-panel"
          role="tabpanel"
          aria-labelledby={`install-tab-${activeAudience}`}
          className={styles.way}
        >
          <div className={styles.wayCopy}>
            <p className={styles.brand}>
              {audienceIcon}
              {activeAudience === 'human'
                ? <Translate id="homepage.install.audience.human">For humans</Translate>
                : <Translate id="homepage.install.audience.agent">For agents</Translate>}
            </p>
            <p className={styles.channelNote}>
              <strong><Translate id="homepage.install.channel">Development channel</Translate></strong>
              <span aria-hidden="true"> · </span>
              <Translate id="homepage.install.channelDescription">Matches the latest docs, including the built-in model runtime.</Translate>
            </p>
            <div className={styles.actions}>
              {activeAudience === 'agent'
                ? (
                    <Link className={styles.guide} to={AGENT_INSTALL_DOC_PATH}>
                      <Translate id="homepage.install.agentCta">
                        Agent installation guide
                      </Translate>
                    </Link>
                  )
                : (
                    <Link className={styles.guide} to="/docs/installation">
                      <Translate id="homepage.install.primaryCta">
                        Installation guide
                      </Translate>
                    </Link>
                  )}
              {activeAudience === 'agent'
                ? (
                    <a className={duo.more} href={AGENT_SKILL_PATH} rel="noopener noreferrer" target="_blank">
                      <Translate id="homepage.install.secondaryCta">View raw skill</Translate>
                    </a>
                  )
                : (
                    <Link className={duo.more} to="/docs/intro">
                      <Translate id="homepage.install.docsCta">Read the docs</Translate>
                    </Link>
                  )}
            </div>
          </div>

          <Terminal
            className={styles.terminal}
            text={copyText}
            shell={activeAudience === 'human'}
            caption={activeAudience === 'human'
              ? <Translate id="homepage.install.caption.human">terminal</Translate>
              : <Translate id="homepage.install.caption.agent">prompt for your agent</Translate>}
          />
        </div>
      </div>
    </section>
  )
}
