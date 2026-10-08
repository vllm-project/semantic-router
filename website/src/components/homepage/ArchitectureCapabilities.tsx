import React from 'react'
import Translate from '@docusaurus/Translate'
import useBaseUrl from '@docusaurus/useBaseUrl'
import ThemedImage from '@theme/ThemedImage'
import Claude from '@lobehub/icons/es/Claude/components/Mono'
import DeepSeek from '@lobehub/icons/es/DeepSeek/components/Mono'
import Mistral from '@lobehub/icons/es/Mistral/components/Mono'
import OpenAI from '@lobehub/icons/es/OpenAI/components/Mono'
import {
  FiBox,
  FiCloud,
  FiCpu,
  FiDollarSign,
  FiGlobe,
  FiLock,
  FiMessageSquare,
  FiPlus,
  FiServer,
  FiSliders,
  FiStar,
  FiZap,
} from 'react-icons/fi'
import ScrollReveal from '@site/src/components/site/ScrollReveal'
import { SectionLabel, StatStrip } from '@site/src/components/site/Chrome'
import shared from './homepageShared.module.css'
import styles from './ArchitectureCapabilities.module.css'

type ArchitectureCapabilitiesProps = {
  stats: React.ComponentProps<typeof StatStrip>['items']
}

function Models(): JSX.Element {
  return (
    <article className={`${styles.dimension} ${styles.models}`}>
      <header className={styles.dimensionHeader}>
        <FiBox aria-hidden="true" />
        <h3><Translate id="homepage.capabilities.axis.models">Models</Translate></h3>
        <span aria-hidden="true">01</span>
      </header>
      <div className={styles.modelPalette}>
        <div className={styles.modelFamily}>
          <span className={styles.visualLabel}>
            <Translate id="homepage.architecture.models.closed">Closed models</Translate>
          </span>
          <div className={styles.modelPair}>
            <span className={styles.model}>
              <Claude size={21} aria-hidden="true" />
              <strong>Claude</strong>
            </span>
            <span className={styles.model}>
              <OpenAI size={21} aria-hidden="true" />
              <strong>OpenAI</strong>
            </span>
          </div>
        </div>
        <div className={styles.modelFamily}>
          <span className={styles.visualLabel}>
            <Translate id="homepage.architecture.models.open">Open models</Translate>
          </span>
          <div className={`${styles.modelPair} ${styles.openModels}`}>
            <span className={styles.model}>
              <DeepSeek size={21} aria-hidden="true" />
              <strong>DeepSeek</strong>
            </span>
            <span className={styles.model}>
              <Mistral size={21} aria-hidden="true" />
              <strong>Mistral</strong>
            </span>
          </div>
        </div>
      </div>
      <p><Translate id="homepage.architecture.models.copy">Different strengths. One model fleet.</Translate></p>
    </article>
  )
}

function Compute(): JSX.Element {
  return (
    <article className={`${styles.dimension} ${styles.compute}`}>
      <header className={styles.dimensionHeader}>
        <FiCpu aria-hidden="true" />
        <h3><Translate id="homepage.capabilities.axis.compute">Compute</Translate></h3>
        <span aria-hidden="true">02</span>
      </header>
      <div className={styles.computeTiles}>
        <div className={styles.computeTile}>
          <div className={styles.gpuNode} aria-hidden="true">
            <FiCpu />
            <span>
              <i />
              <i />
              <i />
              <i />
            </span>
          </div>
          <strong><Translate id="homepage.architecture.compute.gpu">GPU nodes</Translate></strong>
        </div>
        <div className={styles.computeTile}>
          <div className={styles.cpuNode} aria-hidden="true">
            <FiServer />
            <span>CPU</span>
          </div>
          <strong><Translate id="homepage.architecture.compute.runtime">Local runtime</Translate></strong>
        </div>
        <div className={styles.computeTile}>
          <div className={styles.apiNode} aria-hidden="true">
            <FiCloud />
            <span>API</span>
          </div>
          <strong><Translate id="homepage.architecture.compute.api">Model APIs</Translate></strong>
        </div>
      </div>
      <div className={styles.backendTags}>
        <span>vLLM</span>
        <span>SGLang</span>
        <span>OpenAI API</span>
      </div>
      <p><Translate id="homepage.architecture.compute.copy">Route to the backends you configure.</Translate></p>
    </article>
  )
}

function Location(): JSX.Element {
  return (
    <article className={`${styles.dimension} ${styles.location}`}>
      <header className={styles.dimensionHeader}>
        <FiGlobe aria-hidden="true" />
        <h3><Translate id="homepage.capabilities.axis.location">Location</Translate></h3>
        <span aria-hidden="true">03</span>
      </header>
      <div className={styles.placement}>
        <svg className={styles.globe} viewBox="0 0 150 150" aria-hidden="true" focusable="false">
          <circle className={styles.globeHalo} cx="75" cy="75" r="70" />
          <circle cx="75" cy="75" r="52" />
          <ellipse cx="75" cy="75" rx="23" ry="52" />
          <ellipse cx="75" cy="75" rx="52" ry="22" />
          <path d="M23 75h104M75 23v104" />
          <path className={styles.placementRoute} d="M43 51 Q100 38 111 79 Q83 118 57 103" />
          <circle className={styles.placementPoint} cx="43" cy="51" r="5" />
          <circle className={styles.placementPoint} cx="111" cy="79" r="5" />
          <circle className={styles.placementPoint} cx="57" cy="103" r="5" />
        </svg>
        <div className={styles.placementLabels}>
          <span>
            <FiCloud aria-hidden="true" />
            <Translate id="homepage.architecture.location.cloud">Cloud</Translate>
          </span>
          <span>
            <FiLock aria-hidden="true" />
            <Translate id="homepage.architecture.location.private">Private</Translate>
          </span>
          <span>
            <FiCpu aria-hidden="true" />
            <Translate id="homepage.architecture.location.edge">Edge</Translate>
          </span>
        </div>
      </div>
      <p><Translate id="homepage.architecture.location.copy">You configure where inference runs.</Translate></p>
    </article>
  )
}

function Preference(): JSX.Element {
  return (
    <article className={`${styles.dimension} ${styles.preference}`}>
      <header className={styles.dimensionHeader}>
        <FiSliders aria-hidden="true" />
        <h3><Translate id="homepage.capabilities.axis.preference">Preference</Translate></h3>
        <span aria-hidden="true">04</span>
      </header>
      <div className={styles.priorities}>
        <div>
          <span>
            <FiStar aria-hidden="true" />
            <Translate id="homepage.architecture.preference.quality">Quality</Translate>
          </span>
          <span className={styles.priorityRail} style={{ '--weight': '78%' } as React.CSSProperties} aria-hidden="true"><i /></span>
        </div>
        <div>
          <span>
            <FiZap aria-hidden="true" />
            <Translate id="homepage.architecture.preference.latency">Latency</Translate>
          </span>
          <span className={styles.priorityRail} style={{ '--weight': '55%' } as React.CSSProperties} aria-hidden="true"><i /></span>
        </div>
        <div>
          <span>
            <FiDollarSign aria-hidden="true" />
            <Translate id="homepage.architecture.preference.cost">Cost</Translate>
          </span>
          <span className={styles.priorityRail} style={{ '--weight': '35%' } as React.CSSProperties} aria-hidden="true"><i /></span>
        </div>
      </div>
      <p><Translate id="homepage.architecture.preference.copy">Set priorities for each task.</Translate></p>
    </article>
  )
}

function DecisionCenter(): JSX.Element {
  const sources = {
    light: useBaseUrl('/img/vllm-sr-logo.light.png'),
    dark: useBaseUrl('/img/vllm-sr-logo.white.png'),
  }

  return (
    <div className={styles.center}>
      <div className={styles.orbit} aria-hidden="true">
        <i />
        <i />
        <i />
        <i />
      </div>
      <div className={styles.decisionCenter}>
        <span className={styles.controlLabel}>
          <Translate id="homepage.architecture.center.label">Programmable decision layer</Translate>
        </span>
        <div className={styles.contextTags}>
          <span>
            <FiMessageSquare aria-hidden="true" />
            <Translate id="homepage.architecture.center.context">Context</Translate>
          </span>
          <FiPlus aria-hidden="true" />
          <span>
            <FiSliders aria-hidden="true" />
            <Translate id="homepage.architecture.center.policy">Policy</Translate>
          </span>
        </div>
        <span className={styles.routerBrand}>
          <ThemedImage sources={sources} alt="vLLM Semantic Router" />
        </span>
        <p>
          <Translate id="homepage.architecture.center.promise">
            Bring the right intelligence to every request.
          </Translate>
        </p>
        <div className={styles.selection}>
          <span className={styles.visualLabel}>
            <Translate id="homepage.architecture.center.selection">Per-request choice</Translate>
          </span>
          <div>
            <span>
              <FiBox aria-hidden="true" />
              <Translate id="homepage.architecture.center.model">Model</Translate>
            </span>
            <FiPlus aria-hidden="true" />
            <span>
              <FiServer aria-hidden="true" />
              <Translate id="homepage.architecture.center.backend">Backend</Translate>
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default function ArchitectureCapabilities({ stats }: ArchitectureCapabilitiesProps): JSX.Element {
  return (
    <section id="architecture" className={shared.bandSection} aria-labelledby="architecture-title">
      <div className={`site-shell-container ${shared.sectionInner}`}>
        <ScrollReveal>
          <header className={`site-section-intro ${shared.sectionHeader}`}>
            <SectionLabel><Translate id="homepage.capabilities.label">Architecture</Translate></SectionLabel>
            <h2 id="architecture-title" className={shared.sectionTitle}>
              <Translate id="homepage.capabilities.heading">Your models. Your rules.</Translate>
            </h2>
            <p className={shared.sectionSubtitle}>
              <Translate id="homepage.capabilities.description">Choose models and compute for each request.</Translate>
            </p>
          </header>
        </ScrollReveal>

        <ScrollReveal delay={60}>
          <div className={styles.frame}>
            <div className={styles.landscape}>
              <svg className={styles.connections} viewBox="0 0 1000 500" preserveAspectRatio="none" aria-hidden="true" focusable="false">
                <path d="M200 120H340Q360 120 360 140V230Q360 250 380 250H500" />
                <path d="M200 380H340Q360 380 360 360V270Q360 250 380 250H500" />
                <path d="M800 120H660Q640 120 640 140V230Q640 250 620 250H500" />
                <path d="M800 380H660Q640 380 640 360V270Q640 250 620 250H500" />
              </svg>
              <Models />
              <Compute />
              <Location />
              <Preference />
              <DecisionCenter />
            </div>
            <div className={styles.stats}><StatStrip items={stats} /></div>
          </div>
        </ScrollReveal>
      </div>
    </section>
  )
}
