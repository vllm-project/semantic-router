import React, { useMemo } from 'react'
import { motion } from 'framer-motion'
import styles from './OpenClawPage.module.css'
import {
  OPENCLAW_FEATURES,
  type OpenClawStatus,
} from './OpenClawPageSupport'

export const ArchitectureTab: React.FC<{
  containers: OpenClawStatus[]
}> = ({ containers }) => {
  const kernelModules = useMemo(() => OPENCLAW_FEATURES, [])

  const topClawNodes = useMemo(() => {
    const noRedNodeLabels = new Set(['memory claw', 'analyst claw'])
    const primary = containers
      .slice()
      .sort((a, b) => Number(b.healthy) - Number(a.healthy) || Number(b.running) - Number(a.running))
      .slice(0, 4)
      .map(container => {
        const label = (container.agentName || container.containerName).trim() || container.containerName
        const rawState = container.healthy ? 'healthy' : container.running ? 'starting' : 'stopped'
        const state = noRedNodeLabels.has(label.toLowerCase()) && rawState === 'stopped' ? 'healthy' : rawState
        return {
          id: container.containerName,
          label,
          role: (container.agentRole || 'Claw Agent').trim() || 'Claw Agent',
          state,
        }
      })

    const fallback = [
      { id: 'routing-claw', label: 'Routing Claw', role: 'Intent Router', state: 'healthy' as const },
      { id: 'guard-claw', label: 'Guard Claw', role: 'Safety Guard', state: 'healthy' as const },
      { id: 'memory-claw', label: 'Memory Claw', role: 'Context Keeper', state: 'healthy' as const },
      { id: 'planner-claw', label: 'Planner Claw', role: 'Task Planner', state: 'healthy' as const },
    ]

    const used = new Set(primary.map(node => node.label.toLowerCase()))
    const merged = [...primary]
    for (const node of fallback) {
      if (merged.length >= 4) break
      if (used.has(node.label.toLowerCase())) continue
      merged.push(node)
      used.add(node.label.toLowerCase())
    }
    return merged
  }, [containers])

  const modelNodes = useMemo(() => {
    return [
      { name: 'General Domain · Small', family: 'Fast intent triage, concise answers, low-latency chat turns.' },
      { name: 'General Domain · Large', family: 'Deep multi-turn dialogue, synthesis, and broader world knowledge.' },
      { name: 'Coding Domain · Small', family: 'Lightweight code edits, lint-aware fixes, and script scaffolding.' },
      { name: 'Coding Domain · Large', family: 'Architecture refactors, debugging traces, and complex code generation.' },
      { name: 'Multimodal Vision-Language Pool', family: 'Image-grounded understanding, visual QA, and scene-aware dialogue.' },
      { name: 'Multimodal Audio-Speech Pool', family: 'Speech understanding, voice instructions, and audio-event interpretation.' },
      { name: 'Multimodal Document-Insight Pool', family: 'PDF/table/chart comprehension with cross-page evidence synthesis.' },
      { name: 'Multimodal Action-Orchestration Pool', family: 'Grounded tool calling over text, image, and structured interface states.' },
    ]
  }, [])

  return (
    <div className={styles.teamDashboard}>
      <section className={styles.productSection}>
        <div className={styles.kernelIntro}>
          <div className={styles.kernelIntroBody}>
            <span className={styles.kernelBadge}>Full Mesh</span>
            <h3 className={styles.kernelTitle}>Claw Operating System</h3>
            <p className={styles.kernelSubtitle}>
              Top-layer Claws route intent into Semantic Router. Kernel capabilities then project requests into the model mesh.
            </p>
          </div>
          <div className={styles.kernelMeta}>
            <span>Layered Architecture View</span>
            <span>Conceptual Flow</span>
            <span>Not Runtime Inventory</span>
          </div>
        </div>

        <div className={styles.kernelFlow}>
          <motion.div
            className={styles.kernelLayerRail}
            initial={{ opacity: 0, y: -8 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.35 }}
            transition={{ duration: 0.28 }}
          >
            <div className={styles.kernelLayerLabel}>Layer 1 · Claw Layer</div>
            <div className={`${styles.kernelNodeGrid} ${styles.kernelNodeGridTwoByTwo}`}>
              {topClawNodes.map((node, index) => (
                <motion.article
                  key={node.id}
                  className={`${styles.kernelNodeCard} ${styles.kernelClawNodeCard} ${
                    node.state === 'healthy'
                      ? styles.kernelNodeHealthy
                      : node.state === 'starting'
                        ? styles.kernelNodeStarting
                        : styles.kernelNodeStopped
                  }`}
                  initial={{ opacity: 0, y: -8 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true, amount: 0.4 }}
                  transition={{ duration: 0.28, delay: index * 0.05 }}
                >
                  <div className={styles.kernelClawNodeHead}>
                    <span className={styles.kernelClawNodeLogoWrap}>
                      <img className={styles.kernelClawNodeLogo} src="/openclaw.svg" alt="" aria-hidden="true" />
                    </span>
                    <div className={styles.kernelNodeTitle}>{node.label}</div>
                  </div>
                  <div className={styles.kernelNodeMeta}>{node.role}</div>
                </motion.article>
              ))}
            </div>
          </motion.div>

          <motion.div
            className={styles.kernelFlowConnector}
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true, amount: 0.2 }}
            transition={{ duration: 0.25, delay: 0.1 }}
          >
            <motion.span
              className={styles.kernelFlowConnectorDot}
              animate={{ y: [-4, 4, -4], opacity: [0.65, 1, 0.65] }}
              transition={{ duration: 2.2, repeat: Infinity, ease: 'easeInOut' }}
            />
          </motion.div>

          <motion.div
            className={`${styles.kernelLayerRail} ${styles.kernelLayerCore}`}
            initial={{ opacity: 0, scale: 0.97 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true, amount: 0.35 }}
            transition={{ duration: 0.35 }}
          >
            <div className={styles.kernelLayerLabel}>Layer 2 · Semantic Router</div>
            <div className={styles.kernelCore}>
              <motion.div
                className={styles.kernelPulse}
                animate={{ scale: [1, 1.08, 1], opacity: [0.35, 0.62, 0.35] }}
                transition={{ duration: 3.2, repeat: Infinity, ease: 'easeInOut' }}
              />
              <div className={styles.kernelCoreHeader}>
                <span className={styles.kernelCoreBadge}>Operating System</span>
                <span className={styles.kernelCoreHint}>Module Management Layer</span>
              </div>
              <div className={styles.kernelCoreLead}>
                <h4 className={styles.kernelCoreTitle}>Signal Driven Decision Runtime</h4>
                <p className={styles.kernelCoreDescription}>
                  Semantic Router acts as a control plane: it manages routing policy, safety, context lifecycle,
                  cross-claw memory sharing, and isolation before dispatching requests to model pools.
                </p>
              </div>
              <div className={styles.kernelFeatureGrid}>
                {kernelModules.map((feature, index) => (
                  <motion.article
                    key={feature.title}
                    className={styles.kernelFeatureCard}
                    initial={{ opacity: 0, y: 8 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true, amount: 0.3 }}
                    transition={{ duration: 0.28, delay: index * 0.04 }}
                  >
                    <div className={styles.kernelFeatureIcon}>{feature.icon}</div>
                    <div className={styles.kernelFeatureBody}>
                      <div className={styles.kernelFeatureTag}>{feature.module}</div>
                      <h4 className={styles.kernelFeatureTitle}>{feature.title}</h4>
                      <p className={styles.kernelFeatureDescription}>{feature.description}</p>
                    </div>
                  </motion.article>
                ))}
              </div>
            </div>
          </motion.div>

          <motion.div
            className={styles.kernelFlowConnector}
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true, amount: 0.2 }}
            transition={{ duration: 0.25, delay: 0.16 }}
          >
            <motion.span
              className={styles.kernelFlowConnectorDot}
              animate={{ y: [-4, 4, -4], opacity: [0.65, 1, 0.65] }}
              transition={{ duration: 2.2, repeat: Infinity, ease: 'easeInOut', delay: 0.45 }}
            />
          </motion.div>

          <motion.div
            className={styles.kernelLayerRail}
            initial={{ opacity: 0, y: 8 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.35 }}
            transition={{ duration: 0.28 }}
          >
            <div className={styles.kernelLayerLabel}>Layer 3 · Model Layer</div>
            <div className={`${styles.kernelNodeGrid} ${styles.kernelModelGrid}`}>
              {modelNodes.map((model, index) => (
                <motion.article
                  key={model.name}
                  className={`${styles.kernelNodeCard} ${styles.kernelModelCard}`}
                  initial={{ opacity: 0, y: 8 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true, amount: 0.35 }}
                  transition={{ duration: 0.28, delay: index * 0.05 }}
                >
                  <div className={styles.kernelNodeTitle}>{model.name}</div>
                  <div className={styles.kernelNodeMeta}>{model.family}</div>
                </motion.article>
              ))}
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  )
}
