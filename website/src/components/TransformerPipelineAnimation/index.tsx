import React, { useState } from 'react'
import { motion, useInView } from 'motion/react'
import styles from './index.module.css'

type BranchMode = 'cls' | 'token' | 'pool' | 'cross'

const TOKENS = ['[CLS]', 'Is', 'machine', 'learning', 'related', 'to', 'AI', '?', '[SEP]']

const ENCODER_SUBS = [
  { label: 'Multi-Head Attention', cls: styles.subAttention, icon: '🔗' },
  { label: 'Add & Norm', cls: styles.subNorm, icon: '➕' },
  { label: 'Feed-Forward', cls: styles.subFFN, icon: '⚙️' },
  { label: 'Add & Norm', cls: styles.subNorm, icon: '➕' },
]

const BRANCH_DATA: Record<BranchMode, {
  icon: string
  title: string
  detail: string
  signals: string[]
  cls: string
  tagCls: string
  taskType: string
}> = {
  cls: {
    icon: '🎯',
    title: 'Sentence-Level (CLS Token)',
    detail: '[CLS] → Linear Head → "computer_science"',
    signals: ['Domain', 'Jailbreak', 'Fact-check', 'Feedback', 'Modality'],
    cls: styles.branchCLS,
    tagCls: styles.signalTagCLS,
    taskType: 'SEQ_CLS',
  },
  token: {
    icon: '🏷️',
    title: 'Token-Level (Per Token)',
    detail: 'Each token → BIO Label → O O B-LOC I-LOC O',
    signals: ['PII Detection'],
    cls: styles.branchToken,
    tagCls: styles.signalTagToken,
    taskType: 'TOKEN_CLS',
  },
  pool: {
    icon: '🌊',
    title: 'Bi-Encoder',
    detail: 'mean-pooling(h₁..hₙ) → [0.23, -0.41, 0.87, ...]',
    signals: ['Semantic Cache', 'Similarity', 'Complexity-CL', 'Jailbreak-CL'],
    cls: styles.branchPool,
    tagCls: styles.signalTagPool,
    taskType: 'EMBEDDING',
  },
  cross: {
    icon: '🔀',
    title: 'Cross-Encoder',
    detail: '[CLS] query [SEP] candidate [SEP] → score',
    signals: ['Rerank', 'Multi-Modal'],
    cls: styles.branchCross,
    tagCls: styles.signalTagCross,
    taskType: 'CROSS_LEARNING',
  },
}

/* ------------------------------------------------------------------ */

const DownArrow: React.FC = () => (
  <div className={styles.downArrow}>
    <div className={styles.downArrowLine} />
  </div>
)

/* ---- Animated stage wrappers ---- */
const stageVariants = {
  hidden: { opacity: 0, y: -16 },
  visible: { opacity: 1, y: 0 },
}

const TransformerPipelineAnimation: React.FC = () => {
  const [mode, setMode] = useState<BranchMode>('cls')
  const ref = React.useRef<HTMLDivElement>(null)
  const inView = useInView(ref, { once: true, margin: '-80px' })

  const containerVariants = {
    hidden: {},
    visible: {
      transition: {
        staggerChildren: 0.18,
        delayChildren: 0.1,
      },
    },
  }

  return (
    <div ref={ref} className={styles.pipelineWrapper}>
      {/* Mode tabs */}
      <div className={styles.tabRow}>
        {(['cls', 'token', 'pool', 'cross'] as BranchMode[]).map(m => (
          <button
            key={m}
            className={`${styles.tab} ${mode === m ? styles.tabActive : ''}`}
            onClick={() => setMode(m)}
          >
            {BRANCH_DATA[m].icon}
            {' '}
            {BRANCH_DATA[m].title.split('(')[0].trim()}
          </button>
        ))}
      </div>

      {/* Pipeline — vertical layout */}
      <motion.div
        className={styles.pipeline}
        variants={containerVariants}
        initial="hidden"
        animate={inView ? 'visible' : 'hidden'}
      >
        {/* Stage 1: Input */}
        <motion.div className={styles.stage} variants={stageVariants} transition={{ duration: 0.5 }}>
          <span className={styles.stageLabel}>Input</span>
          <div className={styles.inputBox}>
            &quot;Is machine learning related to AI?&quot;
          </div>
        </motion.div>

        <DownArrow />

        {/* Stage 2: Tokenizer */}
        <motion.div className={styles.stage} variants={stageVariants} transition={{ duration: 0.5 }}>
          <span className={styles.stageLabel}>Tokenizer</span>
          <div className={styles.tokenRow}>
            {TOKENS.map((t, i) => (
              <motion.span
                key={t + i}
                className={`${styles.token} ${t.startsWith('[') ? styles.tokenSpecial : styles.tokenNormal}`}
                initial={{ opacity: 0, scale: 0.7 }}
                animate={inView ? { opacity: 1, scale: 1 } : {}}
                transition={{ delay: 0.4 + i * 0.06, duration: 0.3 }}
              >
                {t}
              </motion.span>
            ))}
          </div>
        </motion.div>

        <DownArrow />

        {/* Stage 3: Embedding */}
        <motion.div className={styles.stage} variants={stageVariants} transition={{ duration: 0.5 }}>
          <span className={styles.stageLabel}>Embedding</span>
          <div className={styles.embeddingStack}>
            <div className={`${styles.embeddingBar} ${styles.embBarToken}`} />
            <span className={styles.embLabel}>Token Emb</span>
            <div className={`${styles.embeddingBar} ${styles.embBarSeg}`} />
            <span className={styles.embLabel}>Segment Emb</span>
            <div className={`${styles.embeddingBar} ${styles.embBarPos}`} />
            <span className={styles.embLabel}>Position Emb</span>
            <div className={`${styles.embeddingBar} ${styles.embBarSum}`} />
            <span className={styles.embLabel}>h₀ = Σ</span>
          </div>
        </motion.div>

        <DownArrow />

        {/* Stage 4: Encoder Block */}
        <motion.div className={styles.stage} variants={stageVariants} transition={{ duration: 0.5 }}>
          <span className={styles.stageLabel}>Encoder Block</span>
          <div className={styles.encoderBlock}>
            <span className={styles.encoderRepeat}>×N</span>
            {ENCODER_SUBS.map((sub, i) => (
              <motion.div
                key={sub.label + i}
                className={`${styles.encoderSub} ${sub.cls}`}
                initial={{ opacity: 0, y: 8 }}
                animate={inView ? { opacity: 1, y: 0 } : {}}
                transition={{ delay: 1.0 + i * 0.12, duration: 0.35 }}
              >
                <span>{sub.icon}</span>
                <span>{sub.label}</span>
              </motion.div>
            ))}
          </div>
        </motion.div>

        <DownArrow />

        {/* Stage 5: Signals */}
        <motion.div
          className={styles.branchesArea}
          variants={stageVariants}
          transition={{ duration: 0.5 }}
        >
          <span className={styles.stageLabel}>Signals</span>
          <div className={styles.branchGrid}>
            {(['cls', 'token', 'pool', 'cross'] as BranchMode[]).map((key) => {
              const b = BRANCH_DATA[key]
              const isActive = mode === key
              return (
                <motion.div
                  key={key}
                  className={`${styles.branch} ${b.cls}`}
                  initial={{ opacity: 0, y: 10 }}
                  animate={inView
                    ? {
                        opacity: isActive ? 1 : 0.4,
                        y: 0,
                        scale: isActive ? 1 : 0.96,
                      }
                    : {}}
                  transition={{ duration: 0.2 }}
                  onClick={() => setMode(key)}
                  whileHover={{ scale: 1.02 }}
                >
                  <span className={styles.branchIcon}>{b.icon}</span>
                  <div className={styles.branchContent}>
                    <span className={styles.branchTitle}>{b.title}</span>
                    <span className={styles.branchDetail}>{b.detail}</span>
                    <span className={styles.branchDetail} style={{ fontStyle: 'italic' }}>
                      {'TaskType: '}
                      {b.taskType}
                    </span>
                    <div className={styles.branchSignals}>
                      {b.signals.map(s => (
                        <span key={s} className={`${styles.signalTag} ${b.tagCls}`}>{s}</span>
                      ))}
                    </div>
                  </div>
                </motion.div>
              )
            })}
          </div>
        </motion.div>
      </motion.div>
    </div>
  )
}

export default TransformerPipelineAnimation
