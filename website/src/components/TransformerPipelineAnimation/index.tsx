import React, { useState } from 'react'
import { motion, useInView } from 'motion/react'
import styles from './index.module.css'

export type BranchMode = 'cls' | 'token' | 'pool' | 'cross'

interface TransformerPipelineAnimationProps {
  mode?: BranchMode
  onModeChange?: (mode: BranchMode) => void
}

const TOKENS = ['[CLS]', 'Is', 'machine', 'learning', 'related', 'to', 'AI', '?', '[SEP]']

const ENCODER_SUBS = [
  { label: 'Multi-Head Attention', cls: styles.subAttention, icon: 'ATTN' },
  { label: 'Add & Norm', cls: styles.subNorm, icon: 'NORM' },
  { label: 'Feed-Forward', cls: styles.subFFN, icon: 'FFN' },
  { label: 'Add & Norm', cls: styles.subNorm, icon: 'NORM' },
]

const BRANCH_DATA: Record<BranchMode, {
  icon: string
  tabLabel: string
  title: string
  detail: string
  signals: string[]
  cls: string
  tagCls: string
  taskType: string
}> = {
  cls: {
    icon: 'CLS',
    tabLabel: 'Sequence',
    title: 'Sentence-Level (CLS Token)',
    detail: '[CLS] → Linear Head → "computer science"',
    signals: ['Domain', 'Jailbreak', 'Fact-check', 'Feedback', 'Modality'],
    cls: styles.branchCLS,
    tagCls: styles.signalTagCLS,
    taskType: 'SEQ_CLS',
  },
  token: {
    icon: 'BIO',
    tabLabel: 'Token',
    title: 'Token-Level (Per Token)',
    detail: 'Each token → BIO Label → O O B-LOC I-LOC O',
    signals: ['PII Detection'],
    cls: styles.branchToken,
    tagCls: styles.signalTagToken,
    taskType: 'TOKEN_CLS',
  },
  pool: {
    icon: 'EMB',
    tabLabel: 'Embedding',
    title: 'Bi-Encoder',
    detail: 'mean-pooling(h₁..hₙ) → [0.23, -0.41, 0.87, ...]',
    signals: ['Semantic Cache', 'Similarity', 'Complexity-CL', 'Jailbreak-CL'],
    cls: styles.branchPool,
    tagCls: styles.signalTagPool,
    taskType: 'EMBEDDING',
  },
  cross: {
    icon: 'RER',
    tabLabel: 'Rerank',
    title: 'Cross-Encoder',
    detail: '[CLS] query [SEP] candidate [SEP] → score',
    signals: ['Rerank', 'Multi-Modal'],
    cls: styles.branchCross,
    tagCls: styles.signalTagCross,
    taskType: 'CROSS_LEARNING',
  },
}

const DownArrow: React.FC = () => (
  <div className={styles.downArrow}>
    <div className={styles.downArrowLine} />
  </div>
)

const stageVariants = {
  hidden: { opacity: 0, y: -16 },
  visible: { opacity: 1, y: 0 },
}

const TransformerPipelineAnimation: React.FC<TransformerPipelineAnimationProps> = ({
  mode: controlledMode,
  onModeChange,
}) => {
  const [internalMode, setInternalMode] = useState<BranchMode>('cls')
  const mode = controlledMode ?? internalMode
  const ref = React.useRef<HTMLDivElement>(null)
  const inView = useInView(ref, { once: true, margin: '-80px' })

  function setMode(nextMode: BranchMode): void {
    if (controlledMode === undefined) {
      setInternalMode(nextMode)
    }
    onModeChange?.(nextMode)
  }

  const activeBranch = BRANCH_DATA[mode]

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
      <div className={styles.tabRow} role="tablist" aria-label="Encoder pipeline modes">
        {(['cls', 'token', 'pool', 'cross'] as BranchMode[]).map(m => (
          <button
            key={m}
            className={`${styles.tab} ${mode === m ? styles.tabActive : ''}`}
            onClick={() => setMode(m)}
            type="button"
            role="tab"
            aria-selected={mode === m}
          >
            <span className={styles.tabCode}>{BRANCH_DATA[m].icon}</span>
            <span>{BRANCH_DATA[m].tabLabel}</span>
          </button>
        ))}
      </div>

      <motion.div
        className={styles.pipeline}
        variants={containerVariants}
        initial="hidden"
        animate={inView ? 'visible' : 'hidden'}
      >
        <motion.div className={styles.stage} variants={stageVariants} transition={{ duration: 0.5 }}>
          <span className={styles.stageLabel}>Input</span>
          <div className={styles.inputBox}>
            &quot;Is machine learning related to AI?&quot;
          </div>
        </motion.div>

        <DownArrow />

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
                <span className={styles.encoderSubIcon}>{sub.icon}</span>
                <span>{sub.label}</span>
              </motion.div>
            ))}
          </div>
        </motion.div>

        <DownArrow />

        <motion.div
          key={mode}
          className={styles.branchesArea}
          variants={stageVariants}
          transition={{ duration: 0.5 }}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <span className={styles.stageLabel}>Signals</span>
          <div className={`${styles.branch} ${styles.branchActive} ${activeBranch.cls}`}>
            <span className={styles.branchIcon}>{activeBranch.icon}</span>
            <div className={styles.branchContent}>
              <span className={styles.branchTitle}>{activeBranch.title}</span>
              <span className={styles.branchDetail}>{activeBranch.detail}</span>
              <span className={styles.branchTaskType}>
                {'TaskType: '}
                {activeBranch.taskType}
              </span>
              <div className={styles.branchSignals}>
                {activeBranch.signals.map(s => (
                  <span key={s} className={`${styles.signalTag} ${activeBranch.tagCls}`}>{s}</span>
                ))}
              </div>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </div>
  )
}

export default TransformerPipelineAnimation
