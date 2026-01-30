/**
 * Agent Playground Page - Entry point for the agentic workflow visualization
 */

import { useState } from 'react'
import styles from './AgentPlaygroundPage.module.css'
import { AgentPlayground, ComputerUsePlayground, E2BComputerUsePlayground } from '../components/agent'
import AnimatedBackground from '../components/AnimatedBackground'

type AgentMode = 'chat' | 'computer-use' | 'e2b-computer-use'

const AgentPlaygroundPage = () => {
  const [mode, setMode] = useState<AgentMode>('e2b-computer-use')

  return (
    <div className={styles.container}>
      <AnimatedBackground speed="slow" />
      <div className={styles.playgroundWrapper}>
        {/* Mode Selector */}
        <div className={styles.modeSelector}>
          <button
            className={`${styles.modeButton} ${mode === 'chat' ? styles.active : ''}`}
            onClick={() => setMode('chat')}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
            </svg>
            Chat Agent
          </button>
          <button
            className={`${styles.modeButton} ${mode === 'computer-use' ? styles.active : ''}`}
            onClick={() => setMode('computer-use')}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="2" y="3" width="20" height="14" rx="2" />
              <line x1="8" y1="21" x2="16" y2="21" />
              <line x1="12" y1="17" x2="12" y2="21" />
            </svg>
            Browser Agent
          </button>
          <button
            className={`${styles.modeButton} ${mode === 'e2b-computer-use' ? styles.active : ''}`}
            onClick={() => setMode('e2b-computer-use')}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="2" y="3" width="20" height="14" rx="2" />
              <path d="M9 17v4" />
              <path d="M15 17v4" />
              <circle cx="12" cy="10" r="3" />
            </svg>
            E2B Desktop Agent
          </button>
        </div>

        {/* Agent Component */}
        {mode === 'chat' ? (
          <AgentPlayground
            endpoint="/api/router/v1/chat/completions"
            defaultModel="MoM"
            defaultSystemPrompt="You are a helpful AI agent that can use tools to accomplish tasks. Think step by step and use available tools when needed."
          />
        ) : mode === 'computer-use' ? (
          <ComputerUsePlayground
            endpoint="/api/router/v1/chat/completions"
            model="MoM"
          />
        ) : (
          <E2BComputerUsePlayground defaultModel="envoy/auto" />
        )}
      </div>
    </div>
  )
}

export default AgentPlaygroundPage
