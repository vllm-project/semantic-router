import React, { useCallback, useEffect, useState } from 'react'
import styles from './OpenClawPage.module.css'

// --- Types ---

interface SkillTemplate {
  id: string
  name: string
  description: string
  emoji: string
  category: string
  builtin: boolean
}

interface IdentityConfig {
  name: string
  emoji: string
  role: string
  vibe: string
  principles: string
  boundaries: string
  userName: string
  userNotes: string
}

interface ContainerConfig {
  containerName: string
  gatewayPort: number
  authToken: string
  modelBaseUrl: string
  modelApiKey: string
  modelName: string
  memoryBackend: string
  memoryBaseUrl: string
  vectorStore: string
  browserEnabled: boolean
  baseImage: string
  networkMode: string
}

interface OpenClawStatus {
  running: boolean
  containerName: string
  gatewayUrl: string
  port: number
  healthy: boolean
  error: string
}

interface ProvisionResponse {
  success: boolean
  message: string
  workspaceDir: string
  configPath: string
  containerId: string
  dockerCmd: string
  composeYaml: string
}

// --- Provision Steps ---

const PROVISION_STEPS = [
  { key: 'identity', label: 'Identity' },
  { key: 'skills', label: 'Skills' },
  { key: 'config', label: 'Configuration' },
  { key: 'deploy', label: 'Deploy' },
]

// --- Component ---

const OpenClawPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'provision' | 'status'>('provision')
  const [containers, setContainers] = useState<OpenClawStatus[]>([])
  const [statusLoading, setStatusLoading] = useState(true)

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch('/api/openclaw/status')
      if (res.ok) {
        const data = await res.json()
        setContainers(Array.isArray(data) ? data : [])
      }
    } catch {
      // ignore
    } finally {
      setStatusLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchStatus()
    const interval = setInterval(fetchStatus, 15000)
    return () => clearInterval(interval)
  }, [fetchStatus])

  const runningCount = containers.filter(c => c.running).length

  return (
    <div className={styles.container}>
      {/* Header */}
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <h1 className={styles.title}>
            OpenClaw Agent
            {runningCount > 0 && (
              <span className={`${styles.titleBadge} ${styles.badgeRunning}`}>
                {runningCount} Running
              </span>
            )}
          </h1>
          <p className={styles.subtitle}>
            Provision, configure, and manage your OpenClaw AI agents. OpenClaw uses Semantic Router
            for intelligent model routing, memory, and knowledge management.
          </p>
        </div>
        <div className={styles.headerRight}>
          <button className={styles.btnSecondary} onClick={fetchStatus}>
            Refresh
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className={styles.tabs}>
        <button
          className={`${styles.tab} ${activeTab === 'provision' ? styles.tabActive : ''}`}
          onClick={() => setActiveTab('provision')}
        >
          <span className={styles.tabIcon}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2L2 7l10 5 10-5-10-5z" /><path d="M2 17l10 5 10-5" /><path d="M2 12l10 5 10-5" />
            </svg>
          </span>
          Provision
        </button>
        <button
          className={`${styles.tab} ${activeTab === 'status' ? styles.tabActive : ''}`}
          onClick={() => setActiveTab('status')}
        >
          <span className={styles.tabIcon}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
            </svg>
          </span>
          Status ({containers.length})
        </button>
      </div>

      {/* Tab Content */}
      {activeTab === 'provision' && (
        <ProvisionTab
          containers={containers}
          onProvisioned={fetchStatus}
          onSwitchToStatus={() => setActiveTab('status')}
        />
      )}
      {activeTab === 'status' && (
        <StatusTab
          containers={containers}
          statusLoading={statusLoading}
          onRefresh={fetchStatus}
        />
      )}
    </div>
  )
}

// =============================================================
//  Status Tab — Multi-container list + embedded Gateway UI
// =============================================================

const StatusTab: React.FC<{
  containers: OpenClawStatus[]
  statusLoading: boolean
  onRefresh: () => void
}> = ({ containers, statusLoading, onRefresh }) => {
  const [actionLoading, setActionLoading] = useState<string | null>(null)
  const [actionError, setActionError] = useState('')
  const [selectedContainer, setSelectedContainer] = useState<string | null>(null)
  const [gatewayToken, setGatewayToken] = useState('')

  const selected = containers.find(c => c.containerName === selectedContainer)

  useEffect(() => {
    if (selected?.healthy && selectedContainer) {
      setGatewayToken('')
      fetch(`/api/openclaw/token?name=${encodeURIComponent(selectedContainer)}`)
        .then(r => r.json())
        .then(d => { if (d.token) setGatewayToken(d.token) })
        .catch(() => {})
    }
  }, [selected?.healthy, selectedContainer])

  const handleAction = async (action: 'start' | 'stop', name: string) => {
    setActionLoading(name)
    setActionError('')
    try {
      const res = await fetch(`/api/openclaw/${action}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ containerName: name }),
      })
      if (!res.ok) {
        const data = await res.json()
        setActionError(data.error || `Failed to ${action}`)
      }
      setTimeout(onRefresh, 2000)
    } catch (e) {
      setActionError(String(e))
    } finally {
      setActionLoading(null)
    }
  }

  const handleDelete = async (name: string) => {
    if (!confirm(`Remove container "${name}"? This will stop and remove the Docker container.`)) return
    setActionLoading(name)
    setActionError('')
    try {
      const res = await fetch(`/api/openclaw/containers/${encodeURIComponent(name)}`, { method: 'DELETE' })
      if (!res.ok) {
        const data = await res.json()
        setActionError(data.error || 'Failed to remove')
      }
      if (selectedContainer === name) setSelectedContainer(null)
      setTimeout(onRefresh, 1000)
    } catch (e) {
      setActionError(String(e))
    } finally {
      setActionLoading(null)
    }
  }

  if (statusLoading) {
    return (
      <div className={styles.loading}>
        <div className={styles.spinner} />
        <p>Checking OpenClaw containers...</p>
      </div>
    )
  }

  // Gateway UI sub-view
  if (selectedContainer && selected?.healthy) {
    if (!gatewayToken) {
      return (
        <div className={styles.loading}>
          <div className={styles.spinner} />
          <p>Connecting to gateway...</p>
        </div>
      )
    }
    const wsProto = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const proxyBase = `/embedded/openclaw/${encodeURIComponent(selectedContainer)}/`
    const iframeSrc = `${proxyBase}#token=${encodeURIComponent(gatewayToken)}&gatewayUrl=${encodeURIComponent(`${wsProto}://${window.location.host}${proxyBase}`)}`
    return (
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
          <button className={styles.btnSecondary} onClick={() => setSelectedContainer(null)} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="15 18 9 12 15 6" />
            </svg>
            Back to Status
          </button>
          <span style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)' }}>
            {selected.containerName} &mdash; port {selected.port}
          </span>
        </div>
        <div className={styles.iframeContainer}>
          <iframe
            key={gatewayToken}
            className={styles.iframe}
            src={iframeSrc}
            title={`OpenClaw Control UI — ${selectedContainer}`}
            sandbox="allow-same-origin allow-scripts allow-forms allow-popups"
          />
        </div>
      </div>
    )
  }

  return (
    <div>
      {actionError && (
        <div className={styles.errorAlert}>
          <span>{actionError}</span>
          <button onClick={() => setActionError('')} style={{ background: 'none', border: 'none', color: '#ef4444', cursor: 'pointer', fontSize: '1rem' }}>
            &times;
          </button>
        </div>
      )}

      {containers.length === 0 ? (
        <div className={styles.emptyState}>
          <div className={styles.emptyStateIcon}>{'\u{1F433}'}</div>
          <div className={styles.emptyStateText}>
            No OpenClaw containers provisioned yet.<br />
            Use the <strong>Provision</strong> tab to create one.
          </div>
        </div>
      ) : (
        <table className={styles.containerTable}>
          <thead>
            <tr>
              <th>Name</th>
              <th>Port</th>
              <th>Health</th>
              <th>Error</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {containers.map(c => (
              <tr key={c.containerName}>
                <td className={styles.containerTableName}>{c.containerName}</td>
                <td className={styles.containerTablePort}>{c.port}</td>
                <td>
                  <span className={`${styles.healthBadge} ${
                    c.healthy ? styles.healthBadgeHealthy :
                    c.running ? styles.healthBadgeRunning :
                    styles.healthBadgeStopped
                  }`}>
                    {c.healthy ? 'Healthy' : c.running ? 'Starting' : 'Stopped'}
                  </span>
                </td>
                <td style={{ fontSize: '0.8rem', color: 'var(--color-text-secondary)', maxWidth: '240px' }}>
                  {c.error || '\u2014'}
                </td>
                <td>
                  <div className={styles.containerActions}>
                    {c.healthy && (
                      <button
                        className={`${styles.btnSmall} ${styles.btnSmallPrimary}`}
                        onClick={() => setSelectedContainer(c.containerName)}
                      >
                        Dashboard
                      </button>
                    )}
                    {c.running ? (
                      <button
                        className={styles.btnSmall}
                        onClick={() => handleAction('stop', c.containerName)}
                        disabled={actionLoading === c.containerName}
                      >
                        Stop
                      </button>
                    ) : (
                      <button
                        className={styles.btnSmall}
                        onClick={() => handleAction('start', c.containerName)}
                        disabled={actionLoading === c.containerName}
                      >
                        Start
                      </button>
                    )}
                    <button
                      className={`${styles.btnSmall} ${styles.btnSmallDanger}`}
                      onClick={() => handleDelete(c.containerName)}
                      disabled={actionLoading === c.containerName}
                    >
                      Remove
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <div style={{ display: 'flex', gap: '0.75rem' }}>
        <button className={styles.btnSecondary} onClick={onRefresh}>
          Refresh Status
        </button>
      </div>
    </div>
  )
}

// =============================================================
//  Provision Tab — 4-Step Wizard
// =============================================================

const ProvisionTab: React.FC<{
  containers: OpenClawStatus[]
  onProvisioned: () => void
  onSwitchToStatus: () => void
}> = ({ containers, onProvisioned, onSwitchToStatus }) => {
  const [currentStep, setCurrentStep] = useState(0)
  const [skills, setSkills] = useState<SkillTemplate[]>([])
  const [selectedSkills, setSelectedSkills] = useState<string[]>([])
  const [identity, setIdentity] = useState<IdentityConfig>({
    name: '',
    emoji: '',
    role: '',
    vibe: '',
    principles: '',
    boundaries: '',
    userName: '',
    userNotes: '',
  })
  const [container, setContainer] = useState<ContainerConfig>({
    containerName: '',
    gatewayPort: 0,
    authToken: '',
    modelBaseUrl: 'http://127.0.0.1:8801/v1',
    modelApiKey: '',
    modelName: 'auto',
    memoryBackend: 'remote',
    memoryBaseUrl: 'http://127.0.0.1:8080',
    vectorStore: 'openclaw-demo',
    browserEnabled: false,
    baseImage: 'openclaw:local',
    networkMode: 'host',
  })
  const [provisionResult, setProvisionResult] = useState<ProvisionResponse | null>(null)
  const [provisionLoading, setProvisionLoading] = useState(false)
  const [provisionError, setProvisionError] = useState('')

  // Fetch available skills and next port on mount
  useEffect(() => {
    fetch('/api/openclaw/skills')
      .then(r => r.json())
      .then(data => setSkills(data))
      .catch(() => {})
    fetch('/api/openclaw/next-port')
      .then(r => r.json())
      .then(d => {
        if (d.port) setContainer(prev => prev.gatewayPort === 0 ? { ...prev, gatewayPort: d.port } : prev)
      })
      .catch(() => {})
  }, [])

  const nameCollision = container.containerName !== '' && containers.some(c => c.containerName === container.containerName)

  const toggleSkill = (id: string) => {
    setSelectedSkills(prev =>
      prev.includes(id) ? prev.filter(s => s !== id) : [...prev, id]
    )
  }

  const handleProvision = async () => {
    setProvisionLoading(true)
    setProvisionError('')
    setProvisionResult(null)
    try {
      const res = await fetch('/api/openclaw/provision', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ identity, skills: selectedSkills, container }),
      })
      const data = await res.json()
      if (!res.ok) {
        setProvisionError(data.error || 'Provisioning failed')
      } else {
        setProvisionResult(data)
        onProvisioned()
      }
    } catch (e) {
      setProvisionError(String(e))
    } finally {
      setProvisionLoading(false)
    }
  }

  const goToStep = (step: number) => {
    if (step >= 0 && step <= 3) setCurrentStep(step)
  }

  return (
    <div>
      {/* Stepper */}
      <div className={styles.stepper}>
        {PROVISION_STEPS.map((step, idx) => (
          <React.Fragment key={step.key}>
            {idx > 0 && (
              <div className={`${styles.stepConnector} ${idx <= currentStep ? styles.stepConnectorActive : ''}`} />
            )}
            <button
              className={`${styles.stepItem} ${idx === currentStep ? styles.stepActive : ''} ${idx < currentStep ? styles.stepCompleted : ''}`}
              onClick={() => goToStep(idx)}
            >
              <div className={styles.stepCircle}>
                {idx < currentStep ? (
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                ) : (
                  idx + 1
                )}
              </div>
              <span className={styles.stepLabel}>{step.label}</span>
            </button>
          </React.Fragment>
        ))}
      </div>

      {provisionError && (
        <div className={styles.errorAlert}>
          <span>{provisionError}</span>
          <button onClick={() => setProvisionError('')} style={{ background: 'none', border: 'none', color: '#ef4444', cursor: 'pointer', fontSize: '1rem' }}>
            &times;
          </button>
        </div>
      )}

      {/* Step Content */}
      {currentStep === 0 && <IdentityStep identity={identity} setIdentity={setIdentity} />}
      {currentStep === 1 && <SkillsStep skills={skills} selectedSkills={selectedSkills} toggleSkill={toggleSkill} />}
      {currentStep === 2 && (
        <ConfigStep
          container={container}
          setContainer={setContainer}
          nameCollision={nameCollision}
        />
      )}
      {currentStep === 3 && (
        <DeployStep
          identity={identity}
          selectedSkills={selectedSkills}
          skills={skills}
          container={container}
          nameCollision={nameCollision}
          onProvision={handleProvision}
          provisionLoading={provisionLoading}
          provisionResult={provisionResult}
          onSwitchToStatus={onSwitchToStatus}
        />
      )}

      {/* Navigation */}
      <div className={styles.actions}>
        <div className={styles.actionsLeft}>
          {currentStep > 0 && (
            <button className={styles.btnSecondary} onClick={() => goToStep(currentStep - 1)}>
              Back
            </button>
          )}
        </div>
        <div className={styles.actionsRight}>
          {currentStep < 3 && (
            <button className={styles.btnPrimary} onClick={() => goToStep(currentStep + 1)}>
              Next Step
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

// =============================================================
//  Step 1: Identity
// =============================================================

const IdentityStep: React.FC<{
  identity: IdentityConfig
  setIdentity: React.Dispatch<React.SetStateAction<IdentityConfig>>
}> = ({ identity, setIdentity }) => {
  const update = (field: keyof IdentityConfig, value: string) =>
    setIdentity(prev => ({ ...prev, [field]: value }))

  return (
    <div className={styles.stepContent}>
      <h2 className={styles.stepTitle}>Step 1: Agent Identity</h2>
      <p className={styles.stepDescription}>
        Define who your OpenClaw agent is — its name, personality, principles, and boundaries.
        These files form the agent's core identity (SOUL.md, IDENTITY.md, USER.md).
      </p>

      <div className={styles.formRowThree}>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Agent Name</label>
          <input className={styles.textInput} value={identity.name} onChange={e => update('name', e.target.value)} placeholder="Atlas" />
        </div>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Emoji</label>
          <input className={styles.textInput} value={identity.emoji} onChange={e => update('emoji', e.target.value)} placeholder={'\u{1F531}'} />
        </div>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Vibe</label>
          <input className={styles.textInput} value={identity.vibe} onChange={e => update('vibe', e.target.value)} placeholder="Calm, precise, opinionated" />
        </div>
      </div>

      <div className={styles.formGroup}>
        <label className={styles.formLabel}>Role / Creature</label>
        <input className={styles.textInput} value={identity.role} onChange={e => update('role', e.target.value)} placeholder="AI operations engineer" />
        <div className={styles.formHint}>What kind of creature is your agent? SRE, architect, assistant, mentor...</div>
      </div>

      <div className={styles.formGroup}>
        <label className={styles.formLabel}>Core Principles (SOUL.md)</label>
        <textarea className={styles.textArea} value={identity.principles} onChange={e => update('principles', e.target.value)} rows={6} placeholder="Your agent's core truths and principles..." />
        <div className={styles.formHint}>Markdown supported. Define the agent's operating principles and values.</div>
      </div>

      <div className={styles.formGroup}>
        <label className={styles.formLabel}>Boundaries</label>
        <textarea className={styles.textArea} value={identity.boundaries} onChange={e => update('boundaries', e.target.value)} rows={4} placeholder="- Don't run destructive commands without approval..." />
        <div className={styles.formHint}>What should the agent never do? Safety guardrails and limits.</div>
      </div>

      <div className={styles.sectionTitle}>Your Team / User Context</div>

      <div className={styles.formRow}>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Team / User Name</label>
          <input className={styles.textInput} value={identity.userName} onChange={e => update('userName', e.target.value)} placeholder="The Engineering Team" />
        </div>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Context Notes</label>
          <input className={styles.textInput} value={identity.userNotes} onChange={e => update('userNotes', e.target.value)} placeholder="Platform engineering team..." />
        </div>
      </div>
    </div>
  )
}

// =============================================================
//  Step 2: Skills
// =============================================================

const SkillsStep: React.FC<{
  skills: SkillTemplate[]
  selectedSkills: string[]
  toggleSkill: (id: string) => void
}> = ({ skills, selectedSkills, toggleSkill }) => (
  <div className={styles.stepContent}>
    <h2 className={styles.stepTitle}>Step 2: Select Skills</h2>
    <p className={styles.stepDescription}>
      Skills give your agent specialized abilities. Each skill is a SKILL.md file that defines
      a structured workflow. Selected skills are auto-discovered at startup.
    </p>

    <div className={styles.skillGrid}>
      {skills.map(skill => (
        <div
          key={skill.id}
          className={`${styles.skillCard} ${selectedSkills.includes(skill.id) ? styles.skillCardSelected : ''}`}
          onClick={() => toggleSkill(skill.id)}
        >
          <div className={styles.skillCardHeader}>
            <span className={styles.skillCardEmoji}>{skill.emoji}</span>
            <span className={styles.skillCardName}>{skill.name}</span>
            <span className={styles.skillCardCategory}>{skill.category}</span>
          </div>
          <div className={styles.skillCardDesc}>{skill.description}</div>
        </div>
      ))}
    </div>

    <div style={{ marginTop: '1rem', fontSize: '0.8rem', color: 'var(--color-text-secondary)' }}>
      {selectedSkills.length} skill{selectedSkills.length !== 1 ? 's' : ''} selected
    </div>
  </div>
)

// =============================================================
//  Step 3: Configuration
// =============================================================

const ConfigStep: React.FC<{
  container: ContainerConfig
  setContainer: React.Dispatch<React.SetStateAction<ContainerConfig>>
  nameCollision: boolean
}> = ({ container, setContainer, nameCollision }) => {
  const update = (field: keyof ContainerConfig, value: string | number | boolean) =>
    setContainer(prev => ({ ...prev, [field]: value }))

  return (
    <div className={styles.stepContent}>
      <h2 className={styles.stepTitle}>Step 3: Container & Model Configuration</h2>
      <p className={styles.stepDescription}>
        Configure how OpenClaw connects to Semantic Router for model routing and memory,
        and set container parameters.
      </p>

      <div className={styles.sectionTitle}>Container</div>

      <div className={styles.formRowThree}>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Container Name</label>
          <input className={styles.textInput} value={container.containerName} onChange={e => update('containerName', e.target.value)} placeholder="my-agent" />
          {nameCollision && (
            <div className={styles.nameWarning}>
              A container with this name already exists and will be replaced.
            </div>
          )}
        </div>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Gateway Port</label>
          <input className={styles.numberInput} type="number" value={container.gatewayPort} onChange={e => update('gatewayPort', parseInt(e.target.value) || 0)} />
          <div className={styles.formHint}>Auto-assigned if 0 or conflicting</div>
        </div>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Base Image</label>
          <input className={styles.textInput} value={container.baseImage} onChange={e => update('baseImage', e.target.value)} />
        </div>
      </div>

      <div className={styles.formRow}>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Auth Token</label>
          <input className={styles.textInput} value={container.authToken} onChange={e => update('authToken', e.target.value)} placeholder="Auto-generated if empty" />
          <div className={styles.formHint}>Leave blank to auto-generate</div>
        </div>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Network Mode</label>
          <select className={styles.selectInput} value={container.networkMode} onChange={e => update('networkMode', e.target.value)}>
            <option value="host">host (recommended)</option>
            <option value="bridge">bridge</option>
          </select>
        </div>
      </div>

      <div className={styles.sectionTitle}>Model Provider (via Semantic Router)</div>

      <div className={styles.formRow}>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Model Base URL</label>
          <input className={styles.textInput} value={container.modelBaseUrl} onChange={e => update('modelBaseUrl', e.target.value)} />
          <div className={styles.formHint}>Envoy/SR endpoint for confidence-routed inference</div>
        </div>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Model Name</label>
          <input className={styles.textInput} value={container.modelName} onChange={e => update('modelName', e.target.value)} />
          <div className={styles.formHint}>&quot;auto&quot; for SR confidence routing</div>
        </div>
      </div>

      <div className={styles.formGroup}>
        <label className={styles.formLabel}>API Key</label>
        <input className={styles.textInput} type="password" value={container.modelApiKey} onChange={e => update('modelApiKey', e.target.value)} placeholder="vLLM API key" />
      </div>

      <div className={styles.sectionTitle}>Memory Backend (Semantic Router)</div>

      <div className={styles.formRow}>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Memory Backend</label>
          <div className={styles.toggle}>
            <button className={`${styles.toggleOption} ${container.memoryBackend === 'remote' ? styles.toggleOptionSelected : ''}`} onClick={() => update('memoryBackend', 'remote')}>
              Remote (SR API)
            </button>
            <button className={`${styles.toggleOption} ${container.memoryBackend === 'local' ? styles.toggleOptionSelected : ''}`} onClick={() => update('memoryBackend', 'local')}>
              Local (files)
            </button>
          </div>
        </div>
      </div>

      {container.memoryBackend === 'remote' && (
        <div className={styles.formRow}>
          <div className={styles.formGroup}>
            <label className={styles.formLabel}>Memory Base URL</label>
            <input className={styles.textInput} value={container.memoryBaseUrl} onChange={e => update('memoryBaseUrl', e.target.value)} />
          </div>
          <div className={styles.formGroup}>
            <label className={styles.formLabel}>Vector Store Name</label>
            <input className={styles.textInput} value={container.vectorStore} onChange={e => update('vectorStore', e.target.value)} />
          </div>
        </div>
      )}

      <div className={styles.sectionTitle}>Features</div>

      <div className={styles.formGroup}>
        <label className={styles.formLabel}>Browser (Playwright)</label>
        <div className={styles.toggle}>
          <button className={`${styles.toggleOption} ${container.browserEnabled ? styles.toggleOptionSelected : ''}`} onClick={() => update('browserEnabled', true)}>
            Enabled
          </button>
          <button className={`${styles.toggleOption} ${!container.browserEnabled ? styles.toggleOptionSelected : ''}`} onClick={() => update('browserEnabled', false)}>
            Disabled
          </button>
        </div>
        <div className={styles.formHint}>Enable headless browser for web browsing and CUA tasks</div>
      </div>
    </div>
  )
}

// =============================================================
//  Step 4: Deploy
// =============================================================

const DeployStep: React.FC<{
  identity: IdentityConfig
  selectedSkills: string[]
  skills: SkillTemplate[]
  container: ContainerConfig
  nameCollision: boolean
  onProvision: () => void
  provisionLoading: boolean
  provisionResult: ProvisionResponse | null
  onSwitchToStatus: () => void
}> = ({ identity, selectedSkills, skills, container, nameCollision, onProvision, provisionLoading, provisionResult, onSwitchToStatus }) => {
  const [copied, setCopied] = useState('')
  const [showCommands, setShowCommands] = useState(false)

  const copyToClipboard = (text: string, label: string) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(label)
      setTimeout(() => setCopied(''), 2000)
    })
  }

  const selectedSkillNames = skills.filter(s => selectedSkills.includes(s.id))

  return (
    <div className={styles.stepContent}>
      <h2 className={styles.stepTitle}>Step 4: Review & Deploy</h2>
      <p className={styles.stepDescription}>
        Review your configuration, then provision and start the OpenClaw container.
      </p>

      {nameCollision && (
        <div className={styles.errorAlert} style={{ background: 'rgba(234, 179, 8, 0.1)', borderColor: 'rgba(234, 179, 8, 0.3)', color: '#eab308' }}>
          <span>Container &quot;{container.containerName}&quot; already exists and will be replaced upon provisioning.</span>
        </div>
      )}

      {/* Summary */}
      <div className={styles.summaryGrid}>
        <div className={styles.summaryCard}>
          <div className={styles.summaryCardTitle}>Identity</div>
          <div className={styles.summaryCardContent}>
            <strong>{identity.emoji} {identity.name || '(unnamed)'}</strong><br />
            {identity.role || '(no role)'}<br />
            <span style={{ color: 'var(--color-text-secondary)', fontSize: '0.8rem' }}>{identity.vibe}</span>
          </div>
        </div>
        <div className={styles.summaryCard}>
          <div className={styles.summaryCardTitle}>Skills ({selectedSkills.length})</div>
          <div className={styles.summarySkillList}>
            {selectedSkillNames.map(s => (
              <span key={s.id} className={styles.summarySkillBadge}>{s.emoji} {s.name}</span>
            ))}
            {selectedSkills.length === 0 && <span style={{ color: 'var(--color-text-secondary)', fontSize: '0.8rem' }}>No skills selected</span>}
          </div>
        </div>
        <div className={styles.summaryCard}>
          <div className={styles.summaryCardTitle}>Container</div>
          <div className={styles.summaryCardContent}>
            <strong>{container.containerName || '(auto)'}</strong> :{container.gatewayPort || 'auto'}<br />
            Image: {container.baseImage}<br />
            Network: {container.networkMode}
          </div>
        </div>
        <div className={styles.summaryCard}>
          <div className={styles.summaryCardTitle}>Model & Memory</div>
          <div className={styles.summaryCardContent}>
            Model: {container.modelName} via SR<br />
            Memory: {container.memoryBackend === 'remote' ? `Remote (${container.memoryBaseUrl})` : 'Local files'}<br />
            Browser: {container.browserEnabled ? 'Enabled' : 'Disabled'}
          </div>
        </div>
      </div>

      {/* Provision & Start Button */}
      {!provisionResult && (
        <button className={styles.btnSuccess} onClick={onProvision} disabled={provisionLoading}>
          {provisionLoading ? 'Provisioning & starting container...' : 'Provision & Start'}
        </button>
      )}

      {/* Result */}
      {provisionResult?.success && (
        <div className={styles.successCard}>
          <div className={styles.successIcon}>
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" /><polyline points="16 8 10 16 7 13" />
            </svg>
          </div>
          <div className={styles.successTitle}>Container Started</div>
          <div className={styles.successMessage}>
            {provisionResult.message}
            {provisionResult.containerId && (
              <><br /><code style={{ fontSize: '0.75rem' }}>{provisionResult.containerId.slice(0, 12)}</code></>
            )}
          </div>

          <button className={styles.btnPrimary} onClick={onSwitchToStatus} style={{ marginBottom: '1rem' }}>
            Go to Status
          </button>

          {/* Collapsible reference commands */}
          <div style={{ textAlign: 'left' }}>
            <button
              onClick={() => setShowCommands(!showCommands)}
              style={{
                background: 'none', border: 'none', color: 'var(--color-text-secondary)',
                fontSize: '0.8rem', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem',
              }}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
                style={{ transform: showCommands ? 'rotate(90deg)' : 'rotate(0)', transition: 'transform 0.2s' }}>
                <polyline points="9 18 15 12 9 6" />
              </svg>
              Docker commands reference
            </button>

            {showCommands && (
              <>
                {provisionResult.dockerCmd && (
                  <div className={styles.codePreview}>
                    <div className={styles.codePreviewHeader}>
                      <span className={styles.codePreviewLabel}>Docker Run Command</span>
                      <button className={styles.codePreviewCopy} onClick={() => copyToClipboard(provisionResult.dockerCmd, 'docker')}>
                        {copied === 'docker' ? 'Copied!' : 'Copy'}
                      </button>
                    </div>
                    <pre className={styles.codePreviewContent}>{provisionResult.dockerCmd}</pre>
                  </div>
                )}

                {provisionResult.composeYaml && (
                  <div className={styles.codePreview}>
                    <div className={styles.codePreviewHeader}>
                      <span className={styles.codePreviewLabel}>Docker Compose YAML</span>
                      <button className={styles.codePreviewCopy} onClick={() => copyToClipboard(provisionResult.composeYaml, 'compose')}>
                        {copied === 'compose' ? 'Copied!' : 'Copy'}
                      </button>
                    </div>
                    <pre className={styles.codePreviewContent}>{provisionResult.composeYaml}</pre>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default OpenClawPage
