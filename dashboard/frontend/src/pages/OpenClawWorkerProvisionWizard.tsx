import React, { useEffect, useState } from 'react'
import styles from './OpenClawPage.module.css'
import {
  deriveModelBaseUrlFromRouterConfig,
  getInitialModelBaseUrl,
  PROVISION_STEPS,
  type ContainerConfig,
  type IdentityConfig,
  type ProvisionResponse,
  type SkillTemplate,
  type TeamProfile,
} from './OpenClawPageSupport'

export const WorkerProvisionWizard: React.FC<{
  teams: TeamProfile[]
  onProvisioned: () => void
  onSwitchToTeam: () => void
  onSwitchToStatus: () => void
  onCreated?: () => void
}> = ({ teams, onProvisioned, onSwitchToTeam, onSwitchToStatus, onCreated }) => {
  const [currentStep, setCurrentStep] = useState(0)
  const [skills, setSkills] = useState<SkillTemplate[]>([])
  const [selectedSkills, setSelectedSkills] = useState<string[]>([])
  const [selectedTeamId, setSelectedTeamId] = useState('')
  const [identity, setIdentity] = useState<IdentityConfig>({
    name: '',
    emoji: '',
    role: '',
    vibe: '',
    principles: '',
    boundaries: '',
  })
  const [container, setContainer] = useState<ContainerConfig>({
    containerName: '',
    gatewayPort: 0,
    authToken: '',
    modelBaseUrl: getInitialModelBaseUrl(),
    modelName: 'auto',
    memoryBackend: 'local',
    memoryBaseUrl: '',
    vectorStore: 'openclaw-demo',
    browserEnabled: false,
    baseImage: 'ghcr.io/openclaw/openclaw:latest',
    networkMode: 'bridge',
  })
  const [provisionResult, setProvisionResult] = useState<ProvisionResponse | null>(null)
  const [provisionLoading, setProvisionLoading] = useState(false)
  const [provisionError, setProvisionError] = useState('')

  useEffect(() => {
    fetch('/api/router/config/all')
      .then(r => (r.ok ? r.json() : null))
      .then(data => {
        const discoveredModelBaseUrl = deriveModelBaseUrlFromRouterConfig(data)
        if (!discoveredModelBaseUrl) return
        setContainer(prev => {
          if (prev.modelBaseUrl.trim() && prev.modelBaseUrl !== getInitialModelBaseUrl()) {
            return prev
          }
          return { ...prev, modelBaseUrl: discoveredModelBaseUrl }
        })
      })
      .catch(() => {})

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

  const selectedTeam = teams.find(team => team.id === selectedTeamId) || null

  useEffect(() => {
    if (!selectedTeamId && teams.length > 0) {
      setSelectedTeamId(teams[0].id)
    }
  }, [teams, selectedTeamId])

  useEffect(() => {
    if (selectedTeamId && !teams.some(team => team.id === selectedTeamId)) {
      setSelectedTeamId(teams[0]?.id || '')
    }
  }, [teams, selectedTeamId])

  const toggleSkill = (id: string) => {
    setSelectedSkills(prev =>
      prev.includes(id) ? prev.filter(s => s !== id) : [...prev, id]
    )
  }

  const handleProvision = async () => {
    if (!selectedTeamId || !selectedTeam) {
      setProvisionError('Team selection is required before provisioning.')
      return
    }
    setProvisionLoading(true)
    setProvisionError('')
    setProvisionResult(null)
    try {
      const res = await fetch('/api/openclaw/workers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ teamId: selectedTeamId, identity, skills: selectedSkills, container }),
      })
      const data = await res.json()
      if (!res.ok) {
        setProvisionError(data.error || 'Provisioning failed')
      } else {
        setProvisionResult(data)
        onProvisioned()
        onCreated?.()
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

      {teams.length === 0 && (
        <div className={styles.errorAlert} style={{ background: 'rgba(234, 179, 8, 0.1)', borderColor: 'rgba(234, 179, 8, 0.35)', color: '#eab308' }}>
          <span>No team available. Create a team first, then come back to provision.</span>
          <button className={styles.btnSmall} onClick={onSwitchToTeam} type="button">
            Open Team Tab
          </button>
        </div>
      )}

      {currentStep === 0 && (
        <IdentityStep
          identity={identity}
          setIdentity={setIdentity}
          teams={teams}
          selectedTeamId={selectedTeamId}
          setSelectedTeamId={setSelectedTeamId}
          onSwitchToTeam={onSwitchToTeam}
        />
      )}
      {currentStep === 1 && <SkillsStep skills={skills} selectedSkills={selectedSkills} toggleSkill={toggleSkill} />}
      {currentStep === 2 && (
        <ConfigStep
          container={container}
          setContainer={setContainer}
        />
      )}
      {currentStep === 3 && (
        <DeployStep
          identity={identity}
          selectedSkills={selectedSkills}
          skills={skills}
          container={container}
          selectedTeam={selectedTeam}
          teamMissing={!selectedTeamId}
          onProvision={handleProvision}
          provisionLoading={provisionLoading}
          provisionResult={provisionResult}
          onSwitchToStatus={onSwitchToStatus}
        />
      )}

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
            <button
              className={styles.btnPrimary}
              onClick={() => goToStep(currentStep + 1)}
              disabled={currentStep === 0 && !selectedTeamId}
            >
              Next Step
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

const IdentityStep: React.FC<{
  identity: IdentityConfig
  setIdentity: React.Dispatch<React.SetStateAction<IdentityConfig>>
  teams: TeamProfile[]
  selectedTeamId: string
  setSelectedTeamId: React.Dispatch<React.SetStateAction<string>>
  onSwitchToTeam: () => void
}> = ({ identity, setIdentity, teams, selectedTeamId, setSelectedTeamId, onSwitchToTeam }) => {
  const update = (field: keyof IdentityConfig, value: string) =>
    setIdentity(prev => ({ ...prev, [field]: value }))

  return (
    <div className={styles.stepContent}>
      <h2 className={styles.stepTitle}>Step 1: Agent Identity</h2>
      <p className={styles.stepDescription}>
        Define who your OpenClaw agent is — its name, personality, principles, and boundaries.
        These files form the agent&apos;s core identity (SOUL.md, IDENTITY.md).
      </p>

      <div className={styles.sectionTitle}>Team Selection (Required)</div>
      <div className={styles.formRow}>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Target Team</label>
          <select
            className={styles.selectInput}
            value={selectedTeamId}
            onChange={e => setSelectedTeamId(e.target.value)}
          >
            <option value="">Select a team...</option>
            {teams.map(team => (
              <option key={team.id} value={team.id}>
                {team.name}
              </option>
            ))}
          </select>
          <div className={styles.formHint}>Every agent must belong to one team.</div>
        </div>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Need a new team?</label>
          <button className={styles.btnSecondary} onClick={onSwitchToTeam} type="button">
            Go to Team Tab
          </button>
        </div>
      </div>

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
        <div className={styles.formHint}>Markdown supported. Define the agent&apos;s operating principles and values.</div>
      </div>

      <div className={styles.formGroup}>
        <label className={styles.formLabel}>Boundaries</label>
        <textarea className={styles.textArea} value={identity.boundaries} onChange={e => update('boundaries', e.target.value)} rows={4} placeholder="- Don't run destructive commands without approval..." />
        <div className={styles.formHint}>What should the agent never do? Safety guardrails and limits.</div>
      </div>
    </div>
  )
}

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

const ConfigStep: React.FC<{
  container: ContainerConfig
  setContainer: React.Dispatch<React.SetStateAction<ContainerConfig>>
}> = ({ container, setContainer }) => {
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

      <div className={styles.formRow}>
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
            <option value="bridge">bridge (recommended – backend picks best network)</option>
            <option value="host">host</option>
          </select>
          <div className={styles.formHint}>In containerized deployments the backend overrides "bridge" with the shared network for DNS resolution</div>
        </div>
      </div>

      <div className={styles.sectionTitle}>Model Provider (via Semantic Router)</div>

      <div className={styles.formRow}>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Model Base URL</label>
          <input className={styles.textInput} value={container.modelBaseUrl} onChange={e => update('modelBaseUrl', e.target.value)} />
          <div className={styles.formHint}>Auto-discovered from router listeners in current config; editable if needed</div>
        </div>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Model Name</label>
          <input className={styles.textInput} value={container.modelName} onChange={e => update('modelName', e.target.value)} />
          <div className={styles.formHint}>&quot;auto&quot; for SR confidence routing</div>
        </div>
      </div>

      <div className={styles.sectionTitle}>Memory Mode</div>

      <div className={styles.formRow}>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Memory Backend</label>
          <div className={styles.toggle}>
            <button className={`${styles.toggleOption} ${container.memoryBackend === 'remote' ? styles.toggleOptionSelected : ''}`} onClick={() => update('memoryBackend', 'remote')}>
              Remote Embeddings
            </button>
            <button className={`${styles.toggleOption} ${container.memoryBackend === 'local' ? styles.toggleOptionSelected : ''}`} onClick={() => update('memoryBackend', 'local')}>
              Built-in (Recommended)
            </button>
          </div>
        </div>
      </div>

      {container.memoryBackend === 'remote' && (
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Embedding Base URL (Optional)</label>
          <input className={styles.textInput} value={container.memoryBaseUrl} onChange={e => update('memoryBaseUrl', e.target.value)} placeholder="https://your-openai-compatible-endpoint/v1" />
          <div className={styles.formHint}>Used for agents.defaults.memorySearch.remote.baseUrl</div>
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

const DeployStep: React.FC<{
  identity: IdentityConfig
  selectedSkills: string[]
  skills: SkillTemplate[]
  container: ContainerConfig
  selectedTeam: TeamProfile | null
  teamMissing: boolean
  onProvision: () => void
  provisionLoading: boolean
  provisionResult: ProvisionResponse | null
  onSwitchToStatus: () => void
}> = ({ identity, selectedSkills, skills, container, selectedTeam, teamMissing, onProvision, provisionLoading, provisionResult, onSwitchToStatus }) => {
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

      {teamMissing && (
        <div className={styles.errorAlert} style={{ background: 'rgba(239, 68, 68, 0.1)', borderColor: 'rgba(239, 68, 68, 0.35)', color: '#ef4444' }}>
          <span>Team selection is required before deployment.</span>
        </div>
      )}

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
          <div className={styles.summaryCardTitle}>Team</div>
          <div className={styles.summaryCardContent}>
            <strong>{selectedTeam?.name || '(not selected)'}</strong><br />
            {(selectedTeam?.role || 'No role set')}<br />
            <span style={{ color: 'var(--color-text-secondary)', fontSize: '0.8rem' }}>{selectedTeam?.vibe || 'No vibe set'}</span>
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
            <strong>Auto-generated by backend</strong> :{container.gatewayPort || 'auto'}<br />
            Image: {container.baseImage}<br />
            Network: {container.networkMode}
          </div>
        </div>
        <div className={styles.summaryCard}>
          <div className={styles.summaryCardTitle}>Model & Memory</div>
          <div className={styles.summaryCardContent}>
            Model: {container.modelName} via SR<br />
            Memory: {container.memoryBackend === 'remote' ? `Remote embeddings${container.memoryBaseUrl ? ` (${container.memoryBaseUrl})` : ''}` : 'Built-in'}<br />
            Browser: {container.browserEnabled ? 'Enabled' : 'Disabled'}
          </div>
        </div>
      </div>

      {!provisionResult && (
        <button className={styles.btnSuccess} onClick={onProvision} disabled={provisionLoading || teamMissing}>
          {provisionLoading ? 'Provisioning & starting container...' : 'Provision & Start'}
        </button>
      )}

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
            Go to Claw Dashboard
          </button>

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
