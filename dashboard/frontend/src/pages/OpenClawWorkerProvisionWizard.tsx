import React, { useDeferredValue, useEffect, useMemo, useRef, useState } from 'react'
import styles from './OpenClawPage.module.css'
import { CANONICAL_AUTO_MODEL } from '../utils/routerModelSelection'
import {
  createLatestOpenClawRequest,
  fetchOpenClawJSON,
  getOpenClawErrorMessage,
  type LatestOpenClawRequest,
} from '../utils/openClawRequestSupport'
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
import { OpenClawRequestNotice } from './OpenClawRequestNotice'
import { OpenClawWorkerDeployStep as DeployStep } from './OpenClawWorkerDeployStep'

export const WorkerProvisionWizard: React.FC<{
  teams: TeamProfile[]
  onProvisioned: () => void
  onSwitchToTeam: () => void
  onSwitchToStatus: () => void
  onCreated?: () => void
  onBusyChange?: (busy: boolean) => void
}> = ({ teams, onProvisioned, onSwitchToTeam, onSwitchToStatus, onCreated, onBusyChange }) => {
  const [currentStep, setCurrentStep] = useState(0)
  const [skills, setSkills] = useState<SkillTemplate[]>([])
  const [skillsLoading, setSkillsLoading] = useState(true)
  const [skillsError, setSkillsError] = useState('')
  const [routerDiscoveryError, setRouterDiscoveryError] = useState('')
  const [portDiscoveryError, setPortDiscoveryError] = useState('')
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
    modelName: CANONICAL_AUTO_MODEL,
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
  const routerRequestRef = useRef<LatestOpenClawRequest | null>(null)
  const skillsRequestRef = useRef<LatestOpenClawRequest | null>(null)
  const portRequestRef = useRef<LatestOpenClawRequest | null>(null)
  const provisionRequestRef = useRef<LatestOpenClawRequest | null>(null)
  if (!routerRequestRef.current) routerRequestRef.current = createLatestOpenClawRequest()
  if (!skillsRequestRef.current) skillsRequestRef.current = createLatestOpenClawRequest()
  if (!portRequestRef.current) portRequestRef.current = createLatestOpenClawRequest()
  if (!provisionRequestRef.current) provisionRequestRef.current = createLatestOpenClawRequest()

  const loadRouterConfig = async () => {
    await routerRequestRef.current?.run(
      (signal) => fetchOpenClawJSON<unknown>('/api/router/config/all', {}, signal),
      {
        onStart: () => setRouterDiscoveryError(''),
        onSuccess: (data) => {
          const discoveredModelBaseUrl = deriveModelBaseUrlFromRouterConfig(data)
          if (!discoveredModelBaseUrl) return
          setContainer((prev) => {
            if (prev.modelBaseUrl.trim() && prev.modelBaseUrl !== getInitialModelBaseUrl()) {
              return prev
            }
            return { ...prev, modelBaseUrl: discoveredModelBaseUrl }
          })
        },
        onError: (error) => {
          setRouterDiscoveryError(
            getOpenClawErrorMessage(error, 'Could not discover the Semantic Router endpoint.'),
          )
        },
      },
    )
  }

  const loadSkills = async () => {
    await skillsRequestRef.current?.run(
      (signal) => fetchOpenClawJSON<SkillTemplate[]>('/api/openclaw/skills', {}, signal),
      {
        onStart: () => {
          setSkillsLoading(true)
          setSkillsError('')
        },
        onSuccess: (data) => setSkills(Array.isArray(data) ? data : []),
        onError: (error) => {
          setSkillsError(getOpenClawErrorMessage(error, 'Could not load the skill catalog.'))
        },
        onFinish: () => setSkillsLoading(false),
      },
    )
  }

  const loadNextPort = async () => {
    await portRequestRef.current?.run(
      (signal) => fetchOpenClawJSON<{ port?: number }>('/api/openclaw/next-port', {}, signal),
      {
        onStart: () => setPortDiscoveryError(''),
        onSuccess: (data) => {
          if (!data.port) return
          setContainer((prev) =>
            prev.gatewayPort === 0 ? { ...prev, gatewayPort: data.port || 0 } : prev,
          )
        },
        onError: (error) => {
          setPortDiscoveryError(
            getOpenClawErrorMessage(error, 'Could not reserve the next gateway port.'),
          )
        },
      },
    )
  }

  useEffect(() => {
    void Promise.all([loadRouterConfig(), loadSkills(), loadNextPort()])
    return () => {
      routerRequestRef.current?.cancel()
      skillsRequestRef.current?.cancel()
      portRequestRef.current?.cancel()
      provisionRequestRef.current?.cancel()
    }
  }, [])

  useEffect(() => {
    onBusyChange?.(provisionLoading)
  }, [onBusyChange, provisionLoading])

  const selectedTeam = teams.find((team) => team.id === selectedTeamId) || null

  useEffect(() => {
    if (!selectedTeamId && teams.length > 0) {
      setSelectedTeamId(teams[0].id)
    }
  }, [teams, selectedTeamId])

  useEffect(() => {
    if (selectedTeamId && !teams.some((team) => team.id === selectedTeamId)) {
      setSelectedTeamId(teams[0]?.id || '')
    }
  }, [teams, selectedTeamId])

  const toggleSkill = (id: string) => {
    setSelectedSkills((prev) => (prev.includes(id) ? prev.filter((s) => s !== id) : [...prev, id]))
  }

  const handleProvision = async () => {
    if (!selectedTeamId || !selectedTeam) {
      setProvisionError('Team selection is required before provisioning.')
      return
    }
    await provisionRequestRef.current?.run(
      (signal) =>
        fetchOpenClawJSON<ProvisionResponse>(
          '/api/openclaw/workers',
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              teamId: selectedTeamId,
              identity,
              skills: selectedSkills,
              container,
            }),
          },
          signal,
        ),
      {
        onStart: () => {
          setProvisionLoading(true)
          setProvisionError('')
          setProvisionResult(null)
        },
        onSuccess: (data) => {
          setProvisionResult(data)
          onProvisioned()
        },
        onError: (error) => {
          setProvisionError(getOpenClawErrorMessage(error, 'Provisioning failed.'))
        },
        onFinish: () => setProvisionLoading(false),
      },
    )
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
              <div
                className={`${styles.stepConnector} ${idx <= currentStep ? styles.stepConnectorActive : ''}`}
              />
            )}
            <button
              type="button"
              className={`${styles.stepItem} ${idx === currentStep ? styles.stepActive : ''} ${idx < currentStep ? styles.stepCompleted : ''}`}
              onClick={() => goToStep(idx)}
              aria-current={idx === currentStep ? 'step' : undefined}
            >
              <div className={styles.stepCircle}>
                {idx < currentStep ? (
                  <svg
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="3"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
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

      {provisionError ? (
        <OpenClawRequestNotice
          title="Worker provisioning failed"
          message={provisionError}
          retryLabel="Retry provisioning"
          onRetry={() => void handleProvision()}
          onDismiss={() => setProvisionError('')}
        />
      ) : null}

      {routerDiscoveryError ? (
        <OpenClawRequestNotice
          title="Router endpoint discovery failed"
          message={routerDiscoveryError}
          tone="warning"
          retryLabel="Retry discovery"
          onRetry={() => void loadRouterConfig()}
        />
      ) : null}

      {portDiscoveryError ? (
        <OpenClawRequestNotice
          title="Gateway port discovery failed"
          message={portDiscoveryError}
          tone="warning"
          retryLabel="Retry port lookup"
          onRetry={() => void loadNextPort()}
        />
      ) : null}

      {teams.length === 0 && (
        <div className={`${styles.errorAlert} ${styles.warningAlert}`}>
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
      {currentStep === 1 && (
        <SkillsStep
          skills={skills}
          skillsLoading={skillsLoading}
          skillsError={skillsError}
          selectedSkills={selectedSkills}
          toggleSkill={toggleSkill}
          onRetry={() => void loadSkills()}
        />
      )}
      {currentStep === 2 && <ConfigStep container={container} setContainer={setContainer} />}
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
          onDone={onCreated}
        />
      )}

      <div className={styles.actions}>
        <div className={styles.actionsLeft}>
          {currentStep > 0 && (
            <button
              type="button"
              className={styles.btnSecondary}
              onClick={() => goToStep(currentStep - 1)}
            >
              Back
            </button>
          )}
        </div>
        <div className={styles.actionsRight}>
          {currentStep < 3 && (
            <button
              type="button"
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
    setIdentity((prev) => ({ ...prev, [field]: value }))

  return (
    <div className={styles.stepContent}>
      <h2 className={styles.stepTitle}>Step 1: Agent Identity</h2>
      <p className={styles.stepDescription}>
        Define who your OpenClaw agent is — its name, personality, principles, and boundaries. These
        files form the agent&apos;s core identity (SOUL.md, IDENTITY.md).
      </p>

      <div className={styles.sectionTitle}>Team Selection (Required)</div>
      <div className={styles.formRow}>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Target Team</label>
          <select
            className={styles.selectInput}
            value={selectedTeamId}
            onChange={(e) => setSelectedTeamId(e.target.value)}
          >
            <option value="">Select a team...</option>
            {teams.map((team) => (
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
          <input
            className={styles.textInput}
            value={identity.name}
            onChange={(e) => update('name', e.target.value)}
            placeholder="Atlas"
          />
        </div>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Emoji</label>
          <input
            className={styles.textInput}
            value={identity.emoji}
            onChange={(e) => update('emoji', e.target.value)}
            placeholder={'\u{1F531}'}
          />
        </div>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Vibe</label>
          <input
            className={styles.textInput}
            value={identity.vibe}
            onChange={(e) => update('vibe', e.target.value)}
            placeholder="Calm, precise, opinionated"
          />
        </div>
      </div>

      <div className={styles.formGroup}>
        <label className={styles.formLabel}>Role / Creature</label>
        <input
          className={styles.textInput}
          value={identity.role}
          onChange={(e) => update('role', e.target.value)}
          placeholder="AI operations engineer"
        />
        <div className={styles.formHint}>
          What kind of creature is your agent? SRE, architect, assistant, mentor...
        </div>
      </div>

      <div className={styles.formGroup}>
        <label className={styles.formLabel}>Core Principles (SOUL.md)</label>
        <textarea
          className={styles.textArea}
          value={identity.principles}
          onChange={(e) => update('principles', e.target.value)}
          rows={6}
          placeholder="Your agent's core truths and principles..."
        />
        <div className={styles.formHint}>
          Markdown supported. Define the agent&apos;s operating principles and values.
        </div>
      </div>

      <div className={styles.formGroup}>
        <label className={styles.formLabel}>Boundaries</label>
        <textarea
          className={styles.textArea}
          value={identity.boundaries}
          onChange={(e) => update('boundaries', e.target.value)}
          rows={4}
          placeholder="- Don't run destructive commands without approval..."
        />
        <div className={styles.formHint}>
          What should the agent never do? Safety guardrails and limits.
        </div>
      </div>
    </div>
  )
}

export const SkillsStep: React.FC<{
  skills: SkillTemplate[]
  skillsLoading: boolean
  skillsError: string
  selectedSkills: string[]
  toggleSkill: (id: string) => void
  onRetry: () => void
}> = ({ skills, skillsLoading, skillsError, selectedSkills, toggleSkill, onRetry }) => {
  const [search, setSearch] = useState('')
  const [category, setCategory] = useState('all')
  const deferredSearch = useDeferredValue(search)
  const categories = useMemo(
    () => [...new Set(skills.map((skill) => skill.category).filter(Boolean))].sort(),
    [skills],
  )
  const visibleSkills = useMemo(() => {
    const query = deferredSearch.trim().toLowerCase()
    return skills.filter((skill) => {
      if (category !== 'all' && skill.category !== category) return false
      if (!query) return true
      return [skill.id, skill.name, skill.description, skill.category].some((value) =>
        value.toLowerCase().includes(query),
      )
    })
  }, [category, deferredSearch, skills])

  return (
    <div className={styles.stepContent}>
      <h2 className={styles.stepTitle}>Step 2: Select Skills</h2>
      <p className={styles.stepDescription}>
        Skills give your agent specialized abilities. Selected workflows are installed as a
        structured skill set and discovered at startup.
      </p>

      {skillsError ? (
        <OpenClawRequestNotice
          title="Skill catalog is unavailable"
          message={skillsError}
          onRetry={onRetry}
        />
      ) : null}

      <div className={styles.skillCatalogControls}>
        <label>
          <span>Search skills</span>
          <input
            type="search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Name, category, or capability"
          />
        </label>
        <label>
          <span>Category</span>
          <select value={category} onChange={(event) => setCategory(event.target.value)}>
            <option value="all">All categories</option>
            {categories.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
      </div>

      {skillsLoading && skills.length === 0 ? (
        <div className={styles.loading} role="status">
          Loading skill catalog…
        </div>
      ) : visibleSkills.length === 0 ? (
        <div className={styles.emptyState}>
          <div className={styles.emptyStateText}>
            {skills.length === 0
              ? 'No skills are available.'
              : 'No skills match the current filters.'}
          </div>
        </div>
      ) : (
        <div className={styles.skillGrid} role="group" aria-label="Worker skills">
          {visibleSkills.map((skill) => {
            const selected = selectedSkills.includes(skill.id)
            return (
              <label
                key={skill.id}
                className={`${styles.skillCard} ${selected ? styles.skillCardSelected : ''}`}
              >
                <input
                  type="checkbox"
                  className={styles.visuallyHiddenControl}
                  checked={selected}
                  onChange={() => toggleSkill(skill.id)}
                />
                <span className={styles.skillCardHeader}>
                  <span className={styles.skillCardEmoji}>{skill.emoji}</span>
                  <span className={styles.skillCardName}>{skill.name}</span>
                  <span className={styles.skillCardCategory}>{skill.category}</span>
                </span>
                <span className={styles.skillCardDesc}>{skill.description}</span>
              </label>
            )
          })}
        </div>
      )}

      <div className={styles.skillSelectionCount} aria-live="polite">
        {selectedSkills.length} skill{selectedSkills.length === 1 ? '' : 's'} selected
      </div>
    </div>
  )
}

const ConfigStep: React.FC<{
  container: ContainerConfig
  setContainer: React.Dispatch<React.SetStateAction<ContainerConfig>>
}> = ({ container, setContainer }) => {
  const update = (field: keyof ContainerConfig, value: string | number | boolean) =>
    setContainer((prev) => ({ ...prev, [field]: value }))

  return (
    <div className={styles.stepContent}>
      <h2 className={styles.stepTitle}>Step 3: Container & Model Configuration</h2>
      <p className={styles.stepDescription}>
        Configure how OpenClaw connects to Semantic Router for model routing and memory, and set
        container parameters.
      </p>

      <div className={styles.sectionTitle}>Container</div>

      <div className={styles.formRow}>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Gateway Port</label>
          <input
            className={styles.numberInput}
            type="number"
            value={container.gatewayPort}
            onChange={(e) => update('gatewayPort', parseInt(e.target.value) || 0)}
          />
          <div className={styles.formHint}>Auto-assigned if 0 or conflicting</div>
        </div>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Base Image</label>
          <input
            className={styles.textInput}
            value={container.baseImage}
            onChange={(e) => update('baseImage', e.target.value)}
          />
        </div>
      </div>

      <div className={styles.formRow}>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Auth Token</label>
          <input
            className={styles.textInput}
            value={container.authToken}
            onChange={(e) => update('authToken', e.target.value)}
            placeholder="Auto-generated if empty"
          />
          <div className={styles.formHint}>Leave blank to auto-generate</div>
        </div>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Network Mode</label>
          <select
            className={styles.selectInput}
            value={container.networkMode}
            onChange={(e) => update('networkMode', e.target.value)}
          >
            <option value="bridge">bridge (recommended – backend picks best network)</option>
            <option value="host">host</option>
          </select>
          <div className={styles.formHint}>
            In containerized deployments the backend overrides "bridge" with the shared network for
            DNS resolution
          </div>
        </div>
      </div>

      <div className={styles.sectionTitle}>Model Provider (via Semantic Router)</div>

      <div className={styles.formRow}>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Model Base URL</label>
          <input
            className={styles.textInput}
            value={container.modelBaseUrl}
            onChange={(e) => update('modelBaseUrl', e.target.value)}
          />
          <div className={styles.formHint}>
            Auto-discovered from router listeners in current config; editable if needed
          </div>
        </div>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Model Name</label>
          <input
            className={styles.textInput}
            value={container.modelName}
            onChange={(e) => update('modelName', e.target.value)}
          />
          <div className={styles.formHint}>&quot;auto&quot; for SR confidence routing</div>
        </div>
      </div>

      <div className={styles.sectionTitle}>Memory Mode</div>

      <div className={styles.formRow}>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Memory Backend</label>
          <div className={styles.toggle}>
            <button
              type="button"
              className={`${styles.toggleOption} ${container.memoryBackend === 'remote' ? styles.toggleOptionSelected : ''}`}
              onClick={() => update('memoryBackend', 'remote')}
            >
              Remote Embeddings
            </button>
            <button
              type="button"
              className={`${styles.toggleOption} ${container.memoryBackend === 'local' ? styles.toggleOptionSelected : ''}`}
              onClick={() => update('memoryBackend', 'local')}
            >
              Built-in (Recommended)
            </button>
          </div>
        </div>
      </div>

      {container.memoryBackend === 'remote' && (
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Embedding Base URL (Optional)</label>
          <input
            className={styles.textInput}
            value={container.memoryBaseUrl}
            onChange={(e) => update('memoryBaseUrl', e.target.value)}
            placeholder="https://your-openai-compatible-endpoint/v1"
          />
          <div className={styles.formHint}>
            Used for agents.defaults.memorySearch.remote.baseUrl
          </div>
        </div>
      )}

      <div className={styles.sectionTitle}>Features</div>

      <div className={styles.formGroup}>
        <label className={styles.formLabel}>Browser (Playwright)</label>
        <div className={styles.toggle}>
          <button
            type="button"
            className={`${styles.toggleOption} ${container.browserEnabled ? styles.toggleOptionSelected : ''}`}
            onClick={() => update('browserEnabled', true)}
          >
            Enabled
          </button>
          <button
            type="button"
            className={`${styles.toggleOption} ${!container.browserEnabled ? styles.toggleOptionSelected : ''}`}
            onClick={() => update('browserEnabled', false)}
          >
            Disabled
          </button>
        </div>
        <div className={styles.formHint}>
          Enable headless browser for web browsing and CUA tasks
        </div>
      </div>
    </div>
  )
}
