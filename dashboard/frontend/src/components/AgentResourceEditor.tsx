import { useMemo, useState, type FormEvent } from 'react'

import { agentManagementApi } from '../utils/agentManagementApi'
import type {
  AgentProfile,
  AgentProfileInput,
  AgentSessionMode,
  AgentSkill,
  AgentSkillInput,
  AgentToolSource,
  AgentToolSourceInput,
} from '../generated/managementApiContract'
import AgentCapabilityPicker from './AgentCapabilityPicker'
import AgentInlineError from './AgentInlineError'
import AgentResourcePicker from './AgentResourcePicker'
import ProductIcon from './ProductIcon'
import styles from './AgentManagementPanel.module.css'

export type AgentEditableResourceKind = 'profile' | 'skill' | 'connection'
export type AgentEditableResource = AgentProfile | AgentSkill | AgentToolSource
export type AgentResourceInput = AgentProfileInput | AgentSkillInput | AgentToolSourceInput

interface AgentResourceEditorProps {
  kind: AgentEditableResourceKind
  value?: AgentEditableResource
  busy: boolean
  error?: string | null
  onCancel: () => void
  onSave: (input: AgentResourceInput) => void
}

function csv(value: string): string[] {
  return [
    ...new Set(
      value
        .split(',')
        .map((item) => item.trim())
        .filter(Boolean),
    ),
  ]
}

function toCSV(values: readonly string[] | undefined): string {
  return values?.join(', ') ?? ''
}

function CheckboxChoice({
  checked,
  description,
  label,
  onChange,
}: {
  checked: boolean
  description?: string
  label: string
  onChange: (checked: boolean) => void
}) {
  return (
    <label className={`${styles.choice} ${checked ? styles.choiceActive : ''}`}>
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
      />
      <span>
        <strong>{label}</strong>
        {description ? <small>{description}</small> : null}
      </span>
      <ProductIcon name={checked ? 'check' : 'plus'} aria-hidden="true" />
    </label>
  )
}

function ProfileEditor({
  value,
  busy,
  error,
  onCancel,
  onSave,
}: Omit<AgentResourceEditorProps, 'kind'> & { value?: AgentProfile }) {
  const [name, setName] = useState(value?.name ?? '')
  const [description, setDescription] = useState(value?.description ?? '')
  const [supportedModes, setSupportedModes] = useState<AgentSessionMode[]>(
    value?.supportedModes ?? ['chat', 'builder'],
  )
  const [defaultForModes, setDefaultForModes] = useState<AgentSessionMode[]>(
    value?.defaultForModes ?? [],
  )
  const [skillRefs, setSkillRefs] = useState(value?.skills ?? [])
  const [toolNames, setToolNames] = useState<string[]>(value?.toolPolicy.allow ?? [])
  const [skillsAvailable, setSkillsAvailable] = useState(true)
  const [toolsAvailable, setToolsAvailable] = useState(true)
  const [deniedTools, setDeniedTools] = useState<string[]>(value?.toolPolicy.deny ?? [])
  const [deniedToolsAvailable, setDeniedToolsAvailable] = useState(true)
  const [capabilities, setCapabilities] = useState(value?.minimumTargetCapabilities ?? [])
  const [capabilitiesAvailable, setCapabilitiesAvailable] = useState(true)
  const [maximumTurnSeconds, setMaximumTurnSeconds] = useState(value?.maximumTurnSeconds ?? 900)
  const [maximumToolSteps, setMaximumToolSteps] = useState(value?.maximumToolSteps ?? 24)
  const [contextTokenBudget, setContextTokenBudget] = useState(value?.contextTokenBudget ?? 32768)

  const toggleMode = (mode: AgentSessionMode, checked: boolean) => {
    setSupportedModes((current) =>
      checked ? [...new Set([...current, mode])] : current.filter((item) => item !== mode),
    )
    if (!checked) setDefaultForModes((current) => current.filter((item) => item !== mode))
  }

  const submit = (event: FormEvent) => {
    event.preventDefault()
    onSave({
      name: name.trim(),
      ...(description.trim() ? { description: description.trim() } : {}),
      supportedModes,
      defaultForModes: defaultForModes.filter((mode) => supportedModes.includes(mode)),
      minimumTargetCapabilities: capabilities,
      skills: skillRefs,
      toolPolicy: { allow: toolNames, ...(deniedTools.length ? { deny: deniedTools } : {}) },
      approvalPolicy: 'required',
      maximumTurnSeconds,
      maximumToolSteps,
      contextTokenBudget,
    })
  }

  return (
    <form onSubmit={submit} className={styles.editor}>
      <div className={styles.formGrid}>
        <label className={styles.field}>
          <span>
            Name <b>Required</b>
          </span>
          <input
            autoFocus
            required
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="Production Agent"
          />
        </label>
        <label className={`${styles.field} ${styles.fieldWide}`}>
          <span>
            Description <em>Optional</em>
          </span>
          <textarea
            value={description}
            onChange={(event) => setDescription(event.target.value)}
            placeholder="A short purpose for this profile"
            rows={2}
          />
        </label>
      </div>

      <fieldset className={styles.fieldset}>
        <legend>Available in</legend>
        <p>Choose where this profile can run. Mark one default for each mode.</p>
        <div className={styles.modeGrid}>
          {(['chat', 'builder'] as const).map((mode) => (
            <div key={mode} className={styles.modeChoice}>
              <CheckboxChoice
                checked={supportedModes.includes(mode)}
                label={mode === 'chat' ? 'Chat' : 'Builder'}
                description={mode === 'chat' ? 'Everyday model use' : 'Design and validate routing'}
                onChange={(checked) => toggleMode(mode, checked)}
              />
              <label className={styles.defaultToggle}>
                <input
                  type="checkbox"
                  disabled={!supportedModes.includes(mode)}
                  checked={defaultForModes.includes(mode)}
                  onChange={(event) =>
                    setDefaultForModes((current) =>
                      event.target.checked
                        ? [...new Set([...current, mode])]
                        : current.filter((item) => item !== mode),
                    )
                  }
                />
                Default
              </label>
            </div>
          ))}
        </div>
      </fieldset>

      <fieldset className={styles.fieldset}>
        <legend>Skills</legend>
        <p>Choose the playbooks this profile needs.</p>
        <AgentResourcePicker
          label="Skills"
          selectedIds={skillRefs.map((skill) => skill.id)}
          loadPage={(search, cursor, signal) =>
            agentManagementApi.listSkills(search, cursor, 50, signal)
          }
          resolveSelected={(id, signal) =>
            agentManagementApi.getSkill(id, signal).then((detail) => detail.data)
          }
          getId={(skill) => skill.id}
          getLabel={(skill) => skill.name}
          getDescription={(skill) => skill.description}
          onChange={(selected) =>
            setSkillRefs(selected.map((skill) => ({ id: skill.id, revision: skill.revision })))
          }
          onAvailabilityChange={setSkillsAvailable}
        />
      </fieldset>

      <fieldset className={styles.fieldset}>
        <legend>Tools</legend>
        <p>Choose what this profile can use.</p>
        <AgentResourcePicker
          label="Tools"
          selectedIds={toolNames}
          loadPage={(search, cursor, signal) =>
            agentManagementApi.listTools(search, cursor, 50, signal)
          }
          resolveSelected={agentManagementApi.findTool}
          getId={(tool) => tool.name}
          getLabel={(tool) => tool.name}
          getDescription={(tool) => tool.description}
          onChange={(selected) => {
            const next = selected.map((tool) => tool.name)
            setToolNames(next)
            setDeniedTools((current) => current.filter((tool) => !next.includes(tool)))
          }}
          onAvailabilityChange={setToolsAvailable}
        />
      </fieldset>

      <details className={styles.advanced}>
        <summary>
          Advanced settings <ProductIcon name="chevron-down" />
        </summary>
        <div className={styles.formGrid}>
          <div className={`${styles.field} ${styles.fieldWide}`}>
            <span>
              Target capabilities <em>Optional</em>
            </span>
            <AgentCapabilityPicker
              label="target capabilities"
              selected={capabilities}
              onChange={setCapabilities}
              onAvailabilityChange={setCapabilitiesAvailable}
            />
            <small>Only compatible models can be selected.</small>
          </div>
          <label className={styles.field}>
            <span>Turn timeout</span>
            <input
              type="number"
              min={30}
              max={3600}
              value={maximumTurnSeconds}
              onChange={(event) => setMaximumTurnSeconds(Number(event.target.value))}
            />
            <small>Seconds</small>
          </label>
          <label className={styles.field}>
            <span>Tool steps</span>
            <input
              type="number"
              min={1}
              max={100}
              value={maximumToolSteps}
              onChange={(event) => setMaximumToolSteps(Number(event.target.value))}
            />
          </label>
          <label className={styles.field}>
            <span>Context budget</span>
            <input
              type="number"
              min={1024}
              value={contextTokenBudget}
              onChange={(event) => setContextTokenBudget(Number(event.target.value))}
            />
            <small>Tokens</small>
          </label>
          <div className={`${styles.field} ${styles.fieldWide}`}>
            <span>
              Never allow <em>Optional</em>
            </span>
            <AgentResourcePicker
              label="Blocked tools"
              selectedIds={deniedTools}
              loadPage={(search, cursor, signal) =>
                agentManagementApi.listTools(search, cursor, 50, signal)
              }
              resolveSelected={agentManagementApi.findTool}
              getId={(tool) => tool.name}
              getLabel={(tool) => tool.name}
              getDescription={(tool) => tool.description}
              onChange={(selected) => {
                const next = selected.map((tool) => tool.name)
                setDeniedTools(next)
                setToolNames((current) => current.filter((tool) => !next.includes(tool)))
              }}
              onAvailabilityChange={setDeniedToolsAvailable}
            />
          </div>
        </div>
      </details>
      <EditorActions
        busy={busy}
        blocked={
          !skillsAvailable ||
          !toolsAvailable ||
          !deniedToolsAvailable ||
          !capabilitiesAvailable ||
          supportedModes.length === 0
        }
        error={error}
        onCancel={onCancel}
        label={value ? 'Save profile' : 'Create profile'}
      />
    </form>
  )
}

function SkillEditor({
  value,
  busy,
  error,
  onCancel,
  onSave,
}: Omit<AgentResourceEditorProps, 'kind'> & { value?: AgentSkill }) {
  const [name, setName] = useState(value?.name ?? '')
  const [description, setDescription] = useState(value?.description ?? '')
  const [instructions, setInstructions] = useState(value?.instructions ?? '')
  const [requiredTools, setRequiredTools] = useState<string[]>(value?.requiredTools ?? [])
  const [toolsAvailable, setToolsAvailable] = useState(true)
  const [capabilities, setCapabilities] = useState(value?.minimumCapabilities ?? [])
  const [capabilitiesAvailable, setCapabilitiesAvailable] = useState(true)

  return (
    <form
      className={styles.editor}
      onSubmit={(event) => {
        event.preventDefault()
        onSave({
          name: name.trim(),
          description: description.trim(),
          instructions: instructions.trim(),
          requiredTools,
          minimumCapabilities: capabilities,
        })
      }}
    >
      <div className={styles.formGrid}>
        <label className={styles.field}>
          <span>
            Name <b>Required</b>
          </span>
          <input
            autoFocus
            required
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="Routing designer"
          />
        </label>
        <label className={`${styles.field} ${styles.fieldWide}`}>
          <span>
            Description <em>Optional</em>
          </span>
          <input
            value={description}
            onChange={(event) => setDescription(event.target.value)}
            placeholder="What this skill helps the Agent do"
          />
        </label>
        <label className={`${styles.field} ${styles.fieldWide}`}>
          <span>
            Instructions <b>Required</b>
          </span>
          <textarea
            required
            rows={10}
            value={instructions}
            onChange={(event) => setInstructions(event.target.value)}
            placeholder="Describe how this skill should work…"
          />
        </label>
      </div>
      <fieldset className={styles.fieldset}>
        <legend>Required tools</legend>
        <p>Choose the tools this skill needs.</p>
        <AgentResourcePicker
          label="Tools"
          selectedIds={requiredTools}
          loadPage={(search, cursor, signal) =>
            agentManagementApi.listTools(search, cursor, 50, signal)
          }
          resolveSelected={agentManagementApi.findTool}
          getId={(tool) => tool.name}
          getLabel={(tool) => tool.name}
          getDescription={(tool) => tool.description}
          onChange={(selected) => setRequiredTools(selected.map((tool) => tool.name))}
          onAvailabilityChange={setToolsAvailable}
        />
      </fieldset>
      <details className={styles.advanced}>
        <summary>
          Advanced settings <ProductIcon name="chevron-down" />
        </summary>
        <div className={styles.field}>
          <span>
            Minimum capabilities <em>Optional</em>
          </span>
          <AgentCapabilityPicker
            label="minimum capabilities"
            selected={capabilities}
            onChange={setCapabilities}
            onAvailabilityChange={setCapabilitiesAvailable}
          />
        </div>
      </details>
      <EditorActions
        busy={busy}
        blocked={!toolsAvailable || !capabilitiesAvailable}
        error={error}
        onCancel={onCancel}
        label={value ? 'Save skill' : 'Create skill'}
      />
    </form>
  )
}

function ConnectionEditor({
  value,
  busy,
  error,
  onCancel,
  onSave,
}: Omit<AgentResourceEditorProps, 'kind'> & { value?: AgentToolSource }) {
  const [name, setName] = useState(value?.name ?? '')
  const [description, setDescription] = useState(value?.description ?? '')
  const [endpoint, setEndpoint] = useState(value?.endpoint ?? '')
  const [credentialId, setCredentialId] = useState(value?.credentialId ?? '')
  const [credentialsAvailable, setCredentialsAvailable] = useState(true)
  const [formError, setFormError] = useState<string | null>(null)
  const [allowedHosts, setAllowedHosts] = useState(toCSV(value?.egressPolicy.allowedHosts))
  const [allowedPorts, setAllowedPorts] = useState(
    toCSV(value?.egressPolicy.allowedPorts?.map(String)),
  )
  const [allowedPrivateCidrs, setAllowedPrivateCidrs] = useState(
    toCSV(value?.egressPolicy.allowedPrivateCidrs),
  )

  const parsedEndpoint = useMemo(() => {
    try {
      const parsed = new URL(endpoint.trim())
      if (
        parsed.protocol !== 'https:' ||
        !parsed.hostname ||
        parsed.username ||
        parsed.password ||
        parsed.search ||
        parsed.hash
      ) {
        return null
      }
      return parsed
    } catch {
      return null
    }
  }, [endpoint])
  const inferredHost = parsedEndpoint?.hostname.toLowerCase() ?? ''

  return (
    <form
      className={styles.editor}
      onSubmit={(event) => {
        event.preventDefault()
        if (!parsedEndpoint) {
          setFormError('Use an HTTPS URL without credentials, query parameters, or fragments.')
          return
        }
        const hosts = csv(allowedHosts || inferredHost).map((host) => host.toLowerCase())
        const ports = csv(allowedPorts).map(Number)
        if (ports.some((port) => !Number.isSafeInteger(port) || port < 1 || port > 65_535)) {
          setFormError('Use ports from 1 to 65535.')
          return
        }
        if (!hosts.includes(inferredHost)) {
          setFormError('Allowed hosts must include the endpoint host.')
          return
        }
        setFormError(null)
        onSave({
          name: name.trim(),
          ...(description.trim() ? { description: description.trim() } : {}),
          kind: 'remote',
          transport: 'streamable_http',
          endpoint: endpoint.trim(),
          ...(credentialId ? { credentialId } : {}),
          egressPolicy: {
            allowedHosts: hosts,
            ...(ports.length ? { allowedPorts: ports } : {}),
            ...(csv(allowedPrivateCidrs).length
              ? { allowedPrivateCidrs: csv(allowedPrivateCidrs) }
              : {}),
          },
        })
      }}
      onInput={() => setFormError(null)}
    >
      <div className={styles.formGrid}>
        <label className={styles.field}>
          <span>
            Name <b>Required</b>
          </span>
          <input
            autoFocus
            required
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="Knowledge tools"
          />
        </label>
        <label className={`${styles.field} ${styles.fieldWide}`}>
          <span>
            Description <em>Optional</em>
          </span>
          <input
            value={description}
            onChange={(event) => setDescription(event.target.value)}
            placeholder="What this connection provides"
          />
        </label>
        <label className={`${styles.field} ${styles.fieldWide}`}>
          <span>
            HTTPS endpoint <b>Required</b>
          </span>
          <input
            type="url"
            required
            pattern="https://.*"
            value={endpoint}
            onChange={(event) => setEndpoint(event.target.value)}
            placeholder="https://tools.example.com/connect"
          />
          <small>Use a direct HTTPS endpoint.</small>
        </label>
        <div className={`${styles.field} ${styles.fieldWide}`}>
          <span>
            Credential <em>Optional</em>
          </span>
          <AgentResourcePicker
            label="Credentials"
            selectedIds={credentialId ? [credentialId] : []}
            loadPage={(search, cursor, signal) =>
              agentManagementApi.listToolCredentials(search, cursor, 50, signal)
            }
            resolveSelected={(id, signal) =>
              agentManagementApi.getToolCredential(id, signal).then((detail) => detail.data)
            }
            getId={(credential) => credential.id}
            getLabel={(credential) => credential.name}
            getDescription={(credential) => (credential.status === 'active' ? 'Ready' : 'Disabled')}
            onChange={(selected) => setCredentialId(selected[selected.length - 1]?.id ?? '')}
            onAvailabilityChange={setCredentialsAvailable}
          />
          <small>Leave empty for a public endpoint.</small>
        </div>
      </div>
      <details className={styles.advanced}>
        <summary>
          Network policy <ProductIcon name="chevron-down" />
        </summary>
        <div className={styles.formGrid}>
          <label className={`${styles.field} ${styles.fieldWide}`}>
            <span>
              Allowed hosts <b>Required</b>
            </span>
            <input
              required
              value={allowedHosts || inferredHost}
              onChange={(event) => setAllowedHosts(event.target.value)}
              placeholder="tools.example.com"
            />
            <small>Comma-separated DNS names. The endpoint host must be included.</small>
          </label>
          <label className={styles.field}>
            <span>
              Allowed ports <em>Optional</em>
            </span>
            <input
              value={allowedPorts}
              onChange={(event) => setAllowedPorts(event.target.value)}
              placeholder="443"
            />
          </label>
          <label className={`${styles.field} ${styles.fieldWide}`}>
            <span>
              Private CIDRs <em>Optional</em>
            </span>
            <input
              value={allowedPrivateCidrs}
              onChange={(event) => setAllowedPrivateCidrs(event.target.value)}
              placeholder="10.24.0.0/16, 2001:db8:1::/64"
            />
            <small>Only these private networks may be reached.</small>
          </label>
        </div>
      </details>
      <EditorActions
        busy={busy}
        blocked={!credentialsAvailable}
        error={formError || error}
        onCancel={onCancel}
        label={value ? 'Save connection' : 'Create connection'}
      />
    </form>
  )
}

function EditorActions({
  busy,
  blocked = false,
  error,
  label,
  onCancel,
}: {
  busy: boolean
  blocked?: boolean
  error?: string | null
  label: string
  onCancel: () => void
}) {
  return (
    <footer className={styles.editorActions}>
      {error ? <AgentInlineError message={error} /> : <span />}
      <div>
        <button type="button" className={styles.secondaryButton} onClick={onCancel} disabled={busy}>
          Cancel
        </button>
        <button type="submit" className={styles.primaryButton} disabled={busy || blocked}>
          {busy ? 'Saving…' : label}
          <ProductIcon name="arrow-right" />
        </button>
      </div>
    </footer>
  )
}

export default function AgentResourceEditor(props: AgentResourceEditorProps) {
  if (props.kind === 'profile')
    return <ProfileEditor {...props} value={props.value as AgentProfile | undefined} />
  if (props.kind === 'skill')
    return <SkillEditor {...props} value={props.value as AgentSkill | undefined} />
  return <ConnectionEditor {...props} value={props.value as AgentToolSource | undefined} />
}
