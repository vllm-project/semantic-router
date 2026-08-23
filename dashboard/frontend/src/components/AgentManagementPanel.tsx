import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { useAuth } from '../contexts/AuthContext'
import {
  canInvokeAgentTools,
  canManageAgent,
  canManageAgentTools,
  canReadAgent,
  canReadAgentTools,
} from '../utils/accessControl'
import { agentManagementApi } from '../utils/agentManagementApi'
import type {
  AgentProfile,
  AgentProfileInput,
  AgentSkill,
  AgentSkillInput,
  AgentToolDefinition,
  AgentToolSource,
  AgentToolSourceInput,
} from '../generated/managementApiContract'
import AgentCredentialDialog from './AgentCredentialDialog'
import AgentInlineError from './AgentInlineError'
import AgentManagementDialog from './AgentManagementDialog'
import AgentResourceEditor, {
  type AgentEditableResource,
  type AgentResourceInput,
} from './AgentResourceEditor'
import ConfirmDialog from './ConfirmDialog'
import ProductIcon from './ProductIcon'
import styles from './AgentManagementPanel.module.css'

export type AgentManagementTab = 'profiles' | 'skills' | 'tools' | 'connections'
type AgentResource = AgentProfile | AgentSkill | AgentToolDefinition | AgentToolSource

interface AgentManagementPanelProps {
  activeTab: AgentManagementTab
  onTabChange: (tab: AgentManagementTab) => void
}

interface ModalState {
  mode: 'create' | 'view' | 'edit'
  tab: AgentManagementTab
  resource?: AgentResource
  etag?: string
}

const TAB_COPY: Record<
  AgentManagementTab,
  { singular: string; title: string; description: string }
> = {
  profiles: { singular: 'profile', title: 'Profiles', description: 'Shape how the Agent works.' },
  skills: { singular: 'skill', title: 'Skills', description: 'Reusable ways of working.' },
  tools: { singular: 'tool', title: 'Tools', description: 'What your Agent can do.' },
  connections: {
    singular: 'connection',
    title: 'Connections',
    description: 'Connect trusted tools.',
  },
}

function resourceId(tab: AgentManagementTab, resource: AgentResource): string {
  return tab === 'tools'
    ? (resource as AgentToolDefinition).name
    : (resource as AgentEditableResource).id
}

function resourceName(resource: AgentResource): string {
  return resource.name
}

function formatDate(value: string | undefined): string {
  if (!value) return '—'
  const parsed = Date.parse(value)
  return Number.isFinite(parsed)
    ? new Intl.DateTimeFormat('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        timeZone: 'UTC',
      }).format(parsed)
    : '—'
}

function statusLabel(status: string | undefined): string {
  if (!status) return 'Available'
  return status.charAt(0).toUpperCase() + status.slice(1)
}

function prettyJSON(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return 'Unavailable'
  }
}

export default function AgentManagementPanel({
  activeTab,
  onTabChange,
}: AgentManagementPanelProps) {
  const { user } = useAuth()
  const readAgent = canReadAgent(user)
  const readTools = canReadAgentTools(user)
  const manageAgent = canManageAgent(user)
  const manageTools = canManageAgentTools(user)
  const invokeTools = canInvokeAgentTools(user)
  const [resources, setResources] = useState<AgentResource[]>([])
  const [cursor, setCursor] = useState<string | undefined>()
  const [hasMore, setHasMore] = useState(false)
  const [search, setSearch] = useState('')
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [modal, setModal] = useState<ModalState | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<AgentResource | null>(null)
  const [approvalTarget, setApprovalTarget] = useState<AgentToolSource | null>(null)
  const [credentialDialog, setCredentialDialog] = useState(false)
  const listGeneration = useRef(0)
  const listController = useRef<AbortController | null>(null)

  const visibleTabs = useMemo<AgentManagementTab[]>(
    () => [
      ...(readAgent ? ['profiles' as const, 'skills' as const] : []),
      ...(readTools ? ['tools' as const, 'connections' as const] : []),
    ],
    [readAgent, readTools],
  )

  useEffect(() => {
    if (!visibleTabs.includes(activeTab) && visibleTabs[0]) onTabChange(visibleTabs[0])
  }, [activeTab, onTabChange, visibleTabs])

  const list = useCallback(
    async (nextCursor?: string, signal?: AbortSignal) => {
      const query = search.trim() || undefined
      if (activeTab === 'profiles')
        return agentManagementApi.listProfiles(query, nextCursor, 50, signal)
      if (activeTab === 'skills')
        return agentManagementApi.listSkills(query, nextCursor, 50, signal)
      if (activeTab === 'tools') return agentManagementApi.listTools(query, nextCursor, 50, signal)
      return agentManagementApi.listToolSources(query, nextCursor, 50, signal)
    },
    [activeTab, search],
  )

  const refresh = useCallback(async () => {
    if (!visibleTabs.includes(activeTab)) return
    const generation = ++listGeneration.current
    listController.current?.abort()
    const controller = new AbortController()
    listController.current = controller
    setLoading(true)
    try {
      const page = await list(undefined, controller.signal)
      if (controller.signal.aborted || generation !== listGeneration.current) return
      setResources(page.data)
      setCursor(page.page.nextCursor)
      setHasMore(page.page.hasMore)
      setError(null)
    } catch (cause) {
      if (controller.signal.aborted || generation !== listGeneration.current) return
      setError(
        cause instanceof Error ? cause.message : `${TAB_COPY[activeTab].title} are unavailable.`,
      )
    } finally {
      if (!controller.signal.aborted && generation === listGeneration.current) setLoading(false)
      if (listController.current === controller) listController.current = null
    }
  }, [activeTab, list, visibleTabs])

  useEffect(() => {
    const timer = window.setTimeout(() => void refresh(), 220)
    return () => {
      window.clearTimeout(timer)
      listController.current?.abort()
      listGeneration.current += 1
    }
  }, [refresh])

  const loadMore = async () => {
    if (!cursor || !hasMore || loading) return
    const generation = ++listGeneration.current
    listController.current?.abort()
    const controller = new AbortController()
    listController.current = controller
    setLoading(true)
    try {
      const page = await list(cursor, controller.signal)
      if (controller.signal.aborted || generation !== listGeneration.current) return
      setResources((current) => {
        const byId = new Map(current.map((item) => [resourceId(activeTab, item), item]))
        page.data.forEach((item) => byId.set(resourceId(activeTab, item), item))
        return [...byId.values()]
      })
      setCursor(page.page.nextCursor)
      setHasMore(page.page.hasMore)
    } catch (cause) {
      if (controller.signal.aborted || generation !== listGeneration.current) return
      setError(cause instanceof Error ? cause.message : 'More results could not be loaded.')
    } finally {
      if (!controller.signal.aborted && generation === listGeneration.current) setLoading(false)
      if (listController.current === controller) listController.current = null
    }
  }

  const openResource = async (resource: AgentResource) => {
    if (activeTab === 'tools') {
      setModal({ mode: 'view', tab: activeTab, resource })
      return
    }
    setBusy(true)
    try {
      const id = resourceId(activeTab, resource)
      const detail =
        activeTab === 'profiles'
          ? await agentManagementApi.getProfile(id)
          : activeTab === 'skills'
            ? await agentManagementApi.getSkill(id)
            : await agentManagementApi.getToolSource(id)
      setModal({ mode: 'view', tab: activeTab, resource: detail.data, etag: detail.etag })
      setError(null)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Details are unavailable.')
    } finally {
      setBusy(false)
    }
  }

  const canManageTab =
    activeTab === 'profiles' || activeTab === 'skills' ? manageAgent : manageTools

  const saveResource = async (input: AgentResourceInput) => {
    if (!modal || modal.tab === 'tools') return
    if (modal.tab === 'profiles' || modal.tab === 'skills' ? !manageAgent : !manageTools) return
    if (modal.tab === 'skills' && (modal.resource as AgentSkill | undefined)?.builtin) return
    setBusy(true)
    setError(null)
    try {
      const existing = modal.resource as AgentEditableResource | undefined
      const profileInput = input as AgentProfileInput
      const { approvalPolicy: immutableApprovalPolicy, ...profilePatch } = profileInput
      const sourceInput = input as AgentToolSourceInput
      const { kind: immutableSourceKind, ...sourcePatch } = sourceInput
      void immutableApprovalPolicy
      void immutableSourceKind
      const detail =
        modal.tab === 'profiles'
          ? existing
            ? await agentManagementApi.patchProfile(
                existing.id,
                { ...profilePatch, description: profileInput.description ?? '' },
                modal.etag ?? '',
              )
            : await agentManagementApi.createProfile(profileInput)
          : modal.tab === 'skills'
            ? existing
              ? await agentManagementApi.patchSkill(
                  existing.id,
                  input as AgentSkillInput,
                  modal.etag ?? '',
                )
              : await agentManagementApi.createSkill(input as AgentSkillInput)
            : existing
              ? await agentManagementApi.patchToolSource(
                  existing.id,
                  {
                    ...sourcePatch,
                    description: sourceInput.description ?? '',
                    ...((existing as AgentToolSource).credentialId && !sourceInput.credentialId
                      ? { credentialId: null }
                      : {}),
                  },
                  modal.etag ?? '',
                )
              : await agentManagementApi.createToolSource(sourceInput)
      setModal({ mode: 'view', tab: modal.tab, resource: detail.data, etag: detail.etag })
      await refresh()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Changes could not be saved.')
    } finally {
      setBusy(false)
    }
  }

  const deleteResource = async () => {
    if (!deleteTarget || !modal || modal.tab === 'tools') return
    if (modal.tab === 'profiles' || modal.tab === 'skills' ? !manageAgent : !manageTools) return
    if (modal.tab === 'skills' && (deleteTarget as AgentSkill).builtin) return
    setBusy(true)
    try {
      const id = resourceId(modal.tab, deleteTarget)
      if (modal.tab === 'profiles') await agentManagementApi.deleteProfile(id, modal.etag ?? '')
      else if (modal.tab === 'skills') await agentManagementApi.deleteSkill(id, modal.etag ?? '')
      else await agentManagementApi.deleteToolSource(id, modal.etag ?? '')
      setDeleteTarget(null)
      setModal(null)
      await refresh()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The resource could not be deleted.')
    } finally {
      setBusy(false)
    }
  }

  const testConnection = async (source: AgentToolSource) => {
    if (!invokeTools || source.status !== 'active') return
    setBusy(true)
    setError(null)
    try {
      const receipt = await agentManagementApi.testToolSource(source.id)
      await agentManagementApi.waitForOperation(receipt.operationId)
      const detail = await agentManagementApi.getToolSource(source.id)
      setModal({ mode: 'view', tab: 'connections', resource: detail.data, etag: detail.etag })
      setError(null)
      await refresh()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The connection test failed.')
    } finally {
      setBusy(false)
    }
  }

  const approveConnection = async () => {
    if (!manageTools || !approvalTarget?.discoveryDigest || modal?.tab !== 'connections') return
    setBusy(true)
    setError(null)
    try {
      const detail = await agentManagementApi.approveToolSource(
        approvalTarget.id,
        approvalTarget.discoveryDigest,
        modal.etag ?? '',
      )
      setModal({ mode: 'view', tab: 'connections', resource: detail.data, etag: detail.etag })
      setApprovalTarget(null)
      await refresh()
    } catch (cause) {
      setError(
        cause instanceof Error ? cause.message : 'The discovered tools could not be approved.',
      )
    } finally {
      setBusy(false)
    }
  }

  const toggleConnection = async (source: AgentToolSource) => {
    if (!manageTools || modal?.tab !== 'connections') return
    setBusy(true)
    setError(null)
    try {
      const detail = await agentManagementApi.patchToolSource(
        source.id,
        { status: source.status === 'active' ? 'disabled' : 'active' },
        modal.etag ?? '',
      )
      setModal({ mode: 'view', tab: 'connections', resource: detail.data, etag: detail.etag })
      await refresh()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The connection could not be updated.')
    } finally {
      setBusy(false)
    }
  }

  if (visibleTabs.length === 0) {
    return (
      <div className={styles.emptyState}>
        <ProductIcon name="shield" />
        <strong>Agent isn't available</strong>
        <span>Ask your workspace admin for access.</span>
      </div>
    )
  }

  const copy = TAB_COPY[activeTab]
  return (
    <>
      <section className={styles.panel}>
        <header className={styles.toolbar}>
          <div>
            <h2>{copy.title}</h2>
            <p>{copy.description}</p>
          </div>
          <div className={styles.toolbarActions}>
            <label className={styles.search}>
              <ProductIcon name="search" />
              <span className={styles.srOnly}>Search {copy.title}</span>
              <input
                type="search"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                placeholder={`Search ${copy.title.toLowerCase()}`}
              />
            </label>
            {activeTab === 'connections' && readTools ? (
              <button
                type="button"
                className={styles.secondaryButton}
                onClick={() => setCredentialDialog(true)}
                disabled={busy}
              >
                <ProductIcon name="key" />
                Credentials
              </button>
            ) : null}
            {canManageTab && activeTab !== 'tools' ? (
              <button
                type="button"
                className={styles.primaryButton}
                onClick={() => setModal({ mode: 'create', tab: activeTab })}
                disabled={busy || loading}
              >
                <ProductIcon name="plus" />
                New {copy.singular}
              </button>
            ) : null}
          </div>
        </header>

        {error && !modal ? (
          <p className={styles.inlineError} role="alert">
            <ProductIcon name="alert" />
            {error}
            <button type="button" onClick={() => void refresh()}>
              Retry
            </button>
          </p>
        ) : null}

        <div className={styles.tableWrap} aria-busy={loading || busy}>
          <table className={styles.table}>
            <thead>
              <TableHeader tab={activeTab} />
            </thead>
            <tbody>
              {resources.map((resource) => (
                <ResourceRow
                  key={resourceId(activeTab, resource)}
                  tab={activeTab}
                  resource={resource}
                  disabled={busy}
                  onOpen={() => void openResource(resource)}
                />
              ))}
            </tbody>
          </table>
          {loading && resources.length === 0 ? (
            <div className={styles.emptyState} role="status">
              <ProductIcon name="refresh" />
              <strong>Loading…</strong>
            </div>
          ) : null}
          {!loading && resources.length === 0 ? (
            <div className={styles.emptyState}>
              <ProductIcon
                name={
                  activeTab === 'profiles'
                    ? 'user'
                    : activeTab === 'skills'
                      ? 'puzzle'
                      : activeTab === 'tools'
                        ? 'tool'
                        : 'plug'
                }
              />
              <strong>No {copy.title.toLowerCase()}</strong>
              <span>
                {search
                  ? 'Try a different search.'
                  : `Your ${copy.title.toLowerCase()} will appear here.`}
              </span>
            </div>
          ) : null}
        </div>
        <footer className={styles.pagination}>
          <span>{resources.length} loaded</span>
          {hasMore ? (
            <button type="button" onClick={() => void loadMore()} disabled={loading}>
              {loading ? 'Loading…' : 'Load more'}
            </button>
          ) : (
            <span>Up to date</span>
          )}
        </footer>
      </section>

      {modal ? (
        <AgentManagementDialog
          eyebrow={
            modal.mode === 'create'
              ? `New ${TAB_COPY[modal.tab].singular}`
              : TAB_COPY[modal.tab].title
          }
          title={
            modal.resource ? resourceName(modal.resource) : `Create ${TAB_COPY[modal.tab].singular}`
          }
          description={
            modal.mode === 'view'
              ? undefined
              : 'Start with the essentials. Fine-tune only when needed.'
          }
          busy={busy}
          onClose={() => {
            setModal(null)
            setError(null)
          }}
        >
          {modal.mode === 'view' && modal.resource ? (
            <ResourceView
              tab={modal.tab}
              resource={modal.resource}
              canManage={
                modal.tab === 'profiles'
                  ? manageAgent
                  : modal.tab === 'skills'
                    ? manageAgent && !(modal.resource as AgentSkill).builtin
                    : manageTools
              }
              busy={busy}
              error={error}
              onEdit={() => {
                setError(null)
                setModal((current) => (current ? { ...current, mode: 'edit' } : current))
              }}
              onDelete={() => {
                setError(null)
                setDeleteTarget(modal.resource!)
              }}
              onTest={
                modal.tab === 'connections' &&
                invokeTools &&
                (modal.resource as AgentToolSource).status === 'active'
                  ? () => void testConnection(modal.resource as AgentToolSource)
                  : undefined
              }
              onApprove={
                modal.tab === 'connections'
                  ? () => {
                      setError(null)
                      setApprovalTarget(modal.resource as AgentToolSource)
                    }
                  : undefined
              }
              onToggle={
                modal.tab === 'connections'
                  ? () => void toggleConnection(modal.resource as AgentToolSource)
                  : undefined
              }
            />
          ) : modal.tab !== 'tools' ? (
            <AgentResourceEditor
              kind={
                modal.tab === 'profiles'
                  ? 'profile'
                  : modal.tab === 'skills'
                    ? 'skill'
                    : 'connection'
              }
              value={modal.resource as AgentEditableResource | undefined}
              busy={busy}
              error={error}
              onCancel={() =>
                modal.resource ? setModal({ ...modal, mode: 'view' }) : setModal(null)
              }
              onSave={(input) => void saveResource(input)}
            />
          ) : null}
        </AgentManagementDialog>
      ) : null}

      {credentialDialog ? (
        <AgentCredentialDialog canManage={manageTools} onClose={() => setCredentialDialog(false)} />
      ) : null}
      <ConfirmDialog
        isOpen={Boolean(deleteTarget)}
        title={`Delete ${deleteTarget ? resourceName(deleteTarget) : 'resource'}?`}
        description="Remove references to this item before deleting it."
        error={deleteTarget ? error : null}
        confirmLabel="Delete"
        pending={busy}
        onCancel={() => {
          setDeleteTarget(null)
          setError(null)
        }}
        onConfirm={deleteResource}
      />
      <ConfirmDialog
        isOpen={Boolean(approvalTarget)}
        title="Approve discovered tools?"
        description={
          approvalTarget?.availability === 'drifted'
            ? 'The remote tool set changed. Review the list before approving it.'
            : 'Review the discovered tools before making them available.'
        }
        error={approvalTarget ? error : null}
        confirmLabel="Approve tools"
        eyebrow="Connection review"
        pending={busy}
        tone="neutral"
        onCancel={() => {
          setApprovalTarget(null)
          setError(null)
        }}
        onConfirm={approveConnection}
      />
    </>
  )
}

function TableHeader({ tab }: { tab: AgentManagementTab }) {
  if (tab === 'profiles')
    return (
      <tr>
        <th>Name</th>
        <th>Modes</th>
        <th>Skills</th>
        <th>Status</th>
        <th>Updated</th>
      </tr>
    )
  if (tab === 'skills')
    return (
      <tr>
        <th>Name</th>
        <th>Type</th>
        <th>Tools</th>
        <th>Capabilities</th>
        <th>Updated</th>
      </tr>
    )
  if (tab === 'tools')
    return (
      <tr>
        <th>Name</th>
        <th>Action</th>
        <th>Permissions</th>
        <th>Timeout</th>
        <th>Availability</th>
      </tr>
    )
  return (
    <tr>
      <th>Name</th>
      <th>Transport</th>
      <th>Tools</th>
      <th>Status</th>
      <th>Updated</th>
    </tr>
  )
}

function ResourceRow({
  tab,
  resource,
  disabled,
  onOpen,
}: {
  tab: AgentManagementTab
  resource: AgentResource
  disabled: boolean
  onOpen: () => void
}) {
  const profile = resource as AgentProfile
  const skill = resource as AgentSkill
  const tool = resource as AgentToolDefinition
  const connection = resource as AgentToolSource
  return (
    <tr onClick={disabled ? undefined : onOpen}>
      <td>
        <button
          type="button"
          className={styles.rowOpen}
          disabled={disabled}
          onClick={(event) => {
            event.stopPropagation()
            onOpen()
          }}
          aria-label={`Open ${resourceName(resource)}`}
        >
          <span className={styles.rowIdentity}>
            <span className={styles.rowIcon}>
              <ProductIcon
                name={
                  tab === 'profiles'
                    ? 'user'
                    : tab === 'skills'
                      ? 'puzzle'
                      : tab === 'tools'
                        ? 'tool'
                        : 'plug'
                }
              />
            </span>
            <span>
              <strong>{resourceName(resource)}</strong>
              <small>
                {'description' in resource && resource.description
                  ? resource.description
                  : tab === 'tools'
                    ? tool.description
                    : '—'}
              </small>
            </span>
          </span>
        </button>
      </td>
      {tab === 'profiles' ? (
        <>
          <td>
            {profile.supportedModes.map((mode) => (
              <span key={mode} className={styles.chip}>
                {mode === 'chat' ? 'Chat' : 'Builder'}
              </span>
            ))}
          </td>
          <td>{profile.skills.length}</td>
          <td>
            <span className={styles.status}>{statusLabel(profile.status)}</span>
          </td>
          <td>{formatDate(profile.updatedAt)}</td>
        </>
      ) : null}
      {tab === 'skills' ? (
        <>
          <td>{skill.builtin ? 'Built in' : 'Custom'}</td>
          <td>{skill.requiredTools.length}</td>
          <td>{skill.minimumCapabilities.length || '—'}</td>
          <td>{formatDate(skill.updatedAt)}</td>
        </>
      ) : null}
      {tab === 'tools' ? (
        <>
          <td>
            <span className={styles.chip}>{tool.class}</span>
          </td>
          <td>{tool.requiredPermissions.length || 'None'}</td>
          <td>{Math.round(tool.timeoutMilliseconds / 1000)}s</td>
          <td>Ready</td>
        </>
      ) : null}
      {tab === 'connections' ? (
        <>
          <td>Streamable HTTP</td>
          <td>{connection.discoveredTools.length}</td>
          <td>
            <span className={styles.status}>
              {statusLabel(connection.availability.replace(/_/g, ' '))}
            </span>
          </td>
          <td>{formatDate(connection.updatedAt)}</td>
        </>
      ) : null}
    </tr>
  )
}

interface ResourceViewProps {
  tab: AgentManagementTab
  resource: AgentResource
  canManage: boolean
  busy: boolean
  error?: string | null
  onEdit: () => void
  onDelete: () => void
  onTest?: () => void
  onApprove?: () => void
  onToggle?: () => void
}

function ResourceView({
  tab,
  resource,
  canManage,
  busy,
  error,
  onEdit,
  onDelete,
  onTest,
  onApprove,
  onToggle,
}: ResourceViewProps) {
  const profile = resource as AgentProfile
  const skill = resource as AgentSkill
  const tool = resource as AgentToolDefinition
  const connection = resource as AgentToolSource
  return (
    <div className={styles.resourceView}>
      {error ? <AgentInlineError message={error} /> : null}
      {tab === 'profiles' ? (
        <div className={styles.detailGrid}>
          <Detail label="Modes" value={profile.supportedModes.join(', ')} />
          <Detail label="Default for" value={profile.defaultForModes.join(', ') || 'None'} />
          <Detail label="Skills" value={String(profile.skills.length)} />
          <Detail label="Tools" value={String(profile.toolPolicy.allow.length)} />
          <Detail label="Publishing" value="Review required" />
          <Detail label="Turn timeout" value={`${profile.maximumTurnSeconds}s`} />
          <Detail label="Context" value={`${profile.contextTokenBudget.toLocaleString()} tokens`} />
        </div>
      ) : null}
      {tab === 'skills' ? (
        <>
          <div className={styles.detailGrid}>
            <Detail label="Type" value={skill.builtin ? 'Built in' : 'Custom'} />
            <Detail label="Required tools" value={skill.requiredTools.join(', ') || 'None'} />
            <Detail label="Capabilities" value={skill.minimumCapabilities.join(', ') || 'None'} />
          </div>
          <section className={styles.instructions}>
            <h3>Instructions</h3>
            <pre>{skill.instructions || 'No instructions to show.'}</pre>
          </section>
        </>
      ) : null}
      {tab === 'tools' ? (
        <>
          <div className={styles.detailGrid}>
            <Detail label="Action" value={tool.class} />
            <Detail label="Idempotency" value={tool.idempotency} />
            <Detail label="Timeout" value={`${tool.timeoutMilliseconds} ms`} />
            <Detail label="Permissions" value={tool.requiredPermissions.join(', ') || 'None'} />
          </div>
          <section className={styles.instructions}>
            <h3>Input schema</h3>
            <pre>{prettyJSON(tool.inputSchema)}</pre>
          </section>
          <section className={styles.instructions}>
            <h3>Output schema</h3>
            <pre>{prettyJSON(tool.outputSchema)}</pre>
          </section>
        </>
      ) : null}
      {tab === 'connections' ? (
        <>
          <div className={styles.detailGrid}>
            <Detail label="Endpoint" value={connection.endpoint} />
            <Detail label="Transport" value="Streamable HTTP" />
            <Detail label="Credential" value={connection.credentialId ? 'Configured' : 'None'} />
            <Detail label="Tools" value={String(connection.discoveredTools.length)} />
            <Detail
              label="Availability"
              value={statusLabel(connection.availability.replace(/_/g, ' '))}
            />
            <Detail label="Last update" value={formatDate(connection.updatedAt)} />
          </div>
          {connection.discoveredTools.length ? (
            <section className={styles.instructions}>
              <h3>Discovered tools</h3>
              <pre>{connection.discoveredTools.map((item) => item.name).join('\n')}</pre>
            </section>
          ) : null}
          <section className={styles.instructions}>
            <h3>Network policy</h3>
            <pre>{prettyJSON(connection.egressPolicy)}</pre>
          </section>
        </>
      ) : null}
      {(canManage && tab !== 'tools') || onTest ? (
        <footer className={styles.viewActions}>
          {canManage && tab !== 'tools' ? (
            <button
              type="button"
              className={styles.dangerButton}
              onClick={onDelete}
              disabled={busy}
            >
              <ProductIcon name="trash" />
              Delete
            </button>
          ) : (
            <span />
          )}
          <div>
            {canManage && onToggle ? (
              <button
                type="button"
                className={styles.secondaryButton}
                onClick={onToggle}
                disabled={busy}
              >
                {connection.status === 'active' ? 'Disable' : 'Enable'}
              </button>
            ) : null}
            {onTest ? (
              <button
                type="button"
                className={styles.secondaryButton}
                onClick={onTest}
                disabled={busy}
              >
                <ProductIcon name="play" />
                {busy ? 'Testing…' : 'Test connection'}
              </button>
            ) : null}
            {canManage &&
            onApprove &&
            (connection.availability === 'pending_approval' ||
              connection.availability === 'drifted') ? (
              <button
                type="button"
                className={styles.secondaryButton}
                onClick={onApprove}
                disabled={busy}
              >
                <ProductIcon name="check" />
                Approve tools
              </button>
            ) : null}
            {canManage && tab !== 'tools' ? (
              <button
                type="button"
                className={styles.primaryButton}
                onClick={onEdit}
                disabled={busy}
              >
                <ProductIcon name="edit" />
                Edit
              </button>
            ) : null}
          </div>
        </footer>
      ) : null}
    </div>
  )
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div className={styles.detail}>
      <span>{label}</span>
      <strong title={value}>{value}</strong>
    </div>
  )
}
