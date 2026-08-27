import type {
  AgentSkill,
  AgentToolDefinition,
  AgentToolSource,
} from '../generated/managementApiContract'
import AgentInlineError from './AgentInlineError'
import type { AgentManagementTab } from './AgentManagementPanel'
import { resourceName, type AgentResource } from './AgentManagementResourceSupport'
import ProductIcon from './ProductIcon'
import styles from './AgentManagementPanel.module.css'

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

function stringList(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === 'string')
    : []
}

export function AgentResourceTableHeader({ tab }: { tab: AgentManagementTab }) {
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

export function AgentResourceRow({
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
  const skill = resource as AgentSkill
  const tool = resource as AgentToolDefinition
  const connection = resource as AgentToolSource
  const skillTools = stringList(skill.requiredTools)
  const skillCapabilities = stringList(skill.minimumCapabilities)
  const toolPermissions = stringList(tool.requiredPermissions)
  const discoveredTools = Array.isArray(connection.discoveredTools)
    ? connection.discoveredTools
    : []
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
              <ProductIcon name={tab === 'skills' ? 'puzzle' : tab === 'tools' ? 'tool' : 'plug'} />
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
      {tab === 'skills' ? (
        <>
          <td>{skill.builtin ? 'Built in' : 'Custom'}</td>
          <td>{skillTools.length}</td>
          <td>{skillCapabilities.length || '—'}</td>
          <td>{formatDate(skill.updatedAt)}</td>
        </>
      ) : null}
      {tab === 'tools' ? (
        <>
          <td>
            <span className={styles.chip}>{tool.class}</span>
          </td>
          <td>{toolPermissions.length || 'None'}</td>
          <td>{Math.round((Number(tool.timeoutMilliseconds) || 0) / 1000)}s</td>
          <td>Ready</td>
        </>
      ) : null}
      {tab === 'connections' ? (
        <>
          <td>Streamable HTTP</td>
          <td>{discoveredTools.length}</td>
          <td>
            <span className={styles.status}>
              {statusLabel(connection.availability?.replace(/_/g, ' '))}
            </span>
          </td>
          <td>{formatDate(connection.updatedAt)}</td>
        </>
      ) : null}
    </tr>
  )
}

interface AgentResourceViewProps {
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

export function AgentResourceView({
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
}: AgentResourceViewProps) {
  const skill = resource as AgentSkill
  const tool = resource as AgentToolDefinition
  const connection = resource as AgentToolSource
  const skillTools = stringList(skill.requiredTools)
  const skillCapabilities = stringList(skill.minimumCapabilities)
  const toolPermissions = stringList(tool.requiredPermissions)
  const discoveredTools = Array.isArray(connection.discoveredTools)
    ? connection.discoveredTools
    : []
  return (
    <div className={styles.resourceView}>
      {error ? <AgentInlineError message={error} /> : null}
      {tab === 'skills' ? (
        <>
          <div className={styles.detailGrid}>
            <Detail label="Type" value={skill.builtin ? 'Built in' : 'Custom'} />
            <Detail label="Required tools" value={skillTools.join(', ') || 'None'} />
            <Detail label="Capabilities" value={skillCapabilities.join(', ') || 'None'} />
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
            <Detail label="Permissions" value={toolPermissions.join(', ') || 'None'} />
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
            <Detail label="Tools" value={String(discoveredTools.length)} />
            <Detail
              label="Availability"
              value={statusLabel(connection.availability?.replace(/_/g, ' '))}
            />
            <Detail label="Last update" value={formatDate(connection.updatedAt)} />
          </div>
          {discoveredTools.length ? (
            <section className={styles.instructions}>
              <h3>Discovered tools</h3>
              <pre>{discoveredTools.map((item) => item.name).join('\n')}</pre>
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
