import ProductIcon from '../components/ProductIcon'
import type { AccessControlViewProps as Props } from './AccessControlViewTypes'
import { Empty, ListToolbar, Pagination, Status } from './AccessControlViewPrimitives'
import { date, friendlyAction, number } from './AccessControlViewSupport'
import { rangeLabel } from './accessControlUsageRange'
import styles from './AccessControlPage.module.css'

function UsageFilters(props: Props) {
  const subjects =
    props.usageScope.type === 'user'
      ? props.users
      : props.usageScope.type === 'team'
        ? props.teams
        : props.usageScope.type === 'key'
          ? props.keys
          : []
  const models = [
    ...new Set([
      ...props.usage.byModel.map((item) => item.id),
      ...props.groups.flatMap((group) => group.resources.map((resource) => resource.resourceId)),
    ]),
  ]
  return (
    <div className={styles.filterRail}>
      <div className={styles.segmented}>
        {(['today', '7d', '30d'] as const).map((range) => (
          <button
            type="button"
            key={range}
            className={props.usageScope.range === range ? styles.segmentedActive : ''}
            onClick={() => props.onUsageScopeChange({ ...props.usageScope, range })}
          >
            {rangeLabel(range)}
          </button>
        ))}
      </div>
      <label>
        <span>Scope</span>
        <select
          value={props.usageScope.type}
          onChange={(event) =>
            props.onUsageScopeChange({
              ...props.usageScope,
              type: event.target.value as Props['usageScope']['type'],
              id: '',
            })
          }
        >
          <option value="global">All traffic</option>
          <option value="user">User</option>
          <option value="team">Team</option>
          <option value="key">API key</option>
        </select>
      </label>
      {props.usageScope.type !== 'global' ? (
        <label>
          <span>{props.usageScope.type === 'key' ? 'API key' : props.usageScope.type}</span>
          <select
            value={props.usageScope.id}
            onChange={(event) =>
              props.onUsageScopeChange({ ...props.usageScope, id: event.target.value })
            }
          >
            <option value="">All</option>
            {subjects.map((subject) => (
              <option value={subject.id} key={subject.id}>
                {'prefix' in subject ? `${subject.name} · ${subject.prefix}` : subject.name}
              </option>
            ))}
          </select>
        </label>
      ) : null}
      <label>
        <span>Model</span>
        <select
          value={props.usageScope.model}
          onChange={(event) =>
            props.onUsageScopeChange({ ...props.usageScope, model: event.target.value })
          }
        >
          <option value="">All models</option>
          {models.map((model) => (
            <option key={model} value={model}>
              {model}
            </option>
          ))}
        </select>
      </label>
      {props.loading ? <span className={styles.filterLoading}>Updating…</span> : null}
    </div>
  )
}

export function RequestLogsView(props: Props) {
  return (
    <div className={styles.viewStack}>
      <UsageFilters {...props} />
      <ListToolbar
        state={props.pageState}
        onChange={props.onPageStateChange}
        placeholder="Search request ID, model, or error"
      />
      <div className={styles.dataTable}>
        <div className={`${styles.dataRow} ${styles.logColumns} ${styles.dataHeader}`}>
          <span>Request</span>
          <span>User / team</span>
          <span>Tokens</span>
          <span>Latency</span>
          <span>Status</span>
          <span>Time</span>
          <span />
        </div>
        {props.requestPage.items.map((item) => (
          <div
            className={`${styles.dataRow} ${styles.logColumns} ${styles.dataRowInteractive}`}
            key={item.id}
            role="link"
            tabIndex={0}
            onClick={() => props.onOpenLog(item.id)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') props.onOpenLog(item.id)
            }}
          >
            <div className={styles.stackCell}>
              <strong>{item.model}</strong>
              <code title={item.requestId}>{item.requestId.slice(0, 16)}…</code>
            </div>
            <div className={styles.stackCell}>
              <span>
                {item.teamId
                  ? props.teams.find((team) => team.id === item.teamId)?.name || item.teamId
                  : props.users.find((user) => user.id === item.userId)?.name || item.userId}
              </span>
              <small>{props.keys.find((key) => key.id === item.keyId)?.name || item.keyId}</small>
            </div>
            <div className={styles.stackCell}>
              <span>{number(item.totalTokens)}</span>
              <small>
                {number(item.promptTokens)} in · {number(item.completionTokens)} out
              </small>
            </div>
            <div className={styles.stackCell}>
              <span>{number(item.latencyMs)} ms</span>
              <small>{item.ttftMs ? `${number(item.ttftMs)} ms TTFT` : 'TTFT unavailable'}</small>
            </div>
            <Status
              value={item.statusCode < 400 ? 'active' : 'disabled'}
              label={String(item.statusCode)}
            />
            <span>{date(item.createdAt)}</span>
            <span className={styles.rowChevron} aria-hidden="true">
              <ProductIcon name="chevron-right" />
            </span>
          </div>
        ))}
        {props.requestPage.items.length === 0 ? (
          <Empty
            title="No requests in this scope"
            detail="Adjust the filters or send a request with a managed API key."
          />
        ) : null}
      </div>
      <Pagination
        total={props.requestPage.total}
        state={props.pageState}
        onChange={props.onPageStateChange}
      />
    </div>
  )
}

export function AuditView(props: Props) {
  return (
    <div className={styles.viewStack}>
      <ListToolbar
        state={props.pageState}
        onChange={props.onPageStateChange}
        placeholder="Search actor, action, or resource"
      />
      <div className={styles.dataTable}>
        <div className={`${styles.dataRow} ${styles.auditColumns} ${styles.dataHeader}`}>
          <span>Event</span>
          <span>Actor</span>
          <span>Resource</span>
          <span>Time</span>
        </div>
        {props.auditPage.items.map((item) => (
          <div className={`${styles.dataRow} ${styles.auditColumns}`} key={item.id}>
            <div className={styles.stackCell}>
              <strong>{friendlyAction(item.action)}</strong>
              <code>{item.action}</code>
            </div>
            <span>{item.actorEmail || 'System'}</span>
            <div className={styles.stackCell}>
              <span>{item.resourceType}</span>
              <small>{item.resourceId || '—'}</small>
            </div>
            <span>{date(item.createdAt)}</span>
          </div>
        ))}
        {props.auditPage.items.length === 0 ? (
          <Empty title="No audit events found" detail="Administrative changes will appear here." />
        ) : null}
      </div>
      <Pagination
        total={props.auditPage.total}
        state={props.pageState}
        onChange={props.onPageStateChange}
      />
    </div>
  )
}
