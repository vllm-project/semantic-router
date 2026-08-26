import ProductIcon from '../components/ProductIcon'
import type { AccessControlViewProps as Props } from './AccessControlViewTypes'
import { Empty, EntityTable, ListToolbar, Pagination, Status } from './AccessControlViewPrimitives'
import { date } from './AccessControlViewSupport'
import { rateLimitPolicySummary } from './AccessControlDetailSupport'
import styles from './AccessControlPage.module.css'

export function KeysView(props: Props) {
  const filtered = props.keys
  return (
    <EntityTable
      toolbar={
        <ListToolbar
          state={props.pageState}
          onChange={props.onPageStateChange}
          placeholder="Search key name or ID"
        />
      }
      pagination={
        <Pagination
          total={props.entityTotals['api-keys']}
          state={props.pageState}
          onChange={props.onPageStateChange}
        />
      }
    >
      <div className={`${styles.dataRow} ${styles.keyColumns} ${styles.dataHeader}`}>
        <span>API key</span>
        <span>Owner</span>
        <span>Model visibility</span>
        <span>Quota</span>
        <span>Status</span>
        <span />
      </div>
      {filtered.map((key) => {
        return (
          <div
            className={`${styles.dataRow} ${styles.keyColumns} ${styles.dataRowInteractive}`}
            key={key.id}
            role="link"
            tabIndex={0}
            onClick={() => props.onOpenKey(key.id)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') props.onOpenKey(key.id)
            }}
          >
            <div className={styles.stackCell}>
              <strong>{key.name}</strong>
              <code>{key.prefix}••••••••</code>
            </div>
            <div className={styles.stackCell}>
              <span>{props.ownerName(key)}</span>
              <small>{key.ownerType === 'user' ? 'Personal' : 'Team'}</small>
            </div>
            <div className={styles.stackCell}>
              <span>View policy</span>
              <small>Resolved per key</small>
            </div>
            <div className={styles.stackCell}>
              <span>View quota</span>
              <small>Live capacity in details</small>
            </div>
            <div className={styles.stackCell}>
              <Status value={key.status} />
              <small>Used {date(key.lastUsedAt)}</small>
            </div>
            <span className={styles.rowChevron} aria-hidden="true">
              <ProductIcon name="chevron-right" />
            </span>
          </div>
        )
      })}
      {filtered.length === 0 ? (
        <Empty title="No API keys found" detail="Create a scoped key for a user or team." />
      ) : null}
    </EntityTable>
  )
}

export function GroupsView(props: Props) {
  const filtered = props.groups
  return (
    <EntityTable
      toolbar={
        <ListToolbar
          state={props.pageState}
          onChange={props.onPageStateChange}
          placeholder="Search access groups"
        />
      }
      pagination={
        <Pagination
          total={props.entityTotals['access-groups']}
          state={props.pageState}
          onChange={props.onPageStateChange}
        />
      }
    >
      <div className={`${styles.dataRow} ${styles.groupColumns} ${styles.dataHeader}`}>
        <span>Access group</span>
        <span>Models</span>
        <span>Assigned to</span>
        <span>Updated</span>
        <span />
      </div>
      {filtered.map((group) => (
        <div
          className={`${styles.dataRow} ${styles.groupColumns} ${styles.dataRowInteractive}`}
          key={group.id}
          role="link"
          tabIndex={0}
          onClick={() => props.onOpenEntity(group.id)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') props.onOpenEntity(group.id)
          }}
        >
          <div className={styles.stackCell}>
            <strong>{group.name}</strong>
            <span>{group.description || 'Reusable model grant'}</span>
          </div>
          <div className={styles.tagList}>
            {group.resources.map((resource) => (
              <code key={`${resource.resourceType}:${resource.resourceId}`}>
                {props.resourceName(resource.resourceType, resource.resourceId)}
              </code>
            ))}
          </div>
          <div className={styles.stackCell}>
            <span>{group.assignmentCount} assignments</span>
            <small>Assigned from users, Teams, or keys</small>
          </div>
          <span>{date(group.updatedAt)}</span>
          <span className={styles.rowChevron} aria-hidden="true">
            <ProductIcon name="chevron-right" />
          </span>
        </div>
      ))}
      {filtered.length === 0 ? (
        <Empty
          title="No access groups found"
          detail="Choose models once, then assign the group to users, Teams, or keys."
        />
      ) : null}
    </EntityTable>
  )
}

export function BudgetsView(props: Props) {
  const filtered = props.budgets
  return (
    <EntityTable
      toolbar={
        <ListToolbar
          state={props.pageState}
          onChange={props.onPageStateChange}
          placeholder="Search budgets"
        />
      }
      pagination={
        <Pagination
          total={props.entityTotals.budgets}
          state={props.pageState}
          onChange={props.onPageStateChange}
        />
      }
    >
      <div className={`${styles.dataRow} ${styles.budgetColumns} ${styles.dataHeader}`}>
        <span>Budget</span>
        <span>Used by</span>
        <span>Limits</span>
        <span>Mode</span>
        <span>Updated</span>
        <span />
      </div>
      {filtered.map((budget) => (
        <div
          className={`${styles.dataRow} ${styles.budgetColumns} ${styles.dataRowInteractive}`}
          key={budget.id}
          role="link"
          tabIndex={0}
          onClick={() => props.onOpenEntity(budget.id)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') props.onOpenEntity(budget.id)
          }}
        >
          <div className={styles.stackCell}>
            <strong>{budget.name}</strong>
            <Status
              value={budget.enabled ? 'active' : 'disabled'}
              label={budget.enabled ? 'Enforced' : 'Paused'}
            />
          </div>
          <div className={styles.stackCell}>
            <span className={styles.scopeBadge}>{budget.assignmentCount} assignments</span>
            <small>{budget.description || 'Reusable quota'}</small>
          </div>
          <div className={styles.stackCell}>
            <span>{rateLimitPolicySummary(budget.rules, 3)}</span>
            <small>
              {budget.rules.length} independent limit{budget.rules.length === 1 ? '' : 's'}
            </small>
          </div>
          <div className={styles.stackCell}>
            <span>
              {budget.rules.some((rule) => rule.enforcement === 'enforce')
                ? 'Enforced'
                : 'Observe only'}
            </span>
            <small>
              {budget.rules.filter((rule) => rule.enforcement === 'shadow').length} shadow
            </small>
          </div>
          <span>{date(budget.updatedAt)}</span>
          <span className={styles.rowChevron} aria-hidden="true">
            <ProductIcon name="chevron-right" />
          </span>
        </div>
      ))}
      {filtered.length === 0 ? (
        <Empty
          title="No budgets found"
          detail="Create a quota once, then assign it to users, Teams, or keys."
        />
      ) : null}
    </EntityTable>
  )
}
