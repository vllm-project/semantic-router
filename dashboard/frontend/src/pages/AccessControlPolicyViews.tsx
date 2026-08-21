import type { AccessControlViewProps as Props } from './AccessControlViewTypes'
import {
  Empty,
  EntityTable,
  ListToolbar,
  Pagination,
  Quota,
  Status,
} from './AccessControlViewPrimitives'
import { date, keyPolicy, number } from './AccessControlViewSupport'
import styles from './AccessControlPage.module.css'

export function KeysView(props: Props) {
  const filtered = props.keys
  return (
    <EntityTable
      toolbar={
        <ListToolbar
          state={props.pageState}
          onChange={props.onPageStateChange}
          placeholder="Search keys or owners"
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
        const policy = keyPolicy(key, props.groups)
        const linkedBudget = props.budgets.find(
          (budget) => budget.id === (key.effectiveBudgetId || key.budgetId),
        )
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
              <span>{policy.patterns.length ? policy.patterns.join(', ') : 'No models'}</span>
              <small>{policy.direct ? 'Key override' : 'Inherited policy'}</small>
            </div>
            <div className={styles.stackCell}>
              <span>{linkedBudget?.name || 'Inherited'}</span>
              <small>
                {linkedBudget
                  ? `${number(linkedBudget.rpm)} RPM · ${number(linkedBudget.tpm)} TPM`
                  : 'Owner policy applies'}
              </small>
            </div>
            <div className={styles.stackCell}>
              <Status value={key.status} />
              <small>Used {date(key.lastUsedAt)}</small>
            </div>
            <span className={styles.rowChevron} aria-hidden="true">
              ›
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
            {group.modelPatterns.map((pattern) => (
              <code key={pattern}>{pattern}</code>
            ))}
          </div>
          <div className={styles.stackCell}>
            <span>{group.assignmentCount} assignments</span>
            <small>Assigned from users, Teams, or keys</small>
          </div>
          <span>{date(group.updatedAt)}</span>
          <span className={styles.rowChevron} aria-hidden="true">
            ›
          </span>
        </div>
      ))}
      {filtered.length === 0 ? (
        <Empty
          title="No access groups found"
          detail="Bundle model patterns once, then assign them to users, teams, or keys."
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
        <span>Requests / min</span>
        <span>Tokens / min</span>
        <span>Daily tokens</span>
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
          <Quota value={budget.rpm} />
          <Quota value={budget.tpm} />
          <Quota value={budget.dailyTokens} />
          <span className={styles.rowChevron} aria-hidden="true">
            ›
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
