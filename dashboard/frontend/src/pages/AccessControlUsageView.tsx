import { type CSSProperties, useEffect, useState } from 'react'

import type { AccessOverview, UsageSlice, UsageSummary } from '../utils/inferenceAccessApi'
import AccessAsyncResourcePicker from './AccessAsyncResourcePicker'
import AccessControlUsageTrend from './AccessControlUsageTrend'
import type { AccessControlSelectorSources } from './accessControlSelectorSources'
import { costCoverageLabel, formatCosts } from './AccessControlDetailSupport'
import {
  dateInputValue,
  rangeLabel,
  usageRangeDays,
  type UsageGranularity,
  type UsageScope,
} from './accessControlUsageRange'
import styles from './AccessControlPage.module.css'

export type { UsageScope } from './accessControlUsageRange'

interface Props {
  overview: AccessOverview
  usage: UsageSummary
  selectors: AccessControlSelectorSources
  usageScope: UsageScope
  onUsageScopeChange: (value: UsageScope) => void
  loading: boolean
}

const number = (value?: number | string | null) => {
  if (value === undefined || value === null) return '—'
  const formatter = new Intl.NumberFormat('en-US')
  return typeof value === 'string' ? formatter.format(BigInt(value)) : formatter.format(value)
}
const percent = (part: number, total: number) =>
  total ? `${((part / total) * 100).toFixed(1)}%` : '—'

export default function AccessControlUsageView(props: Props) {
  const [metric, setMetric] = useState<'tokens' | 'requests' | 'latency'>('tokens')
  const [dimension, setDimension] = useState<'model' | 'user' | 'team' | 'key'>('key')
  const tokensPerRequest = props.usage.requests
    ? Math.round(props.usage.totalTokens / props.usage.requests)
    : 0
  const outputShare = props.usage.totalTokens
    ? (props.usage.completionTokens / props.usage.totalTokens) * 100
    : 0
  const dimensionItems =
    dimension === 'model'
      ? props.usage.byModel
      : dimension === 'user'
        ? props.usage.byUser
        : dimension === 'team'
          ? props.usage.byTeam
          : props.usage.byKey
  const dimensionLabels = useUsageDimensionLabels(dimension, dimensionItems, props.selectors)
  const label = (id: string) => {
    return dimension === 'model' ? id : dimensionLabels[id] || id
  }

  return (
    <div className={styles.viewStack}>
      <UsageFilters {...props} />

      <AccessPosture overview={props.overview} />

      <div className={styles.usageMetricGrid}>
        <UsageMetric
          label="Requests"
          value={number(props.usage.requests)}
          detail={`${number(props.usage.failed)} failed`}
          tone="blue"
        />
        <UsageMetric
          label="Success rate"
          value={percent(props.usage.successful, props.usage.requests)}
          detail={`${number(props.usage.successful)} successful`}
          tone="green"
        />
        <UsageMetric
          label="Total tokens"
          value={number(props.usage.totalTokens)}
          detail={`${number(tokensPerRequest)} per request`}
          tone="violet"
        />
        <UsageMetric
          label="Spend"
          value={formatCosts(props.usage.costs)}
          detail={costCoverageLabel(props.usage.costs)}
          tone="violet"
        />
        <UsageMetric
          label="P95 latency"
          value={`${number(props.usage.p95LatencyMs)} ms`}
          detail={`${number(props.usage.averageLatencyMs)} ms average`}
          tone="amber"
        />
        <UsageMetric
          label="P95 first token"
          value={props.usage.p95TtftMs ? `${number(props.usage.p95TtftMs)} ms` : '—'}
          detail={
            props.usage.averageTtftMs
              ? `${number(props.usage.averageTtftMs)} ms average`
              : 'No TTFT samples'
          }
          tone="neutral"
        />
      </div>

      <section className={`${styles.panel} ${styles.usageTrendPanel}`}>
        <div className={styles.usagePanelHeader}>
          <div>
            <span>{`1 ${props.usage.granularity} per point`}</span>
            <h3>Traffic over time</h3>
          </div>
          <div className={styles.segmented}>
            {(['tokens', 'requests', 'latency'] as const).map((item) => (
              <button
                type="button"
                key={item}
                className={metric === item ? styles.segmentedActive : ''}
                onClick={() => setMetric(item)}
              >
                {item}
              </button>
            ))}
          </div>
        </div>
        <AccessControlUsageTrend
          points={props.usage.series}
          metric={metric}
          granularity={props.usage.granularity}
        />
      </section>

      <div className={styles.usageInsightGrid}>
        <section className={styles.panel}>
          <div className={styles.usagePanelHeader}>
            <div>
              <span>Token mix</span>
              <h3>Input vs output</h3>
            </div>
            <strong>{number(props.usage.totalTokens)}</strong>
          </div>
          <div className={styles.tokenMix}>
            <div
              className={styles.tokenDonut}
              style={{ '--output-share': `${outputShare * 3.6}deg` } as CSSProperties}
            >
              <span>
                {outputShare.toFixed(0)}%<small>output</small>
              </span>
            </div>
            <div className={styles.tokenLegend}>
              <div>
                <i className={styles.tokenInput} />
                <span>Input</span>
                <strong>{number(props.usage.promptTokens)}</strong>
              </div>
              <div>
                <i className={styles.tokenOutput} />
                <span>Output</span>
                <strong>{number(props.usage.completionTokens)}</strong>
              </div>
            </div>
          </div>
        </section>
        <section className={styles.panel}>
          <div className={styles.usagePanelHeader}>
            <div>
              <span>Reliability</span>
              <h3>Successful requests</h3>
            </div>
            <strong>{percent(props.usage.successful, props.usage.requests)}</strong>
          </div>
          <div className={styles.reliabilityBar}>
            <i
              style={{
                width: `${props.usage.requests ? (props.usage.successful / props.usage.requests) * 100 : 0}%`,
              }}
            />
          </div>
          <div className={styles.reliabilityStats}>
            <div>
              <span>Successful</span>
              <strong>{number(props.usage.successful)}</strong>
            </div>
            <div>
              <span>Failed</span>
              <strong>{number(props.usage.failed)}</strong>
            </div>
            <div>
              <span>Active keys</span>
              <strong>{number(props.usage.activeKeys)}</strong>
            </div>
          </div>
        </section>
        <section className={styles.panel}>
          <div className={styles.usagePanelHeader}>
            <div>
              <span>Latency</span>
              <h3>Response timing</h3>
            </div>
          </div>
          <div className={styles.timingStack}>
            <Timing
              label="Average latency"
              value={props.usage.averageLatencyMs}
              max={Math.max(props.usage.p95LatencyMs, 1)}
            />
            <Timing
              label="P95 latency"
              value={props.usage.p95LatencyMs}
              max={Math.max(props.usage.p95LatencyMs, 1)}
            />
            <Timing
              label="Average first token"
              value={props.usage.averageTtftMs}
              max={Math.max(props.usage.p95LatencyMs, 1)}
            />
          </div>
        </section>
      </div>

      <section className={`${styles.panel} ${styles.usageRankingPanel}`}>
        <div className={styles.usagePanelHeader}>
          <div>
            <span>Actual usage</span>
            <h3>{dimension === 'key' ? 'Spend by API key' : `Usage by ${dimension}`}</h3>
          </div>
          <div className={styles.segmented}>
            {(['model', 'user', 'team', 'key'] as const).map((item) => (
              <button
                type="button"
                key={item}
                className={dimension === item ? styles.segmentedActive : ''}
                onClick={() => setDimension(item)}
              >
                {item === 'key' ? 'API keys' : `${item}s`}
              </button>
            ))}
          </div>
        </div>
        <UsageRanking items={dimensionItems} label={label} />
      </section>
    </div>
  )
}

function AccessPosture({ overview }: { overview: AccessOverview }) {
  const identityCount = addAvailableCounts(overview.users, overview.teams)
  const identityDetail =
    overview.users !== null && overview.teams !== null
      ? `${number(overview.users)} users · ${number(overview.teams)} teams`
      : overview.users !== null
        ? `${number(overview.users)} users in scope`
        : `${number(overview.teams)} teams in scope`

  return (
    <section className={`${styles.panel} ${styles.usagePosturePanel}`}>
      <div className={styles.usagePanelHeader}>
        <div>
          <span>Control plane</span>
          <h3>Access at a glance</h3>
        </div>
        <strong>
          {overview.activeKeys === null
            ? 'Scoped view'
            : `${number(overview.activeKeys)} active keys`}
        </strong>
      </div>
      <div className={styles.usagePostureGrid}>
        <UsagePostureItem
          label="Today"
          value={number(overview.requestsToday)}
          detail={`${number(overview.tokensToday)} tokens`}
        />
        {overview.activeKeys !== null && (
          <UsagePostureItem
            label="API keys"
            value={number(overview.activeKeys)}
            detail={
              overview.expiringKeys === null
                ? 'available to you'
                : `${number(overview.expiringKeys)} expiring soon`
            }
          />
        )}
        {identityCount !== null && (
          <UsagePostureItem
            label="Identities"
            value={number(identityCount)}
            detail={identityDetail}
          />
        )}
        {overview.accessGroups !== null && (
          <UsagePostureItem
            label="Access groups"
            value={number(overview.accessGroups)}
            detail="model grants"
          />
        )}
        {overview.enabledBudgets !== null && (
          <UsagePostureItem
            label="Budgets"
            value={number(overview.enabledBudgets)}
            detail="active quota policies"
          />
        )}
      </div>
    </section>
  )
}

function addAvailableCounts(left: string | null, right: string | null): string | null {
  if (left === null && right === null) return null
  return (BigInt(left ?? '0') + BigInt(right ?? '0')).toString()
}

function UsagePostureItem({
  label,
  value,
  detail,
}: {
  label: string
  value: string
  detail: string
}) {
  return (
    <article className={styles.usagePostureItem}>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </article>
  )
}

function UsageFilters(props: Props) {
  const [customOpen, setCustomOpen] = useState(false)
  const today = dateInputValue(new Date())
  const weekAgo = new Date()
  weekAgo.setDate(weekAgo.getDate() - 6)
  const [customFrom, setCustomFrom] = useState(
    props.usageScope.customFrom || dateInputValue(weekAgo),
  )
  const [customTo, setCustomTo] = useState(props.usageScope.customTo || today)
  const [customError, setCustomError] = useState('')
  const models = [...new Set(props.usage.byModel.map((item) => item.id))]
  const changeRange = (range: UsageScope['range']) => {
    const next = { ...props.usageScope, range }
    const days = usageRangeDays(next)
    if ((next.granularity === 'minute' && days > 2) || (next.granularity === 'hour' && days > 90)) {
      next.granularity = 'auto'
    }
    props.onUsageScopeChange(next)
  }
  const applyCustomRange = () => {
    const from = Date.parse(`${customFrom}T00:00:00`)
    const to = Date.parse(`${customTo}T23:59:59`)
    if (!customFrom || !customTo || Number.isNaN(from) || Number.isNaN(to) || from > to) {
      setCustomError('Choose a valid date range.')
      return
    }
    if (to - from > 366 * 86_400_000) {
      setCustomError('Choose up to 366 days.')
      return
    }
    const next: UsageScope = {
      ...props.usageScope,
      range: 'custom',
      customFrom,
      customTo,
    }
    if (next.granularity === 'minute' && usageRangeDays(next) > 2) next.granularity = 'auto'
    if (next.granularity === 'hour' && usageRangeDays(next) > 90) next.granularity = 'auto'
    setCustomError('')
    setCustomOpen(false)
    props.onUsageScopeChange(next)
  }
  const rangeDays = usageRangeDays(props.usageScope)
  return (
    <div className={styles.filterRail}>
      <div className={styles.segmented}>
        {(['today', '7d', '30d', 'mtd', 'ytd'] as const).map((range) => (
          <button
            type="button"
            key={range}
            className={props.usageScope.range === range ? styles.segmentedActive : ''}
            onClick={() => changeRange(range)}
          >
            {rangeLabel(range)}
          </button>
        ))}
        <button
          type="button"
          className={props.usageScope.range === 'custom' ? styles.segmentedActive : ''}
          onClick={() => setCustomOpen((open) => !open)}
          aria-expanded={customOpen}
        >
          Custom
        </button>
      </div>
      {customOpen ? (
        <div className={styles.customRangePopover}>
          <div>
            <label>
              <span>From</span>
              <input
                type="date"
                value={customFrom}
                max={customTo || today}
                onChange={(event) => setCustomFrom(event.target.value)}
              />
            </label>
            <label>
              <span>To</span>
              <input
                type="date"
                value={customTo}
                min={customFrom}
                max={today}
                onChange={(event) => setCustomTo(event.target.value)}
              />
            </label>
          </div>
          {customError ? <p role="alert">{customError}</p> : null}
          <div>
            <button type="button" onClick={() => setCustomOpen(false)}>
              Cancel
            </button>
            <button type="button" onClick={applyCustomRange}>
              Apply
            </button>
          </div>
        </div>
      ) : null}
      <label>
        <span>Granularity</span>
        <select
          value={props.usageScope.granularity}
          onChange={(event) =>
            props.onUsageScopeChange({
              ...props.usageScope,
              granularity: event.target.value as UsageGranularity,
            })
          }
        >
          <option value="auto">Auto</option>
          <option value="minute" disabled={rangeDays > 2}>
            Minute
          </option>
          <option value="hour" disabled={rangeDays > 90}>
            Hour
          </option>
          <option value="day">Day</option>
        </select>
      </label>
      <label>
        <span>Scope</span>
        <select
          value={props.usageScope.type}
          onChange={(event) =>
            props.onUsageScopeChange({
              ...props.usageScope,
              type: event.target.value as UsageScope['type'],
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
        <div className={styles.usageScopePicker}>
          <span>{props.usageScope.type === 'key' ? 'API key' : props.usageScope.type}</span>
          <UsageSubjectPicker {...props} />
        </div>
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
    </div>
  )
}

function UsageSubjectPicker(props: Props) {
  const selectedIds = props.usageScope.id ? [props.usageScope.id] : []
  const onChange = (ids: string[]) =>
    props.onUsageScopeChange({ ...props.usageScope, id: ids[0] || '' })
  const common = {
    selectedIds,
    onChange,
    optional: true,
    compact: true,
  }
  if (props.usageScope.type === 'user') {
    return (
      <AccessAsyncResourcePicker
        key="usage-user"
        {...common}
        ariaLabel="Search usage users"
        source={props.selectors.users}
        optionalTitle="All users"
        optionalDescription="No user filter"
        compactEmptyLabel="All users"
        placeholder="Search name or email"
        emptyText="No users found"
      />
    )
  }
  if (props.usageScope.type === 'team') {
    return (
      <AccessAsyncResourcePicker
        key="usage-team"
        {...common}
        ariaLabel="Search usage Teams"
        source={props.selectors.teams}
        optionalTitle="All Teams"
        optionalDescription="No Team filter"
        compactEmptyLabel="All Teams"
        placeholder="Search Team name"
        emptyText="No Teams found"
      />
    )
  }
  return (
    <AccessAsyncResourcePicker
      key="usage-key"
      {...common}
      ariaLabel="Search usage API keys"
      source={props.selectors.keys}
      optionalTitle="All API keys"
      optionalDescription="No key filter"
      compactEmptyLabel="All API keys"
      placeholder="Search key name or ID"
      emptyText="No API keys found"
    />
  )
}

function useUsageDimensionLabels(
  dimension: 'model' | 'user' | 'team' | 'key',
  items: UsageSlice[],
  selectors: AccessControlSelectorSources,
) {
  const [labels, setLabels] = useState<Record<string, string>>({})

  useEffect(() => {
    if (dimension === 'model') return
    let cancelled = false
    const ids = [...new Set(items.map((item) => item.id))].slice(0, 100)
    const missing = ids.filter((id) => !labels[`${dimension}:${id}`])
    if (!missing.length) return

    void (async () => {
      const next: Record<string, string> = {}
      for (let offset = 0; offset < missing.length; offset += 6) {
        const batch = missing.slice(offset, offset + 6)
        const results = await Promise.allSettled(
          batch.map(async (id) => {
            if (dimension === 'user') return (await selectors.users.detail(id)).name
            if (dimension === 'team') return (await selectors.teams.detail(id)).name
            return (await selectors.keys.detail(id)).name
          }),
        )
        if (cancelled) return
        results.forEach((result, index) => {
          if (result.status === 'fulfilled') next[`${dimension}:${batch[index]}`] = result.value
        })
      }
      if (!cancelled && Object.keys(next).length) {
        setLabels((current) => ({ ...current, ...next }))
      }
    })()

    return () => {
      cancelled = true
    }
  }, [dimension, items, labels, selectors])

  return Object.fromEntries(
    items.map((item) => [item.id, labels[`${dimension}:${item.id}`] || item.id]),
  )
}

function UsageMetric({
  label,
  value,
  detail,
  tone,
}: {
  label: string
  value: string
  detail: string
  tone: string
}) {
  return (
    <article className={`${styles.usageMetric} ${styles[`metric${tone}`]}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </article>
  )
}

function Timing({ label, value, max }: { label: string; value: number; max: number }) {
  return (
    <div>
      <div>
        <span>{label}</span>
        <strong>{value ? `${number(value)} ms` : '—'}</strong>
      </div>
      <div>
        <i style={{ width: `${value ? Math.max(3, (value / max) * 100) : 0}%` }} />
      </div>
    </div>
  )
}

function UsageRanking({ items, label }: { items: UsageSlice[]; label: (id: string) => string }) {
  return (
    <div className={styles.usageTableWrap}>
      <table className={styles.usageTable}>
        <thead>
          <tr>
            <th>Name</th>
            <th>Requests</th>
            <th>Success</th>
            <th>Input</th>
            <th>Output</th>
            <th>Total tokens</th>
            <th>Spend</th>
            <th>P95</th>
          </tr>
        </thead>
        <tbody>
          {items.map((item) => (
            <tr key={item.id}>
              <td title={label(item.id)}>{label(item.id)}</td>
              <td>{number(item.requests)}</td>
              <td>{percent(item.successful, item.requests)}</td>
              <td>{number(item.promptTokens)}</td>
              <td>{number(item.completionTokens)}</td>
              <td>{number(item.totalTokens)}</td>
              <td title={formatCosts(item.costs)}>{formatCosts(item.costs)}</td>
              <td>{number(item.p95LatencyMs)} ms</td>
            </tr>
          ))}
          {!items.length ? (
            <tr>
              <td colSpan={8}>
                <div className={styles.usageEmpty}>
                  <strong>No activity</strong>
                  <span>No usage in this window.</span>
                </div>
              </td>
            </tr>
          ) : null}
        </tbody>
      </table>
    </div>
  )
}
