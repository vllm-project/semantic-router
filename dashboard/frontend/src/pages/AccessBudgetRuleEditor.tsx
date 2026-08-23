import type { RateLimitRule } from '../utils/routerManagementTypes'
import ProductIcon from '../components/ProductIcon'
import styles from './AccessControlPage.module.css'
import {
  allowedAlgorithms,
  durationInput,
  isoDuration,
  normalizeRule,
} from './accessBudgetRuleSupport'

interface Props {
  rules: RateLimitRule[]
  onChange: (rules: RateLimitRule[]) => void
}

const metricOptions: Array<{ value: RateLimitRule['metric']; label: string }> = [
  { value: 'requests', label: 'Requests' },
  { value: 'total_tokens', label: 'Tokens' },
  { value: 'input_tokens', label: 'Input tokens' },
  { value: 'output_tokens', label: 'Output tokens' },
  { value: 'served_total_tokens', label: 'Served tokens' },
  { value: 'served_input_tokens', label: 'Served input tokens' },
  { value: 'served_output_tokens', label: 'Served output tokens' },
  { value: 'cost', label: 'Spend' },
  { value: 'concurrent_requests', label: 'Concurrent requests' },
]

const algorithmLabels: Record<RateLimitRule['algorithm'], string> = {
  sliding_log: 'Rolling window',
  calendar_window: 'Calendar window',
  token_bucket: 'Token bucket',
  gcra: 'Smooth rate',
  concurrency: 'Concurrency',
}

function newRule(): RateLimitRule {
  return {
    metric: 'requests',
    algorithm: 'sliding_log',
    limit: '60',
    window: 'PT1M',
    accounting: 'request',
    enforcement: 'enforce',
  }
}

export default function AccessBudgetRuleEditor({ rules, onChange }: Props) {
  const update = (index: number, patch: Partial<RateLimitRule>) =>
    onChange(
      rules.map((rule, candidate) =>
        candidate === index ? normalizeRule({ ...rule, ...patch }) : rule,
      ),
    )
  const remove = (index: number) => onChange(rules.filter((_, candidate) => candidate !== index))

  return (
    <section className={styles.budgetRuleSection}>
      <header>
        <div>
          <strong>Limits</strong>
          <span>Each limit is enforced independently.</span>
        </div>
        <button type="button" onClick={() => onChange([...rules, newRule()])}>
          <ProductIcon name="plus" /> Add limit
        </button>
      </header>
      <div className={styles.budgetRuleList}>
        {rules.map((rule, index) => {
          const algorithms = allowedAlgorithms(rule.metric)
          return (
            <article className={styles.budgetRule} key={rule.ruleId || index}>
              <div className={styles.budgetRuleHeading}>
                <strong>Limit {index + 1}</strong>
                <button
                  type="button"
                  onClick={() => remove(index)}
                  aria-label={`Remove limit ${index + 1}`}
                >
                  <ProductIcon name="trash" /> Remove
                </button>
              </div>
              <div className={styles.budgetRuleGrid}>
                <label>
                  <span>Measure</span>
                  <select
                    value={rule.metric}
                    onChange={(event) =>
                      update(index, { metric: event.target.value as RateLimitRule['metric'] })
                    }
                  >
                    {metricOptions.map((option) => (
                      <option value={option.value} key={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  <span>Policy</span>
                  <select
                    value={rule.algorithm}
                    onChange={(event) =>
                      update(index, { algorithm: event.target.value as RateLimitRule['algorithm'] })
                    }
                  >
                    {algorithms.map((algorithm) => (
                      <option value={algorithm} key={algorithm}>
                        {algorithmLabels[algorithm]}
                      </option>
                    ))}
                  </select>
                </label>
                {rule.algorithm === 'sliding_log' ? (
                  <>
                    <ExactLimit rule={rule} onChange={(limit) => update(index, { limit })} />
                    <DurationField
                      label="Window"
                      value={rule.window}
                      onChange={(window) => update(index, { window })}
                    />
                  </>
                ) : null}
                {rule.algorithm === 'calendar_window' ? (
                  <>
                    <ExactLimit rule={rule} onChange={(limit) => update(index, { limit })} />
                    <label>
                      <span>Period</span>
                      <select
                        value={rule.period || 'day'}
                        onChange={(event) =>
                          update(index, { period: event.target.value as 'day' | 'month' })
                        }
                      >
                        <option value="day">Day</option>
                        <option value="month">Month</option>
                      </select>
                    </label>
                    <label>
                      <span>Timezone</span>
                      <input
                        value={rule.timezone || 'UTC'}
                        onChange={(event) => update(index, { timezone: event.target.value })}
                        placeholder="UTC"
                        required
                      />
                    </label>
                  </>
                ) : null}
                {rule.algorithm === 'token_bucket' ? (
                  <>
                    <ExactField
                      label="Capacity"
                      value={rule.capacity}
                      onChange={(capacity) => update(index, { capacity })}
                    />
                    <ExactField
                      label="Refill amount"
                      value={rule.refillAmount}
                      onChange={(refillAmount) => update(index, { refillAmount })}
                    />
                    <DurationField
                      label="Refill every"
                      value={rule.refillPeriod}
                      onChange={(refillPeriod) => update(index, { refillPeriod })}
                    />
                  </>
                ) : null}
                {rule.algorithm === 'gcra' ? (
                  <>
                    <DurationField
                      label="Request spacing"
                      value={rule.emissionInterval}
                      onChange={(emissionInterval) => update(index, { emissionInterval })}
                    />
                    <ExactField
                      label="Burst tolerance"
                      value={String(rule.burstTolerance ?? 0)}
                      allowZero
                      onChange={(value) => update(index, { burstTolerance: Number(value) })}
                    />
                  </>
                ) : null}
                {rule.algorithm === 'concurrency' ? (
                  <ExactLimit rule={rule} onChange={(limit) => update(index, { limit })} />
                ) : null}
                <label>
                  <span>Mode</span>
                  <select
                    value={rule.enforcement}
                    onChange={(event) =>
                      update(index, {
                        enforcement: event.target.value as RateLimitRule['enforcement'],
                      })
                    }
                  >
                    <option value="enforce">Enforce</option>
                    <option value="shadow">Observe only</option>
                  </select>
                </label>
              </div>
              {rule.metric === 'cost' ? (
                <p>Spend uses the workspace billing currency and actual settled model usage.</p>
              ) : null}
            </article>
          )
        })}
        {!rules.length ? (
          <div className={styles.budgetRuleEmpty}>
            <strong>No limits</strong>
            <span>Add a request, token, spend, or concurrency limit.</span>
          </div>
        ) : null}
      </div>
    </section>
  )
}

function ExactLimit({
  rule,
  onChange,
}: {
  rule: RateLimitRule
  onChange: (value: string) => void
}) {
  return (
    <ExactField
      label={rule.metric === 'cost' ? 'Amount' : 'Limit'}
      value={rule.limit}
      decimal={rule.metric === 'cost'}
      onChange={onChange}
    />
  )
}

function ExactField({
  label,
  value,
  decimal = false,
  allowZero = false,
  onChange,
}: {
  label: string
  value?: string
  decimal?: boolean
  allowZero?: boolean
  onChange: (value: string) => void
}) {
  return (
    <label>
      <span>{label}</span>
      <input
        type="text"
        inputMode={decimal ? 'decimal' : 'numeric'}
        pattern={
          decimal ? '(0|[1-9][0-9]*)(\\.[0-9]{1,15})?' : allowZero ? '[0-9]+' : '[1-9][0-9]*'
        }
        value={value || ''}
        onChange={(event) => onChange(event.target.value)}
        required
      />
    </label>
  )
}

function DurationField({
  label,
  value,
  onChange,
}: {
  label: string
  value?: string
  onChange: (value: string) => void
}) {
  return (
    <label>
      <span>{label}</span>
      <input
        type="text"
        value={durationInput(value)}
        onChange={(event) => onChange(isoDuration(event.target.value))}
        pattern="([1-9][0-9]*)(s|m|h|d)|P.*"
        placeholder="8h"
        title="Use a duration such as 60s, 5m, 8h, or 1d."
        required
      />
    </label>
  )
}
