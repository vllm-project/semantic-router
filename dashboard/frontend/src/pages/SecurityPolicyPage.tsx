import React, { useCallback, useEffect, useState } from 'react'
import styles from './SecurityPolicyPage.module.css'

interface Subject {
  kind: string
  name: string
}

interface RoleMapping {
  name: string
  subjects: Subject[]
  role: string
  model_refs: string[]
  priority: number
}

interface RateLimitTier {
  name: string
  group?: string
  user?: string
  rpm: number
  tpm?: number
}

interface SecurityPolicy {
  role_mappings: RoleMapping[]
  rate_tiers: RateLimitTier[]
  updated_at: string
}

interface GeneratedFragment {
  role_bindings: unknown[]
  decisions: unknown[]
  ratelimit: unknown
}

const emptyPolicy: SecurityPolicy = {
  role_mappings: [],
  rate_tiers: [],
  updated_at: '',
}

const SecurityPolicyPage: React.FC = () => {
  const [policy, setPolicy] = useState<SecurityPolicy>(emptyPolicy)
  const [fragment, setFragment] = useState<GeneratedFragment | null>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [modelInputs, setModelInputs] = useState<Record<number, string>>({})

  const fetchPolicy = useCallback(async () => {
    try {
      setLoading(true)
      const res = await fetch('/api/security/policy')
      if (!res.ok) throw new Error(`Failed to load policy: ${res.statusText}`)
      const data = await res.json()
      if (data.rate_tiers) {
        data.rate_tiers = data.rate_tiers.map((tier: RateLimitTier) => ({
          ...tier,
          group: tier.group ?? '',
          user: tier.user ?? '',
          tpm: tier.tpm ?? 0,
        }))
      }
      setPolicy(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load policy')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void fetchPolicy()
  }, [fetchPolicy])

  const handleSave = async () => {
    try {
      setSaving(true)
      setError(null)
      setSuccess(null)
      const res = await fetch('/api/security/policy', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(policy),
      })
      if (!res.ok) {
        const text = await res.text()
        let message = `Failed to save policy (${res.status})`
        try {
          message = JSON.parse(text).error || message
        } catch {
          message =
            res.status === 403
              ? 'Permission denied. Only admins can modify the security policy.'
              : message
        }
        throw new Error(message)
      }
      const data = await res.json()
      setPolicy(data.policy)
      setFragment(data.fragment)
      setSuccess(data.message)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save policy')
    } finally {
      setSaving(false)
    }
  }

  const handlePreview = async () => {
    try {
      setError(null)
      const res = await fetch('/api/security/policy/preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(policy),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || 'Failed to generate preview')
      setFragment(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate preview')
    }
  }

  const addRoleMapping = () => {
    setPolicy((previous) => ({
      ...previous,
      role_mappings: [
        ...previous.role_mappings,
        {
          name: `role-mapping-${previous.role_mappings.length + 1}`,
          subjects: [{ kind: 'Group', name: '' }],
          role: '',
          model_refs: [],
          priority: (previous.role_mappings.length + 1) * 10,
        },
      ],
    }))
  }

  const removeRoleMapping = (index: number) => {
    setPolicy((previous) => ({
      ...previous,
      role_mappings: previous.role_mappings.filter((_, itemIndex) => itemIndex !== index),
    }))
  }

  const updateRoleMapping = (index: number, field: string, value: unknown) => {
    setPolicy((previous) => ({
      ...previous,
      role_mappings: previous.role_mappings.map((mapping, itemIndex) =>
        itemIndex === index ? { ...mapping, [field]: value } : mapping,
      ),
    }))
  }

  const addRateTier = () => {
    setPolicy((previous) => ({
      ...previous,
      rate_tiers: [
        ...previous.rate_tiers,
        {
          name: `tier-${previous.rate_tiers.length + 1}`,
          group: '',
          user: '',
          rpm: 60,
          tpm: 0,
        },
      ],
    }))
  }

  const removeRateTier = (index: number) => {
    setPolicy((previous) => ({
      ...previous,
      rate_tiers: previous.rate_tiers.filter((_, itemIndex) => itemIndex !== index),
    }))
  }

  const updateRateTier = (index: number, field: string, value: unknown) => {
    setPolicy((previous) => ({
      ...previous,
      rate_tiers: previous.rate_tiers.map((tier, itemIndex) =>
        itemIndex === index ? { ...tier, [field]: value } : tier,
      ),
    }))
  }

  if (loading) {
    return (
      <div className={styles.loading} role="status" aria-live="polite">
        <span className={styles.loadingIndicator} aria-hidden="true" />
        <p>Loading security policy...</p>
      </div>
    )
  }

  return (
    <main className={styles.page}>
      <header className={styles.hero}>
        <p className={styles.eyebrow}>Access control</p>
        <h1>Security Policy</h1>
        <p className={styles.intro}>
          Map RBAC roles and groups to model access policies and rate-limit tiers. Changes generate
          router config fragments that must be applied to take effect.
        </p>
      </header>

      {error ? (
        <div className={`${styles.notice} ${styles.errorNotice}`} role="alert">
          {error}
        </div>
      ) : null}

      {success ? (
        <div
          className={`${styles.notice} ${styles.successNotice}`}
          role="status"
          aria-live="polite"
        >
          {success}
        </div>
      ) : null}

      <section className={styles.section} aria-labelledby="role-mappings-heading">
        <div className={styles.sectionHeader}>
          <div>
            <p className={styles.sectionKicker}>Policy routing</p>
            <h2 id="role-mappings-heading">Role-to-Model Mappings</h2>
            <p>Bind users or groups to router roles and an explicit model allowlist.</p>
          </div>
          <button type="button" className={styles.primaryButton} onClick={addRoleMapping}>
            <span aria-hidden="true">+</span> Add Mapping
          </button>
        </div>

        {policy.role_mappings.length === 0 ? (
          <div className={styles.emptyState}>
            No role mappings configured. Add a mapping to define model access.
          </div>
        ) : (
          <div className={styles.mappingList}>
            {policy.role_mappings.map((mapping, index) => (
              <article className={styles.mappingCard} data-testid="role-mapping-card" key={index}>
                <div className={styles.mappingGrid}>
                  <div className={styles.field}>
                    <label htmlFor={`mapping-name-${index}`}>Name</label>
                    <input
                      id={`mapping-name-${index}`}
                      className={styles.input}
                      value={mapping.name}
                      onChange={(event) => updateRoleMapping(index, 'name', event.target.value)}
                    />
                  </div>
                  <div className={styles.field}>
                    <label htmlFor={`mapping-role-${index}`}>Router Role</label>
                    <input
                      id={`mapping-role-${index}`}
                      className={styles.input}
                      value={mapping.role}
                      onChange={(event) => updateRoleMapping(index, 'role', event.target.value)}
                      placeholder="e.g., premium_tier"
                    />
                  </div>
                  <div className={styles.field}>
                    <label htmlFor={`mapping-models-${index}`}>Models (comma-separated)</label>
                    <input
                      id={`mapping-models-${index}`}
                      className={styles.input}
                      value={modelInputs[index] ?? mapping.model_refs.join(', ')}
                      onChange={(event) =>
                        setModelInputs((previous) => ({
                          ...previous,
                          [index]: event.target.value,
                        }))
                      }
                      onBlur={(event) => {
                        updateRoleMapping(
                          index,
                          'model_refs',
                          event.target.value
                            .split(',')
                            .map((value) => value.trim())
                            .filter(Boolean),
                        )
                        setModelInputs((previous) => {
                          const next = { ...previous }
                          delete next[index]
                          return next
                        })
                      }}
                      placeholder="e.g., gpt-4, claude-3"
                    />
                  </div>
                  <button
                    type="button"
                    className={styles.removeButton}
                    onClick={() => removeRoleMapping(index)}
                    aria-label={`Remove role mapping ${mapping.name || index + 1}`}
                  >
                    <span aria-hidden="true">×</span>
                  </button>
                </div>

                <fieldset className={styles.subjects}>
                  <legend>Subjects (groups/users)</legend>
                  {mapping.subjects.map((subject, subjectIndex) => (
                    <div className={styles.subjectRow} key={subjectIndex}>
                      <label
                        className={styles.visuallyHidden}
                        htmlFor={`subject-kind-${index}-${subjectIndex}`}
                      >
                        Subject type {subjectIndex + 1}
                      </label>
                      <select
                        id={`subject-kind-${index}-${subjectIndex}`}
                        className={`${styles.input} ${styles.subjectKind}`}
                        value={subject.kind}
                        onChange={(event) => {
                          const newSubjects = [...mapping.subjects]
                          newSubjects[subjectIndex] = { ...subject, kind: event.target.value }
                          updateRoleMapping(index, 'subjects', newSubjects)
                        }}
                      >
                        <option value="Group">Group</option>
                        <option value="User">User</option>
                      </select>
                      <label
                        className={styles.visuallyHidden}
                        htmlFor={`subject-name-${index}-${subjectIndex}`}
                      >
                        Subject name {subjectIndex + 1}
                      </label>
                      <input
                        id={`subject-name-${index}-${subjectIndex}`}
                        className={styles.input}
                        value={subject.name}
                        onChange={(event) => {
                          const newSubjects = [...mapping.subjects]
                          newSubjects[subjectIndex] = { ...subject, name: event.target.value }
                          updateRoleMapping(index, 'subjects', newSubjects)
                        }}
                        placeholder="Group or user name"
                      />
                    </div>
                  ))}
                </fieldset>
              </article>
            ))}
          </div>
        )}
      </section>

      <section className={styles.section} aria-labelledby="rate-tiers-heading">
        <div className={styles.sectionHeader}>
          <div>
            <p className={styles.sectionKicker}>Traffic guardrails</p>
            <h2 id="rate-tiers-heading">Rate Limit Tiers</h2>
            <p>Set request and token budgets for a matching group or individual user.</p>
          </div>
          <button type="button" className={styles.primaryButton} onClick={addRateTier}>
            <span aria-hidden="true">+</span> Add Tier
          </button>
        </div>

        {policy.rate_tiers.length === 0 ? (
          <div className={styles.emptyState}>
            No rate tiers configured. Add a tier to define traffic limits.
          </div>
        ) : (
          <div
            className={styles.tableScroller}
            data-testid="rate-tier-scroller"
            role="region"
            aria-label="Rate limit tiers table"
            tabIndex={0}
          >
            <table className={styles.rateTable}>
              <caption className={styles.visuallyHidden}>Configured rate limit tiers</caption>
              <thead>
                <tr>
                  <th scope="col">Name</th>
                  <th scope="col">Group</th>
                  <th scope="col">User</th>
                  <th scope="col">RPM</th>
                  <th scope="col">TPM</th>
                  <th scope="col">
                    <span className={styles.visuallyHidden}>Actions</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {policy.rate_tiers.map((tier, index) => (
                  <tr key={index}>
                    <td>
                      <input
                        className={styles.input}
                        aria-label={`Rate tier ${index + 1} name`}
                        value={tier.name}
                        onChange={(event) => updateRateTier(index, 'name', event.target.value)}
                      />
                    </td>
                    <td>
                      <input
                        className={styles.input}
                        aria-label={`Rate tier ${index + 1} group`}
                        value={tier.group}
                        onChange={(event) => updateRateTier(index, 'group', event.target.value)}
                        placeholder="*"
                      />
                    </td>
                    <td>
                      <input
                        className={styles.input}
                        aria-label={`Rate tier ${index + 1} user`}
                        value={tier.user}
                        onChange={(event) => updateRateTier(index, 'user', event.target.value)}
                        placeholder="*"
                      />
                    </td>
                    <td>
                      <input
                        className={`${styles.input} ${styles.numberInput}`}
                        aria-label={`Rate tier ${index + 1} requests per minute`}
                        type="number"
                        value={tier.rpm}
                        onChange={(event) =>
                          updateRateTier(index, 'rpm', parseInt(event.target.value) || 0)
                        }
                      />
                    </td>
                    <td>
                      <input
                        className={`${styles.input} ${styles.numberInput}`}
                        aria-label={`Rate tier ${index + 1} tokens per minute`}
                        type="number"
                        value={tier.tpm}
                        onChange={(event) =>
                          updateRateTier(index, 'tpm', parseInt(event.target.value) || 0)
                        }
                      />
                    </td>
                    <td className={styles.actionCell}>
                      <button
                        type="button"
                        className={styles.removeButton}
                        onClick={() => removeRateTier(index)}
                        aria-label={`Remove rate tier ${tier.name || index + 1}`}
                      >
                        <span aria-hidden="true">×</span>
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <div className={styles.actions} aria-label="Security policy actions">
        <button
          type="button"
          className={styles.secondaryButton}
          onClick={() => void handlePreview()}
        >
          Preview Config Fragment
        </button>
        <button
          type="button"
          className={styles.primaryButton}
          onClick={() => void handleSave()}
          disabled={saving}
        >
          {saving ? 'Saving...' : 'Save & Generate'}
        </button>
      </div>

      {fragment ? (
        <section
          className={`${styles.section} ${styles.fragmentSection}`}
          aria-labelledby="fragment-heading"
        >
          <div className={styles.sectionHeader}>
            <div>
              <p className={styles.sectionKicker}>Generated output</p>
              <h2 id="fragment-heading">Router Config Fragment</h2>
            </div>
          </div>
          <pre className={styles.fragment}>{JSON.stringify(fragment, null, 2)}</pre>
        </section>
      ) : null}
    </main>
  )
}

export default SecurityPolicyPage
