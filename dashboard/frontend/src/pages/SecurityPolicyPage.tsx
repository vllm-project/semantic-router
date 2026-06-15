import React, { useState, useEffect, useCallback } from 'react'

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
        data.rate_tiers = data.rate_tiers.map((t: RateLimitTier) => ({
          ...t,
          group: t.group ?? '',
          user: t.user ?? '',
          tpm: t.tpm ?? 0,
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
        let msg = `Failed to save policy (${res.status})`
        try { msg = JSON.parse(text).error || msg } catch { msg = res.status === 403 ? 'Permission denied. Only admins can modify the security policy.' : msg }
        throw new Error(msg)
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
    setPolicy((prev) => ({
      ...prev,
      role_mappings: [
        ...prev.role_mappings,
        {
          name: `role-mapping-${prev.role_mappings.length + 1}`,
          subjects: [{ kind: 'Group', name: '' }],
          role: '',
          model_refs: [],
          priority: (prev.role_mappings.length + 1) * 10,
        },
      ],
    }))
  }

  const removeRoleMapping = (index: number) => {
    setPolicy((prev) => ({
      ...prev,
      role_mappings: prev.role_mappings.filter((_, i) => i !== index),
    }))
  }

  const updateRoleMapping = (index: number, field: string, value: unknown) => {
    setPolicy((prev) => ({
      ...prev,
      role_mappings: prev.role_mappings.map((m, i) =>
        i === index ? { ...m, [field]: value } : m
      ),
    }))
  }

  const addRateTier = () => {
    setPolicy((prev) => ({
      ...prev,
      rate_tiers: [
        ...prev.rate_tiers,
        {
          name: `tier-${prev.rate_tiers.length + 1}`,
          group: '',
          user: '',
          rpm: 60,
          tpm: 0,
        },
      ],
    }))
  }

  const removeRateTier = (index: number) => {
    setPolicy((prev) => ({
      ...prev,
      rate_tiers: prev.rate_tiers.filter((_, i) => i !== index),
    }))
  }

  const updateRateTier = (index: number, field: string, value: unknown) => {
    setPolicy((prev) => ({
      ...prev,
      rate_tiers: prev.rate_tiers.map((t, i) =>
        i === index ? { ...t, [field]: value } : t
      ),
    }))
  }

  if (loading) {
    return (
      <div style={{ padding: '2rem' }}>
        <p style={{ color: 'var(--color-text-secondary)' }}>Loading security policy...</p>
      </div>
    )
  }

  return (
    <div style={{ padding: '2rem', maxWidth: '1200px', margin: '0 auto' }}>
      <div style={{ marginBottom: '2rem' }}>
        <h1 style={{ fontSize: '1.75rem', fontWeight: 700, marginBottom: '0.5rem' }}>
          Security Policy
        </h1>
        <p style={{ color: 'var(--color-text-secondary)', lineHeight: 1.6 }}>
          Map RBAC roles and groups to model access policies and rate-limit tiers.
          Changes here generate router config fragments that must be applied to take effect.
        </p>
      </div>

      {error && (
        <div
          style={{
            padding: '0.75rem 1rem',
            marginBottom: '1rem',
            borderRadius: '0.5rem',
            background: 'rgba(255, 60, 60, 0.1)',
            border: '1px solid rgba(255, 60, 60, 0.3)',
            color: '#ff6b6b',
          }}
        >
          {error}
        </div>
      )}

      {success && (
        <div
          style={{
            padding: '0.75rem 1rem',
            marginBottom: '1rem',
            borderRadius: '0.5rem',
            background: 'rgba(118, 185, 0, 0.1)',
            border: '1px solid rgba(118, 185, 0, 0.3)',
            color: '#76b900',
          }}
        >
          {success}
        </div>
      )}

      {/* Role Mappings Section */}
      <section style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
          <h2 style={{ fontSize: '1.25rem', fontWeight: 600 }}>Role-to-Model Mappings</h2>
          <button
            onClick={addRoleMapping}
            style={{
              padding: '0.5rem 1rem',
              borderRadius: '0.5rem',
              background: 'var(--color-primary)',
              color: '#081000',
              fontWeight: 600,
              border: 'none',
              cursor: 'pointer',
            }}
          >
            + Add Mapping
          </button>
        </div>

        {policy.role_mappings.length === 0 ? (
          <div style={{
            padding: '2rem',
            textAlign: 'center',
            borderRadius: '0.75rem',
            border: '1px dashed var(--color-border)',
            color: 'var(--color-text-secondary)',
          }}>
            No role mappings configured. Click "Add Mapping" to define role-to-model access policies.
          </div>
        ) : (
          policy.role_mappings.map((mapping, index) => (
            <div
              key={index}
              style={{
                padding: '1.25rem',
                marginBottom: '0.75rem',
                borderRadius: '0.75rem',
                border: '1px solid var(--color-border)',
                background: 'var(--color-bg-secondary)',
              }}
            >
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr auto', gap: '0.75rem', alignItems: 'end' }}>
                <div>
                  <label style={{ display: 'block', marginBottom: '0.25rem', fontSize: '0.8rem', color: 'var(--color-text-secondary)' }}>Name</label>
                  <input
                    value={mapping.name}
                    onChange={(e) => updateRoleMapping(index, 'name', e.target.value)}
                    style={inputStyle}
                  />
                </div>
                <div>
                  <label style={{ display: 'block', marginBottom: '0.25rem', fontSize: '0.8rem', color: 'var(--color-text-secondary)' }}>Router Role</label>
                  <input
                    value={mapping.role}
                    onChange={(e) => updateRoleMapping(index, 'role', e.target.value)}
                    placeholder="e.g., premium_tier"
                    style={inputStyle}
                  />
                </div>
                <div>
                  <label style={{ display: 'block', marginBottom: '0.25rem', fontSize: '0.8rem', color: 'var(--color-text-secondary)' }}>Models (comma-separated)</label>
                  <input
                    value={modelInputs[index] ?? mapping.model_refs.join(', ')}
                    onChange={(e) => setModelInputs((prev) => ({ ...prev, [index]: e.target.value }))}
                    onBlur={(e) => {
                      updateRoleMapping(
                        index,
                        'model_refs',
                        e.target.value.split(',').map((s) => s.trim()).filter(Boolean)
                      )
                      setModelInputs((prev) => { const next = { ...prev }; delete next[index]; return next })
                    }}
                    placeholder="e.g., gpt-4, claude-3"
                    style={inputStyle}
                  />
                </div>
                <button
                  onClick={() => removeRoleMapping(index)}
                  style={{ padding: '0.5rem', background: 'none', border: 'none', color: '#ff6b6b', cursor: 'pointer', fontSize: '1.1rem' }}
                  title="Remove"
                >
                  x
                </button>
              </div>
              <div style={{ marginTop: '0.5rem' }}>
                <label style={{ display: 'block', marginBottom: '0.25rem', fontSize: '0.8rem', color: 'var(--color-text-secondary)' }}>
                  Subjects (groups/users)
                </label>
                {mapping.subjects.map((subject, si) => (
                  <div key={si} style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.25rem' }}>
                    <select
                      value={subject.kind}
                      onChange={(e) => {
                        const newSubjects = [...mapping.subjects]
                        newSubjects[si] = { ...subject, kind: e.target.value }
                        updateRoleMapping(index, 'subjects', newSubjects)
                      }}
                      style={{ ...inputStyle, width: '100px' }}
                    >
                      <option value="Group">Group</option>
                      <option value="User">User</option>
                    </select>
                    <input
                      value={subject.name}
                      onChange={(e) => {
                        const newSubjects = [...mapping.subjects]
                        newSubjects[si] = { ...subject, name: e.target.value }
                        updateRoleMapping(index, 'subjects', newSubjects)
                      }}
                      placeholder="Group or user name"
                      style={{ ...inputStyle, flex: 1 }}
                    />
                  </div>
                ))}
              </div>
            </div>
          ))
        )}
      </section>

      {/* Rate Limit Tiers Section */}
      <section style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
          <h2 style={{ fontSize: '1.25rem', fontWeight: 600 }}>Rate Limit Tiers</h2>
          <button
            onClick={addRateTier}
            style={{
              padding: '0.5rem 1rem',
              borderRadius: '0.5rem',
              background: 'var(--color-primary)',
              color: '#081000',
              fontWeight: 600,
              border: 'none',
              cursor: 'pointer',
            }}
          >
            + Add Tier
          </button>
        </div>

        {policy.rate_tiers.length === 0 ? (
          <div style={{
            padding: '2rem',
            textAlign: 'center',
            borderRadius: '0.75rem',
            border: '1px dashed var(--color-border)',
            color: 'var(--color-text-secondary)',
          }}>
            No rate tiers configured. Click "Add Tier" to set per-role/group rate limits.
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--color-border)' }}>
                  <th style={thStyle}>Name</th>
                  <th style={thStyle}>Group</th>
                  <th style={thStyle}>User</th>
                  <th style={thStyle}>RPM</th>
                  <th style={thStyle}>TPM</th>
                  <th style={{ ...thStyle, width: 40 }}></th>
                </tr>
              </thead>
              <tbody>
                {policy.rate_tiers.map((tier, index) => (
                  <tr key={index} style={{ borderBottom: '1px solid var(--color-border)' }}>
                    <td style={tdStyle}>
                      <input value={tier.name} onChange={(e) => updateRateTier(index, 'name', e.target.value)} style={inputStyle} />
                    </td>
                    <td style={tdStyle}>
                      <input value={tier.group} onChange={(e) => updateRateTier(index, 'group', e.target.value)} placeholder="*" style={inputStyle} />
                    </td>
                    <td style={tdStyle}>
                      <input value={tier.user} onChange={(e) => updateRateTier(index, 'user', e.target.value)} placeholder="*" style={inputStyle} />
                    </td>
                    <td style={tdStyle}>
                      <input type="number" value={tier.rpm} onChange={(e) => updateRateTier(index, 'rpm', parseInt(e.target.value) || 0)} style={{ ...inputStyle, width: 80 }} />
                    </td>
                    <td style={tdStyle}>
                      <input type="number" value={tier.tpm} onChange={(e) => updateRateTier(index, 'tpm', parseInt(e.target.value) || 0)} style={{ ...inputStyle, width: 80 }} />
                    </td>
                    <td style={tdStyle}>
                      <button
                        onClick={() => removeRateTier(index)}
                        style={{ background: 'none', border: 'none', color: '#ff6b6b', cursor: 'pointer' }}
                      >
                        x
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* Action Buttons */}
      <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '2rem' }}>
        <button
          onClick={handlePreview}
          style={{
            padding: '0.75rem 1.5rem',
            borderRadius: '0.5rem',
            background: 'var(--color-bg-secondary)',
            color: 'var(--color-text)',
            fontWeight: 600,
            border: '1px solid var(--color-border)',
            cursor: 'pointer',
          }}
        >
          Preview Config Fragment
        </button>
        <button
          onClick={() => void handleSave()}
          disabled={saving}
          style={{
            padding: '0.75rem 1.5rem',
            borderRadius: '0.5rem',
            background: 'var(--color-primary)',
            color: '#081000',
            fontWeight: 700,
            border: 'none',
            cursor: saving ? 'not-allowed' : 'pointer',
            opacity: saving ? 0.6 : 1,
          }}
        >
          {saving ? 'Saving...' : 'Save & Generate'}
        </button>
      </div>

      {/* Generated Fragment Preview */}
      {fragment && (
        <section>
          <h2 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '0.75rem' }}>
            Generated Router Config Fragment
          </h2>
          <pre
            style={{
              padding: '1rem',
              borderRadius: '0.75rem',
              border: '1px solid var(--color-border)',
              background: 'var(--color-bg-secondary)',
              overflow: 'auto',
              fontSize: '0.85rem',
              lineHeight: 1.5,
              maxHeight: '400px',
            }}
          >
            {JSON.stringify(fragment, null, 2)}
          </pre>
        </section>
      )}
    </div>
  )
}

const inputStyle: React.CSSProperties = {
  width: '100%',
  padding: '0.5rem 0.75rem',
  borderRadius: '0.375rem',
  border: '1px solid var(--color-border)',
  background: 'var(--color-bg)',
  color: 'var(--color-text)',
  fontSize: '0.875rem',
}

const thStyle: React.CSSProperties = {
  textAlign: 'left',
  padding: '0.5rem 0.75rem',
  fontSize: '0.8rem',
  color: 'var(--color-text-secondary)',
  fontWeight: 600,
}

const tdStyle: React.CSSProperties = {
  padding: '0.5rem 0.75rem',
}

export default SecurityPolicyPage
