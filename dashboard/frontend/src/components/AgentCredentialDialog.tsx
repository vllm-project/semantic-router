import { useEffect, useRef, useState, type FormEvent } from 'react'

import { agentManagementApi } from '../utils/agentManagementApi'
import type { AgentToolCredential } from '../generated/managementApiContract'
import AgentManagementDialog from './AgentManagementDialog'
import AgentInlineError from './AgentInlineError'
import ConfirmDialog from './ConfirmDialog'
import ProductIcon from './ProductIcon'
import styles from './AgentManagementPanel.module.css'

interface AgentCredentialDialogProps {
  canManage: boolean
  onClose: () => void
}

export default function AgentCredentialDialog({ canManage, onClose }: AgentCredentialDialogProps) {
  const [credentials, setCredentials] = useState<AgentToolCredential[]>([])
  const [search, setSearch] = useState('')
  const [cursor, setCursor] = useState<string | undefined>()
  const [hasMore, setHasMore] = useState(false)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [name, setName] = useState('')
  const [secret, setSecret] = useState('')
  const [rotateTarget, setRotateTarget] = useState<AgentToolCredential | null>(null)
  const [rotateSecret, setRotateSecret] = useState('')
  const [deleteTarget, setDeleteTarget] = useState<AgentToolCredential | null>(null)
  const loadGeneration = useRef(0)
  const loadController = useRef<AbortController | null>(null)

  const load = async (nextCursor?: string) => {
    const generation = ++loadGeneration.current
    loadController.current?.abort()
    const controller = new AbortController()
    loadController.current = controller
    setLoading(true)
    try {
      const page = await agentManagementApi.listToolCredentials(
        search.trim() || undefined,
        nextCursor,
        50,
        controller.signal,
      )
      if (controller.signal.aborted || generation !== loadGeneration.current) return
      setCredentials((current) =>
        nextCursor
          ? [...new Map([...current, ...page.data].map((item) => [item.id, item])).values()]
          : page.data,
      )
      setCursor(page.page.nextCursor)
      setHasMore(page.page.hasMore)
      setError(null)
    } catch (cause) {
      if (controller.signal.aborted || generation !== loadGeneration.current) return
      setError(cause instanceof Error ? cause.message : 'Credentials are unavailable.')
    } finally {
      if (!controller.signal.aborted && generation === loadGeneration.current) setLoading(false)
      if (loadController.current === controller) loadController.current = null
    }
  }

  useEffect(() => {
    const timer = window.setTimeout(() => void load(), 180)
    return () => {
      window.clearTimeout(timer)
      loadController.current?.abort()
      loadGeneration.current += 1
    }
  }, [search]) // eslint-disable-line react-hooks/exhaustive-deps

  const create = async (event: FormEvent) => {
    event.preventDefault()
    if (!canManage || !name.trim() || !secret) return
    setBusy(true)
    try {
      await agentManagementApi.createToolCredential({ name: name.trim(), secret })
      setName('')
      setSecret('')
      await load()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The credential could not be created.')
    } finally {
      setBusy(false)
    }
  }

  const updateStatus = async (credential: AgentToolCredential) => {
    if (!canManage) return
    setBusy(true)
    try {
      const detail = await agentManagementApi.getToolCredential(credential.id)
      await agentManagementApi.patchToolCredential(
        credential.id,
        { status: credential.status === 'active' ? 'disabled' : 'active' },
        detail.etag,
      )
      await load()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The credential could not be updated.')
    } finally {
      setBusy(false)
    }
  }

  const rotate = async (event: FormEvent) => {
    event.preventDefault()
    if (!canManage || !rotateTarget || !rotateSecret) return
    setBusy(true)
    try {
      const detail = await agentManagementApi.getToolCredential(rotateTarget.id)
      await agentManagementApi.rotateToolCredential(rotateTarget.id, rotateSecret, detail.etag)
      setRotateTarget(null)
      setRotateSecret('')
      await load()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The secret could not be replaced.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <AgentManagementDialog
        eyebrow="Connections"
        title="Credentials"
        description="Add or replace credentials securely."
        busy={busy}
        onClose={onClose}
      >
        <div className={styles.credentialBody}>
          {canManage ? (
            <form className={styles.credentialCreate} onSubmit={create}>
              <label className={styles.field}>
                <span>
                  Name <b>Required</b>
                </span>
                <input
                  required
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  placeholder="Production token"
                />
              </label>
              <label className={styles.field}>
                <span>
                  Secret <b>Required</b>
                </span>
                <input
                  required
                  type="password"
                  autoComplete="new-password"
                  value={secret}
                  onChange={(event) => setSecret(event.target.value)}
                  placeholder="Paste once"
                />
              </label>
              <button type="submit" className={styles.primaryButton} disabled={busy}>
                <ProductIcon name="plus" />
                Add
              </button>
            </form>
          ) : null}
          {error ? <AgentInlineError message={error} /> : null}
          <label className={styles.pickerSearch}>
            <ProductIcon name="search" />
            <span className={styles.srOnly}>Search credentials</span>
            <input
              type="search"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search credentials"
            />
          </label>
          <div className={styles.credentialList} aria-busy={loading}>
            {loading && credentials.length === 0 ? (
              <div className={styles.emptyState} role="status">
                <strong>Loading…</strong>
              </div>
            ) : null}
            {!loading && credentials.length === 0 ? (
              <div className={styles.emptyState}>
                <strong>No credentials</strong>
                <span>Add one when a connection needs authentication.</span>
              </div>
            ) : null}
            {credentials.map((credential) => (
              <article key={credential.id} className={styles.credentialRow}>
                <div className={styles.credentialMark}>
                  <ProductIcon name="key" />
                </div>
                <div>
                  <strong>{credential.name}</strong>
                  <span>
                    {credential.status === 'active' ? 'Ready' : 'Disabled'} · Updated{' '}
                    {new Intl.DateTimeFormat('en-US', {
                      month: 'short',
                      day: 'numeric',
                      year: 'numeric',
                      timeZone: 'UTC',
                    }).format(new Date(credential.updatedAt))}
                  </span>
                </div>
                {canManage ? (
                  <div className={styles.credentialActions}>
                    <button
                      type="button"
                      onClick={() => void updateStatus(credential)}
                      disabled={busy}
                    >
                      {credential.status === 'active' ? 'Disable' : 'Enable'}
                    </button>
                    <button
                      type="button"
                      onClick={() => setRotateTarget(credential)}
                      disabled={busy}
                    >
                      Replace secret
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setError(null)
                        setDeleteTarget(credential)
                      }}
                      disabled={busy}
                      aria-label={`Delete ${credential.name}`}
                    >
                      <ProductIcon name="trash" />
                    </button>
                  </div>
                ) : null}
              </article>
            ))}
          </div>
          {hasMore ? (
            <button
              type="button"
              className={styles.pickerMore}
              onClick={() => void load(cursor)}
              disabled={loading}
            >
              {loading ? 'Loading…' : 'Load more'}
            </button>
          ) : null}
          {rotateTarget ? (
            <form className={styles.rotateForm} onSubmit={rotate}>
              <div>
                <strong>Replace {rotateTarget.name}</strong>
                <span>The previous secret stops working after this succeeds.</span>
              </div>
              <input
                autoFocus
                required
                type="password"
                autoComplete="new-password"
                value={rotateSecret}
                onChange={(event) => setRotateSecret(event.target.value)}
                placeholder="New secret"
              />
              <button
                type="button"
                className={styles.secondaryButton}
                onClick={() => setRotateTarget(null)}
                disabled={busy}
              >
                Cancel
              </button>
              <button type="submit" className={styles.primaryButton} disabled={busy}>
                Replace
              </button>
            </form>
          ) : null}
        </div>
      </AgentManagementDialog>
      <ConfirmDialog
        isOpen={Boolean(deleteTarget)}
        title="Delete this credential?"
        description="Remove it from every connection first."
        error={deleteTarget ? error : null}
        confirmLabel="Delete"
        pending={busy}
        onCancel={() => {
          setDeleteTarget(null)
          setError(null)
        }}
        onConfirm={async () => {
          if (!canManage || !deleteTarget) return
          setBusy(true)
          try {
            const detail = await agentManagementApi.getToolCredential(deleteTarget.id)
            await agentManagementApi.deleteToolCredential(deleteTarget.id, detail.etag)
            setDeleteTarget(null)
            await load()
          } catch (cause) {
            setError(
              cause instanceof Error ? cause.message : 'The credential could not be deleted.',
            )
          } finally {
            setBusy(false)
          }
        }}
      />
    </>
  )
}
