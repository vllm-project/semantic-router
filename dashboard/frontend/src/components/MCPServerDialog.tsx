import React, { useId, useState } from 'react'
import type { MCPServerConfig, MCPTransportType } from '../tools/mcp'
import useAccessibleDialog from '../hooks/useAccessibleDialog'
import styles from './MCPConfigPanel.module.css'
import { KeyValueEditor } from './KeyValueEditor'
import { buildServerConfig, buildTestServerConfig } from './mcpConfigPanelUtils'
import { StringListEditor } from './StringListEditor'

interface MCPServerDialogProps {
  server: MCPServerConfig | null
  onSave: (config: Omit<MCPServerConfig, 'id'>) => Promise<void>
  onTest: (config: MCPServerConfig) => Promise<{ success: boolean; error?: string }>
  onClose: () => void
}

export const MCPServerDialog: React.FC<MCPServerDialogProps> = ({
  server,
  onSave,
  onTest,
  onClose,
}) => {
  const dialogId = useId()
  const titleId = `${dialogId}-title`
  const nameInputId = `${dialogId}-name`
  const descriptionInputId = `${dialogId}-description`
  const transportLabelId = `${dialogId}-transport-label`
  const commandInputId = `${dialogId}-command`
  const argsLabelId = `${dialogId}-args-label`
  const urlInputId = `${dialogId}-url`
  const headersLabelId = `${dialogId}-headers-label`
  const timeoutInputId = `${dialogId}-timeout`
  const [name, setName] = useState(server?.name || '')
  const [description, setDescription] = useState(server?.description || '')
  const [transport, setTransport] = useState<MCPTransportType>(server?.transport || 'stdio')
  const [enabled, setEnabled] = useState(server?.enabled ?? true)
  const [command, setCommand] = useState(server?.connection?.command || '')
  const [args, setArgs] = useState<string[]>(() => [...(server?.connection?.args || [])])
  const [url, setUrl] = useState(server?.connection?.url || '')
  const [headers, setHeaders] = useState<Record<string, string>>(() => ({
    ...(server?.connection?.headers || {}),
  }))
  const [timeout, setTimeout] = useState(server?.options?.timeout?.toString() || '30000')
  const [autoReconnect, setAutoReconnect] = useState(server?.options?.autoReconnect ?? true)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)
  const [testResult, setTestResult] = useState<{ success: boolean; error?: string } | null>(null)
  const dismissible = !saving && !testing
  const dialogRef = useAccessibleDialog<HTMLDivElement>({
    isOpen: true,
    onClose,
    dismissible,
  })

  const formValues = {
    name,
    description,
    transport,
    enabled,
    command,
    args,
    url,
    headers,
    timeout,
    autoReconnect,
  }
  const isInvalid = !name.trim() || (transport === 'stdio' ? !command.trim() : !url.trim())

  const handleSave = async () => {
    setSaving(true)
    setSaveError(null)
    try {
      await onSave(buildServerConfig(formValues))
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : 'Failed to save the MCP server.')
    } finally {
      setSaving(false)
    }
  }

  const handleTest = async () => {
    setTesting(true)
    setTestResult(null)
    try {
      const result = await onTest(buildTestServerConfig(server?.id, formValues))
      setTestResult(result)
    } catch (err) {
      setTestResult({
        success: false,
        error: err instanceof Error ? err.message : 'Test failed',
      })
    } finally {
      setTesting(false)
    }
  }

  return (
    <div
      className={styles.dialogOverlay}
      role="presentation"
      onMouseDown={dismissible ? onClose : undefined}
    >
      <div
        ref={dialogRef}
        className={styles.dialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-busy={saving || testing}
        tabIndex={-1}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className={styles.dialogHeader}>
          <h3 id={titleId}>{server ? 'Edit MCP Server' : 'Add MCP Server'}</h3>
          <button
            type="button"
            className={styles.closeBtn}
            onClick={onClose}
            disabled={!dismissible}
            aria-label="Close MCP server dialog"
          >
            ×
          </button>
        </div>

        <div className={styles.dialogContent}>
          <div className={styles.formGroup}>
            <label htmlFor={nameInputId}>Name *</label>
            <input
              id={nameInputId}
              type="text"
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="My MCP Server"
              data-dialog-initial-focus
            />
          </div>

          <div className={styles.formGroup}>
            <label htmlFor={descriptionInputId}>Description</label>
            <input
              id={descriptionInputId}
              type="text"
              value={description}
              onChange={(event) => setDescription(event.target.value)}
              placeholder="Optional description"
            />
          </div>

          <div className={styles.formGroup}>
            <span id={transportLabelId}>Transport Protocol *</span>
            <div className={styles.radioGroup} role="radiogroup" aria-labelledby={transportLabelId}>
              <label className={styles.radioLabel}>
                <input
                  type="radio"
                  name="transport"
                  value="stdio"
                  checked={transport === 'stdio'}
                  onChange={() => setTransport('stdio')}
                />
                <div>
                  <span>Stdio</span>
                  <small>Local command line (filesystem, git, etc.)</small>
                </div>
              </label>
              <label className={styles.radioLabel}>
                <input
                  type="radio"
                  name="transport"
                  value="streamable-http"
                  checked={transport === 'streamable-http'}
                  onChange={() => setTransport('streamable-http')}
                />
                <div>
                  <span>Streamable HTTP</span>
                  <small>Remote service with streaming support</small>
                </div>
              </label>
            </div>
          </div>

          {transport === 'stdio' ? (
            <>
              <div className={styles.formGroup}>
                <label htmlFor={commandInputId}>Command *</label>
                <input
                  id={commandInputId}
                  type="text"
                  value={command}
                  onChange={(event) => setCommand(event.target.value)}
                  placeholder="npx"
                />
              </div>
              <div className={styles.formGroup} role="group" aria-labelledby={argsLabelId}>
                <span id={argsLabelId}>Arguments</span>
                <StringListEditor
                  value={args}
                  onChange={setArgs}
                  addLabel="Add argument"
                  emptyLabel="No command arguments configured."
                  itemLabel="Argument"
                  placeholder="--flag or value"
                  disabled={!dismissible}
                />
              </div>
            </>
          ) : (
            <>
              <div className={styles.formGroup}>
                <label htmlFor={urlInputId}>URL *</label>
                <input
                  id={urlInputId}
                  type="text"
                  value={url}
                  onChange={(event) => setUrl(event.target.value)}
                  placeholder="https://api.example.com/mcp"
                />
              </div>
              <div className={styles.formGroup} role="group" aria-labelledby={headersLabelId}>
                <span id={headersLabelId}>Request headers</span>
                <KeyValueEditor
                  value={headers}
                  onChange={setHeaders}
                  addLabel="Add header"
                  emptyLabel="No custom request headers configured."
                  keyLabel="Header"
                  keyPlaceholder="Authorization"
                  valueLabel="Header value"
                  valuePlaceholder="Bearer token"
                  disabled={!dismissible}
                />
              </div>
            </>
          )}

          <div className={styles.formGroup}>
            <label htmlFor={timeoutInputId}>Timeout (ms)</label>
            <input
              id={timeoutInputId}
              type="number"
              value={timeout}
              onChange={(event) => setTimeout(event.target.value)}
              placeholder="30000"
            />
          </div>

          <div className={styles.formGroup}>
            <label className={styles.checkboxLabel}>
              <input
                type="checkbox"
                checked={autoReconnect}
                onChange={(event) => setAutoReconnect(event.target.checked)}
              />
              <span>Auto Reconnect</span>
            </label>
          </div>

          <div className={styles.formGroup}>
            <label className={styles.checkboxLabel}>
              <input
                type="checkbox"
                checked={enabled}
                onChange={(event) => setEnabled(event.target.checked)}
              />
              <span>Enabled</span>
            </label>
          </div>

          {testResult && (
            <div
              className={testResult.success ? styles.testSuccess : styles.testError}
              role={testResult.success ? 'status' : 'alert'}
              aria-live="polite"
            >
              {testResult.success ? '✓ Connection successful!' : `✗ ${testResult.error}`}
            </div>
          )}
          {saveError ? (
            <div className={styles.testError} role="alert">
              {saveError}
            </div>
          ) : null}
        </div>

        <div className={styles.dialogFooter}>
          <button
            type="button"
            className={styles.cancelBtn}
            onClick={onClose}
            disabled={!dismissible}
          >
            Cancel
          </button>
          <button
            type="button"
            className={styles.testBtn}
            onClick={handleTest}
            disabled={saving || testing || isInvalid}
          >
            {testing ? 'Testing...' : 'Test Connection'}
          </button>
          <button
            type="button"
            className={styles.saveBtn}
            onClick={handleSave}
            disabled={saving || testing || isInvalid}
          >
            {saving ? 'Saving...' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  )
}
