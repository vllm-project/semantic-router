import React, { useState } from 'react'
import type { MCPServerConfig, MCPTransportType } from '../tools/mcp'
import styles from './MCPConfigPanel.module.css'
import {
  buildServerConfig,
  buildTestServerConfig,
  toHeaderLines,
} from './mcpConfigPanelUtils'

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
  const [name, setName] = useState(server?.name || '')
  const [description, setDescription] = useState(server?.description || '')
  const [transport, setTransport] = useState<MCPTransportType>(server?.transport || 'stdio')
  const [enabled, setEnabled] = useState(server?.enabled ?? true)
  const [command, setCommand] = useState(server?.connection?.command || '')
  const [args, setArgs] = useState(server?.connection?.args?.join('\n') || '')
  const [url, setUrl] = useState(server?.connection?.url || '')
  const [headers, setHeaders] = useState(() => toHeaderLines(server?.connection?.headers))
  const [timeout, setTimeout] = useState(server?.options?.timeout?.toString() || '30000')
  const [autoReconnect, setAutoReconnect] = useState(server?.options?.autoReconnect ?? true)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<{ success: boolean; error?: string } | null>(null)

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
  const isInvalid = !name || (transport === 'stdio' ? !command : !url)

  const handleSave = async () => {
    setSaving(true)
    try {
      await onSave(buildServerConfig(formValues))
    } catch (err) {
      console.error('Save failed:', err)
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
    <div className={styles.dialogOverlay} onClick={onClose}>
      <div className={styles.dialog} onClick={event => event.stopPropagation()}>
        <div className={styles.dialogHeader}>
          <h3>{server ? 'Edit MCP Server' : 'Add MCP Server'}</h3>
          <button className={styles.closeBtn} onClick={onClose}>×</button>
        </div>

        <div className={styles.dialogContent}>
          <div className={styles.formGroup}>
            <label>Name *</label>
            <input
              type="text"
              value={name}
              onChange={event => setName(event.target.value)}
              placeholder="My MCP Server"
            />
          </div>

          <div className={styles.formGroup}>
            <label>Description</label>
            <input
              type="text"
              value={description}
              onChange={event => setDescription(event.target.value)}
              placeholder="Optional description"
            />
          </div>

          <div className={styles.formGroup}>
            <label>Transport Protocol *</label>
            <div className={styles.radioGroup}>
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
                <label>Command *</label>
                <input
                  type="text"
                  value={command}
                  onChange={event => setCommand(event.target.value)}
                  placeholder="npx"
                />
              </div>
              <div className={styles.formGroup}>
                <label>Arguments (one per line)</label>
                <textarea
                  value={args}
                  onChange={event => setArgs(event.target.value)}
                  placeholder={"-y\n@modelcontextprotocol/server-filesystem\n/Users/workspace"}
                  rows={4}
                />
              </div>
            </>
          ) : (
            <>
              <div className={styles.formGroup}>
                <label>URL *</label>
                <input
                  type="text"
                  value={url}
                  onChange={event => setUrl(event.target.value)}
                  placeholder="https://api.example.com/mcp"
                />
              </div>
              <div className={styles.formGroup}>
                <label>Headers (for authentication, one per line)</label>
                <textarea
                  value={headers}
                  onChange={event => setHeaders(event.target.value)}
                  placeholder={"Authorization: Bearer your-token\nX-API-Key: your-api-key"}
                  rows={3}
                />
                <small className={styles.fieldHint}>Format: Header-Name: value (one per line)</small>
              </div>
            </>
          )}

          <div className={styles.formGroup}>
            <label>Timeout (ms)</label>
            <input
              type="number"
              value={timeout}
              onChange={event => setTimeout(event.target.value)}
              placeholder="30000"
            />
          </div>

          <div className={styles.formGroup}>
            <label className={styles.checkboxLabel}>
              <input
                type="checkbox"
                checked={autoReconnect}
                onChange={event => setAutoReconnect(event.target.checked)}
              />
              <span>Auto Reconnect</span>
            </label>
          </div>

          <div className={styles.formGroup}>
            <label className={styles.checkboxLabel}>
              <input
                type="checkbox"
                checked={enabled}
                onChange={event => setEnabled(event.target.checked)}
              />
              <span>Enabled</span>
            </label>
          </div>

          {testResult && (
            <div className={testResult.success ? styles.testSuccess : styles.testError}>
              {testResult.success ? '✓ Connection successful!' : `✗ ${testResult.error}`}
            </div>
          )}
        </div>

        <div className={styles.dialogFooter}>
          <button className={styles.cancelBtn} onClick={onClose}>
            Cancel
          </button>
          <button className={styles.testBtn} onClick={handleTest} disabled={testing || isInvalid}>
            {testing ? 'Testing...' : 'Test Connection'}
          </button>
          <button className={styles.saveBtn} onClick={handleSave} disabled={saving || isInvalid}>
            {saving ? 'Saving...' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  )
}
