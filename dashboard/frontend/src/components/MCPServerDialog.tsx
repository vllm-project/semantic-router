import React, { useId, useState } from 'react'
import { CircleCheck, Globe, PlugZap, Terminal, X } from 'lucide-react'

import type { MCPServerConfig, MCPTransportType } from '../tools/mcp'
import {
  Alert,
  Button,
  Dialog,
  DialogBody,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogTitle,
  Field,
  Input,
  InputGroup,
  Label,
  RadioGroup,
  RadioGroupItem,
  Switch,
  cn,
} from '../ui'
import { KeyValueEditor } from './KeyValueEditor'
import { StringListEditor } from './StringListEditor'
import {
  MCP_REDACTED_ARGUMENTS,
  buildServerConfig,
  buildTestServerConfig,
  hiddenArgumentsBlockCommandChange,
  type ServerFormValues,
} from './mcpConfigPanelUtils'

interface MCPServerDialogProps {
  server: MCPServerConfig | null
  onSave: (config: Omit<MCPServerConfig, 'id'>) => Promise<void>
  onTest: (config: MCPServerConfig) => Promise<{ success: boolean; error?: string }>
  onClose: () => void
}

const TRANSPORTS: Array<{
  value: MCPTransportType
  label: string
  hint: string
  icon: typeof Terminal
}> = [
  {
    value: 'stdio',
    label: 'Stdio',
    hint: 'Local command line (filesystem, git, etc.)',
    icon: Terminal,
  },
  {
    value: 'streamable-http',
    label: 'Streamable HTTP',
    hint: 'Remote service with streaming support',
    icon: Globe,
  },
]

interface FormBodyProps {
  server: MCPServerConfig | null
  values: ServerFormValues
  busy: boolean
  onChange: (patch: Partial<ServerFormValues>) => void
}

/** The fields of the server editor; the dialog owns saving and testing. */
export function MCPServerFormBody({ server, values, busy, onChange }: FormBodyProps) {
  const id = useId()
  const hiddenArgumentsNeedResolution = hiddenArgumentsBlockCommandChange(
    server,
    values.command,
    values.preserveArguments,
  )
  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <Field label="Name" htmlFor={`${id}-name`}>
          <Input
            id={`${id}-name`}
            value={values.name}
            onChange={(event) => onChange({ name: event.target.value })}
            placeholder="My MCP server"
            disabled={busy}
            autoFocus
            data-dialog-initial-focus
          />
        </Field>
        <Field label="Description" htmlFor={`${id}-description`} optional>
          <Input
            id={`${id}-description`}
            value={values.description}
            onChange={(event) => onChange({ description: event.target.value })}
            placeholder="What its tools do"
            disabled={busy}
          />
        </Field>
      </div>

      <div className="grid gap-1.5">
        <Label id={`${id}-transport`}>Transport</Label>
        <RadioGroup
          aria-labelledby={`${id}-transport`}
          value={values.transport}
          onValueChange={(next) => onChange({ transport: next as MCPTransportType })}
          disabled={Boolean(server) || busy}
          className="grid grid-cols-1 gap-2 sm:grid-cols-2"
        >
          {TRANSPORTS.map((transport) => {
            const active = values.transport === transport.value
            return (
              <label
                key={transport.value}
                htmlFor={`${id}-${transport.value}`}
                className={cn(
                  'flex cursor-pointer items-start gap-3 rounded-lg border bg-card p-3 transition-[border-color,box-shadow]',
                  active
                    ? 'border-primary shadow-[0_0_0_3px_var(--ring)]'
                    : 'border-border hover:border-border-strong',
                  (server || busy) && 'cursor-not-allowed opacity-70',
                )}
              >
                <RadioGroupItem
                  id={`${id}-${transport.value}`}
                  value={transport.value}
                  className="mt-0.5"
                />
                <span className="flex min-w-0 flex-col">
                  <span className="flex items-center gap-1.5 text-[13px] font-medium">
                    <transport.icon className="size-3.5 text-subtle-foreground" />
                    {transport.label}
                  </span>
                  <span className="text-xs text-muted-foreground">{transport.hint}</span>
                </span>
              </label>
            )
          })}
        </RadioGroup>
        {server ? (
          <p className="text-xs text-muted-foreground">
            Transport cannot be changed after creation. Create a new server instead.
          </p>
        ) : null}
      </div>

      {values.transport === 'stdio' ? (
        <>
          <Field label="Command" htmlFor={`${id}-command`}>
            <InputGroup
              id={`${id}-command`}
              value={values.command}
              onChange={(event) => onChange({ command: event.target.value })}
              placeholder="npx"
              leading={<Terminal />}
              disabled={busy}
              spellCheck={false}
              className="pr-3 font-mono text-[12.5px]"
            />
          </Field>
          <div className="grid gap-1.5" role="group" aria-labelledby={`${id}-args`}>
            <Label id={`${id}-args`}>Arguments</Label>
            {values.preserveArguments ? (
              <div className="flex flex-col gap-2 rounded-lg border border-dashed border-border px-3 py-3">
                <p className="text-[13px] text-muted-foreground">
                  Configured arguments are hidden and will be preserved.
                </p>
                {hiddenArgumentsNeedResolution ? (
                  <p role="alert" className="text-xs text-destructive-foreground">
                    Replace or clear the hidden arguments before changing the command.
                  </p>
                ) : null}
                <Button
                  type="button"
                  variant="outline"
                  size="xs"
                  className="w-fit"
                  onClick={() => onChange({ preserveArguments: false })}
                  disabled={busy}
                >
                  Replace or clear arguments
                </Button>
              </div>
            ) : (
              <StringListEditor
                value={values.args}
                onChange={(args) => onChange({ args })}
                addLabel="Add argument"
                emptyLabel={
                  server
                    ? 'No command arguments configured. Saving now clears stored arguments.'
                    : 'No command arguments configured.'
                }
                itemLabel="Argument"
                placeholder="--flag or value"
                disabled={busy}
              />
            )}
          </div>
        </>
      ) : (
        <>
          <Field label="URL" htmlFor={`${id}-url`}>
            <InputGroup
              id={`${id}-url`}
              value={values.url}
              onChange={(event) => onChange({ url: event.target.value })}
              placeholder="https://api.example.com/mcp"
              leading={<Globe />}
              disabled={busy}
              spellCheck={false}
              className="pr-3 font-mono text-[12.5px]"
            />
          </Field>
          <div className="grid gap-1.5" role="group" aria-labelledby={`${id}-headers`}>
            <Label id={`${id}-headers`}>Request headers</Label>
            <KeyValueEditor
              value={values.headers}
              onChange={(headers) => onChange({ headers })}
              addLabel="Add header"
              emptyLabel="No custom request headers configured."
              keyLabel="Header"
              keyPlaceholder="Authorization"
              valueLabel="Header value"
              valuePlaceholder="Bearer token"
              disabled={busy}
            />
            <p className="text-xs text-muted-foreground">
              Reference secrets as <span className="font-mono">{'${VAR}'}</span> where the server
              resolves them from its environment.
            </p>
          </div>
        </>
      )}

      <div className="grid grid-cols-1 items-end gap-4 sm:grid-cols-2">
        <Field label="Timeout" htmlFor={`${id}-timeout`} hint="Milliseconds a call may take.">
          <InputGroup
            id={`${id}-timeout`}
            type="number"
            value={values.timeout}
            onChange={(event) => onChange({ timeout: event.target.value })}
            placeholder="30000"
            disabled={busy}
            className="font-mono text-[12.5px] tabular"
            trailing={<span className="pr-1 text-xs text-subtle-foreground">ms</span>}
          />
        </Field>
        <label className="flex h-9 cursor-pointer items-center justify-between gap-3 rounded-lg border border-border bg-surface-subtle/60 px-3 sm:mb-[22px]">
          <span className="text-[13px] font-medium">Enabled</span>
          <Switch
            checked={values.enabled}
            onCheckedChange={(enabled) => onChange({ enabled })}
            disabled={busy}
            aria-label="Enabled"
          />
        </label>
      </div>
    </div>
  )
}

function initialValues(server: MCPServerConfig | null): ServerFormValues {
  const hasHiddenArguments =
    server?.connection?.args?.length === 1 && server.connection.args[0] === MCP_REDACTED_ARGUMENTS
  return {
    name: server?.name || '',
    description: server?.description || '',
    transport: server?.transport || 'stdio',
    enabled: server?.enabled ?? true,
    command: server?.connection?.command || '',
    args: hasHiddenArguments ? [] : [...(server?.connection?.args || [])],
    preserveArguments: Boolean(hasHiddenArguments),
    url: server?.connection?.url || '',
    headers: { ...(server?.connection?.headers || {}) },
    timeout: server?.options?.timeout?.toString() || '30000',
  }
}

export const MCPServerDialog: React.FC<MCPServerDialogProps> = ({
  server,
  onSave,
  onTest,
  onClose,
}) => {
  const [values, setValues] = useState<ServerFormValues>(() => initialValues(server))
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)
  const [testResult, setTestResult] = useState<{ success: boolean; error?: string } | null>(null)
  const busy = saving || testing
  const isInvalid =
    !values.name.trim() ||
    (values.transport === 'stdio' ? !values.command.trim() : !values.url.trim()) ||
    hiddenArgumentsBlockCommandChange(server, values.command, values.preserveArguments)

  const handleSave = async () => {
    setSaving(true)
    setSaveError(null)
    try {
      await onSave(buildServerConfig(values))
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
      setTestResult(await onTest(buildTestServerConfig(server?.id, values)))
    } catch (err) {
      setTestResult({ success: false, error: err instanceof Error ? err.message : 'Test failed' })
    } finally {
      setTesting(false)
    }
  }

  return (
    <Dialog open onOpenChange={(open) => !open && !busy && onClose()}>
      <DialogContent size="lg" hideClose aria-busy={busy}>
        <div className="flex items-start gap-3 px-6 pb-2 pt-5">
          <div className="flex min-w-0 flex-1 flex-col gap-1">
            <DialogTitle>{server ? 'Edit MCP server' : 'Add MCP server'}</DialogTitle>
            <DialogDescription>
              {server
                ? 'Changes apply the next time the server connects.'
                : 'The dashboard connects to it and lists its tools for decisions to call.'}
            </DialogDescription>
          </div>
          <DialogClose asChild>
            <Button
              variant="ghost"
              size="icon-sm"
              aria-label="Close MCP server dialog"
              disabled={busy}
            >
              <X />
            </Button>
          </DialogClose>
        </div>
        <DialogBody className="flex max-h-[min(70vh,640px)] flex-col gap-4 pb-5">
          <MCPServerFormBody
            server={server}
            values={values}
            busy={busy}
            onChange={(patch) => setValues((current) => ({ ...current, ...patch }))}
          />
          {testResult ? (
            <div role={testResult.success ? 'status' : 'alert'} aria-live="polite">
              <Alert
                tone={testResult.success ? 'success' : 'danger'}
                title={testResult.success ? 'Connection successful' : 'Connection failed'}
                icon={testResult.success ? <CircleCheck /> : undefined}
              >
                {testResult.success
                  ? 'The server answered and listed its tools.'
                  : testResult.error}
              </Alert>
            </div>
          ) : null}
          {saveError ? (
            <div role="alert">
              <Alert tone="danger" title="The server was not saved">
                {saveError}
              </Alert>
            </div>
          ) : null}
        </DialogBody>
        <DialogFooter>
          <Button variant="ghost" onClick={onClose} disabled={busy}>
            Cancel
          </Button>
          <Button
            variant="outline"
            onClick={() => void handleTest()}
            disabled={busy || isInvalid}
            loading={testing}
          >
            {testing ? null : <PlugZap />}
            {testing ? 'Testing…' : 'Test connection'}
          </Button>
          <Button onClick={() => void handleSave()} disabled={busy || isInvalid} loading={saving}>
            {saving ? 'Saving…' : 'Save'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
