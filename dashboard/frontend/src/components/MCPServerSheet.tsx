import React, { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { MoreHorizontal, Pencil, Play, Plug, Power, Trash2, Unplug, Wrench } from 'lucide-react'

import {
  mcpApi,
  type MCPServerState,
  type MCPToolDefinition,
  type MCPToolResult,
} from '../tools/mcp'
import {
  Alert,
  Badge,
  Button,
  CodeView,
  DetailSheet,
  DetailSheetBody,
  DetailSheetFooter,
  DetailSheetHeader,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  EmptyState,
  Field,
  IconTile,
  KeyValueGrid,
  SectionLabel,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  Textarea,
  cn,
} from '../ui'
import type { UnifiedTool } from './mcpConfigPanelTypes'
import {
  SERVER_STATUS_META,
  TRANSPORT_META,
  serverTarget,
  toUnifiedMCPTool,
  toolIcon,
} from './mcpConfigPanelUtils'

function exampleArguments(tool: MCPToolDefinition | undefined): string {
  const properties = tool?.inputSchema?.properties ?? {}
  const example = Object.fromEntries(
    Object.entries(properties).map(([name, property]) => {
      if (property.default !== undefined) return [name, property.default]
      if (property.enum?.length) return [name, property.enum[0]]
      switch (property.type) {
        case 'number':
        case 'integer':
          return [name, 0]
        case 'boolean':
          return [name, false]
        case 'array':
          return [name, []]
        case 'object':
          return [name, {}]
        default:
          return [name, '']
      }
    }),
  )
  return JSON.stringify(example, null, 2)
}

function TestCall({
  server,
  onConnect,
  disabled,
}: {
  server: MCPServerState
  onConnect: () => void
  disabled: boolean
}) {
  const tools = server.tools ?? []
  const [toolName, setToolName] = useState(tools[0]?.name ?? '')
  const tool = tools.find((item) => item.name === toolName) ?? tools[0]
  const [args, setArgs] = useState(() => exampleArguments(tools[0]))
  const [state, setState] = useState<
    | { status: 'idle' }
    | { status: 'running' }
    | { status: 'done'; result: MCPToolResult }
    | { status: 'invalid'; message: string }
    | { status: 'failed'; message: string }
  >({ status: 'idle' })

  if (server.status !== 'connected') {
    return (
      <Alert
        tone={server.status === 'error' ? 'danger' : 'neutral'}
        title={`${server.config.name} is not connected`}
        actions={
          <Button size="sm" variant="outline" onClick={onConnect} disabled={disabled}>
            <Plug />
            Connect
          </Button>
        }
      >
        Connect the server to list its tools, then run a test call.
      </Alert>
    )
  }
  if (!tools.length) {
    return (
      <EmptyState
        compact
        icon={Wrench}
        tone="slate"
        title="No tools listed"
        description="The server is connected but lists no tools."
      />
    )
  }

  const run = async () => {
    if (!tool) return
    let parsed: unknown
    try {
      parsed = args.trim() ? JSON.parse(args) : {}
    } catch (error) {
      setState({
        status: 'invalid',
        message: `Arguments must be valid JSON. ${(error as Error).message}`,
      })
      return
    }
    setState({ status: 'running' })
    try {
      setState({
        status: 'done',
        result: await mcpApi.executeTool(server.config.id, tool.name, parsed),
      })
    } catch (error) {
      setState({
        status: 'failed',
        message: error instanceof Error ? error.message : String(error),
      })
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <Field label="Tool" htmlFor="mcp-test-tool">
        <Select
          value={tool?.name ?? ''}
          onValueChange={(next) => {
            setToolName(next)
            setArgs(exampleArguments(tools.find((item) => item.name === next)))
            setState({ status: 'idle' })
          }}
        >
          <SelectTrigger id="mcp-test-tool" className="w-full font-mono text-[12.5px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {tools.map((item) => (
              <SelectItem key={item.name} value={item.name} className="font-mono text-[12.5px]">
                {item.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </Field>
      <Field
        label="Arguments"
        htmlFor="mcp-test-args"
        error={state.status === 'invalid' ? state.message : undefined}
        hint={tool?.description}
      >
        <Textarea
          id="mcp-test-args"
          value={args}
          onChange={(event) => {
            setArgs(event.target.value)
            if (state.status === 'invalid') setState({ status: 'idle' })
          }}
          rows={6}
          spellCheck={false}
          aria-invalid={state.status === 'invalid' || undefined}
          className="font-mono text-[12.5px] leading-relaxed"
        />
      </Field>
      <div className="flex items-center gap-3">
        <Button onClick={() => void run()} loading={state.status === 'running'} disabled={disabled}>
          {state.status === 'running' ? null : <Play />}
          Run test call
        </Button>
        <span className="text-xs text-muted-foreground">
          Calls the tool directly; no decision or model is involved.
        </span>
      </div>
      <AnimatePresence mode="wait">
        {state.status === 'failed' ? (
          <motion.div
            key="failed"
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
          >
            <Alert tone="danger" title="The call failed">
              {state.message}
            </Alert>
          </motion.div>
        ) : state.status === 'done' ? (
          <motion.div
            key="done"
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="overflow-hidden rounded-xl border border-border bg-surface-sunken/60 dark:bg-surface-sunken"
          >
            <div className="flex h-9 items-center gap-2 border-b border-border bg-surface-subtle/70 px-3 text-xs">
              <Badge size="sm" tone={state.result.success ? 'success' : 'danger'} dot>
                {state.result.success ? 'OK' : 'Error'}
              </Badge>
              <span className="font-mono text-muted-foreground">tools/call · {tool?.name}</span>
              {state.result.execution_time_ms !== undefined ? (
                <span className="ml-auto font-mono tabular text-subtle-foreground">
                  {state.result.execution_time_ms} ms
                </span>
              ) : null}
            </div>
            <CodeView
              language="json"
              code={JSON.stringify(
                state.result.success
                  ? {
                      result: state.result.result,
                      structured_content: state.result.structured_content,
                    }
                  : { error: state.result.error },
                null,
                2,
              )}
            />
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  )
}

interface MCPServerSheetProps {
  server: MCPServerState | null
  open: boolean
  tab: string
  canManage: boolean
  canRunTools: boolean
  actionLoading: string | null
  onTabChange: (tab: string) => void
  onOpenChange: (open: boolean) => void
  onToggleConnection: (server: MCPServerState) => void
  onToggleEnabled: (server: MCPServerState) => void
  onEdit: (server: MCPServerState) => void
  onDelete: (id: string) => void
  onSelectTool: (tool: UnifiedTool) => void
}

export const MCPServerSheet: React.FC<MCPServerSheetProps> = ({
  server,
  open,
  tab,
  canManage,
  canRunTools,
  actionLoading,
  onTabChange,
  onOpenChange,
  onToggleConnection,
  onToggleEnabled,
  onEdit,
  onDelete,
  onSelectTool,
}) => {
  if (!server) return null
  const transport = TRANSPORT_META[server.config.transport] ?? TRANSPORT_META.stdio
  const status = SERVER_STATUS_META[server.status] ?? SERVER_STATUS_META.disconnected
  const busy = actionLoading === server.config.id
  const tools = server.tools ?? []
  const headerNames = Object.keys(server.config.connection.headers ?? {})

  return (
    <DetailSheet open={open} onOpenChange={onOpenChange}>
      <DetailSheetHeader
        icon={<IconTile icon={transport.icon} tone="slate" size="lg" />}
        title={<span className="font-mono">{server.config.name}</span>}
        badges={
          <>
            <Badge tone="outline">{transport.label}</Badge>
            <Badge tone={status.tone === 'neutral' ? 'neutral' : status.tone} dot>
              {status.label}
            </Badge>
            {!server.config.enabled ? <Badge tone="outline">off</Badge> : null}
          </>
        }
        description={
          <span className="min-w-0 truncate font-mono">
            {serverTarget(server.config) || server.config.description}
          </span>
        }
        actions={
          <>
            <Button
              variant="outline"
              size="sm"
              onClick={() => onToggleConnection(server)}
              disabled={!canManage || busy}
              loading={busy}
            >
              {busy ? null : server.status === 'connected' ? <Unplug /> : <Plug />}
              {server.status === 'connected' ? 'Disconnect' : 'Connect'}
            </Button>
            {canManage ? (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" size="icon-sm" aria-label="More actions">
                    <MoreHorizontal />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="w-44">
                  <DropdownMenuItem onSelect={() => onEdit(server)} disabled={busy}>
                    <Pencil />
                    Edit connection
                  </DropdownMenuItem>
                  <DropdownMenuItem onSelect={() => onToggleEnabled(server)} disabled={busy}>
                    <Power />
                    {server.config.enabled ? 'Turn off' : 'Turn on'}
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem
                    variant="destructive"
                    onSelect={() => onDelete(server.config.id)}
                    disabled={busy}
                  >
                    <Trash2 />
                    Delete
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            ) : null}
          </>
        }
      />
      <Tabs value={tab} onValueChange={onTabChange} className="min-h-0 flex-1 gap-0">
        <TabsList className="px-5">
          <TabsTrigger value="tools">
            Tools
            <Badge size="sm" tone="neutral">
              {tools.length}
            </Badge>
          </TabsTrigger>
          {canRunTools ? <TabsTrigger value="test">Test call</TabsTrigger> : null}
          <TabsTrigger value="connection">Connection</TabsTrigger>
        </TabsList>
        <DetailSheetBody>
          <TabsContent value="tools" className="flex flex-col gap-3 px-5 py-5">
            {server.error ? (
              <Alert tone="danger" title="The last connection attempt failed">
                {server.error}
              </Alert>
            ) : null}
            {server.status !== 'connected' ? (
              <p className="text-[13px] text-muted-foreground">
                Connect this server to discover its tools.
              </p>
            ) : tools.length ? (
              <ul className="overflow-hidden rounded-xl border border-border">
                {tools.map((tool) => {
                  const Icon = toolIcon(tool.name)
                  const parameters = Object.keys(tool.inputSchema?.properties ?? {}).length
                  return (
                    <li key={tool.name} className="border-b border-border last:border-0">
                      <button
                        type="button"
                        onClick={() => onSelectTool(toUnifiedMCPTool(server, tool))}
                        aria-label={`View details for ${tool.name}`}
                        className="flex w-full items-center gap-3 px-3.5 py-3 text-left outline-none transition-colors hover:bg-surface-subtle focus-visible:bg-surface-subtle"
                      >
                        <IconTile icon={Icon} tone="slate" size="sm" />
                        <span className="flex min-w-0 flex-1 flex-col">
                          <span className="font-mono text-[12.5px] font-medium">{tool.name}</span>
                          <span className="truncate text-xs text-muted-foreground">
                            {tool.description || 'No description'}
                          </span>
                        </span>
                        <span className="shrink-0 text-[11px] tabular text-subtle-foreground">
                          {parameters} {parameters === 1 ? 'param' : 'params'}
                        </span>
                      </button>
                    </li>
                  )
                })}
              </ul>
            ) : (
              <p className="text-[13px] text-muted-foreground">The server lists no tools.</p>
            )}
          </TabsContent>
          {canRunTools ? (
            <TabsContent value="test" className="px-5 py-5">
              <TestCall
                key={`${server.config.id}-${server.status}`}
                server={server}
                onConnect={() => onToggleConnection(server)}
                disabled={busy}
              />
            </TabsContent>
          ) : null}
          <TabsContent value="connection" className="flex flex-col gap-5 px-5 py-5">
            <KeyValueGrid
              items={[
                {
                  label: 'Transport',
                  value: `${transport.label} · ${transport.hint.replace(/\.$/, '').toLowerCase()}`,
                },
                { label: 'State', value: server.config.enabled ? 'On' : 'Off' },
                {
                  label: server.config.transport === 'stdio' ? 'Command' : 'URL',
                  value: serverTarget(server.config) || '—',
                  mono: true,
                  span: true,
                },
                {
                  label: 'Timeout',
                  value: `${server.config.options?.timeout ?? 30000} ms`,
                  mono: true,
                },
                {
                  label: 'Connected since',
                  value: server.connected_at ? new Date(server.connected_at).toLocaleString() : '—',
                },
                ...(server.config.description
                  ? [{ label: 'Description', value: server.config.description, span: true }]
                  : []),
              ]}
            />
            {server.config.transport === 'streamable-http' ? (
              <section className="flex flex-col gap-2">
                <SectionLabel>Request headers</SectionLabel>
                {headerNames.length ? (
                  <div className="overflow-hidden rounded-lg border border-border">
                    {headerNames.map((name) => (
                      <div
                        key={name}
                        className="flex items-center gap-3 border-b border-border px-3 py-2 font-mono text-[12px] last:border-0"
                      >
                        <span className="w-44 shrink-0 truncate text-muted-foreground">{name}</span>
                        <span className="truncate text-subtle-foreground">••••••••</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-[13px] text-muted-foreground">No custom request headers.</p>
                )}
              </section>
            ) : null}
          </TabsContent>
        </DetailSheetBody>
      </Tabs>
      <DetailSheetFooter>
        <span className={cn('text-xs text-muted-foreground')}>
          {server.status === 'connected'
            ? `${tools.length} ${tools.length === 1 ? 'tool' : 'tools'} available to decisions`
            : 'Tools appear once the server connects'}
        </span>
        <Button variant="ghost" size="sm" className="ml-auto" onClick={() => onOpenChange(false)}>
          Close
        </Button>
      </DetailSheetFooter>
    </DetailSheet>
  )
}
