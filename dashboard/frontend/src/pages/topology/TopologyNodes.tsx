import * as React from 'react'
import { Handle, Position, getBezierPath, type EdgeProps, type NodeProps } from 'reactflow'
import { Blend, CircleAlert, Layers, ListTree, Split } from 'lucide-react'
import { IconTile, ProviderLogo, Tooltip, cn, toneFill, type Tone } from '../../ui'
import { signalTypeIcon } from '../configPageSignalCatalog'
import type { TopoEdgeData, TopoNodeData } from './topologyGraph'

const activeRing: Record<Tone, string> = {
  blue: 'border-chart-1 ring-4 ring-chart-1/15',
  gold: 'border-chart-2 ring-4 ring-chart-2/20',
  violet: 'border-chart-3 ring-4 ring-chart-3/15',
  teal: 'border-chart-4 ring-4 ring-chart-4/18',
  rose: 'border-chart-5 ring-4 ring-chart-5/15',
  green: 'border-chart-6 ring-4 ring-chart-6/18',
  orange: 'border-chart-7 ring-4 ring-chart-7/15',
  slate: 'border-chart-8 ring-4 ring-chart-8/15',
}

function Card({
  data,
  children,
  dashed,
  vertical,
}: {
  data: TopoNodeData
  children: React.ReactNode
  dashed?: boolean
  vertical?: boolean
}) {
  return (
    <div
      style={{ width: data.width, height: data.height }}
      className={cn(
        'relative flex rounded-xl border bg-card text-left shadow-xs transition-[opacity,box-shadow,border-color] duration-300 hover:border-border-strong hover:shadow-sm',
        vertical ? 'flex-col justify-center gap-1.5 px-3' : 'items-center gap-2.5 px-2.5',
        dashed ? 'border-dashed border-border-strong bg-surface-subtle' : 'border-border',
        data.state === 'active' && cn(activeRing[data.tone], 'shadow-md hover:border-inherit'),
        data.state === 'secondary' && 'border-dashed opacity-90',
        data.state === 'dimmed' && 'opacity-30 shadow-none',
      )}
    >
      <Handle type="target" position={Position.Left} isConnectable={false} />
      {children}
      <Handle type="source" position={Position.Right} isConnectable={false} />
    </div>
  )
}

function Text({
  label,
  sublabel,
  compact,
  mono,
}: {
  label: React.ReactNode
  sublabel?: React.ReactNode
  compact: boolean
  mono?: boolean
}) {
  return (
    <span className="flex min-w-0 flex-1 flex-col">
      <span
        className={cn(
          'truncate text-[13.5px] font-medium leading-5 tracking-[-0.005em]',
          mono && 'font-mono text-[12.5px]',
        )}
      >
        {label}
      </span>
      {!compact && sublabel ? (
        <span className="truncate text-xs leading-4 text-subtle-foreground">{sublabel}</span>
      ) : null}
    </span>
  )
}

function Warning({ text }: { text?: string }) {
  if (!text) return null
  return (
    <Tooltip content={text}>
      <CircleAlert className="size-3.5 shrink-0 text-warning" aria-label={text} />
    </Tooltip>
  )
}

function LaneNode({ data }: NodeProps<TopoNodeData>) {
  return (
    <div
      style={{ width: data.width }}
      className="pointer-events-none flex items-baseline gap-2 px-1"
    >
      <span
        className="size-1.5 shrink-0 self-center rounded-full"
        style={{ background: toneFill[data.tone] }}
      />
      <span className="text-xs font-medium uppercase tracking-[0.06em] text-muted-foreground">
        {data.label}
      </span>
      <span className="font-mono text-xs tabular text-subtle-foreground">{data.count}</span>
      {data.sublabel ? (
        <span className="ml-auto truncate text-[11.5px] text-subtle-foreground">
          {data.sublabel}
        </span>
      ) : null}
    </div>
  )
}

function EntryNode({ data }: NodeProps<TopoNodeData>) {
  const slash = data.label.indexOf('/')
  const namespace = slash > 0 ? data.label.slice(0, slash + 1) : ''
  const name = slash > 0 ? data.label.slice(slash + 1) : data.label
  return (
    <Card data={data} vertical>
      <span className="flex items-center gap-2">
        <IconTile icon={Layers} tone="blue" size="xs" />
        <span className="truncate font-mono text-xs text-subtle-foreground">
          {namespace || 'model'}
        </span>
      </span>
      <span className="truncate font-mono text-[12.5px] font-semibold leading-4">{name}</span>
      {data.compact ? null : (
        <span className="truncate text-xs leading-4 text-muted-foreground">{data.sublabel}</span>
      )}
    </Card>
  )
}

function SignalNode({ data }: NodeProps<TopoNodeData>) {
  const Icon = signalTypeIcon(data.signalType ?? '')
  return (
    <Card data={data}>
      <IconTile icon={Icon} tone="gold" size="xs" />
      <Text
        label={data.label}
        sublabel={<span className="font-mono">{data.sublabel}</span>}
        compact={data.compact}
      />
      <Warning text={data.warning} />
      <span className="rounded-full bg-secondary px-1.5 font-mono text-[11px] tabular text-muted-foreground">
        {data.count}
      </span>
    </Card>
  )
}

function ProjectionNode({ data }: NodeProps<TopoNodeData>) {
  return (
    <Card data={data}>
      <IconTile icon={Blend} tone="teal" size="xs" />
      <Text
        label={data.label}
        sublabel={<span className="font-mono">{data.sublabel}</span>}
        compact={data.compact}
        mono
      />
    </Card>
  )
}

function DecisionNode({ data }: NodeProps<TopoNodeData>) {
  return (
    <Card data={data}>
      <IconTile icon={Split} tone={data.tone} size="xs" />
      <Text label={data.label} sublabel={data.sublabel} compact={data.compact} mono />
      <Warning text={data.warning} />
      {data.compact && data.meta ? (
        <span className="font-mono text-[11px] tabular text-subtle-foreground">{data.meta}</span>
      ) : null}
    </Card>
  )
}

function MoreNode({ data }: NodeProps<TopoNodeData>) {
  return (
    <Card data={data} dashed={data.state !== 'active'}>
      <IconTile icon={data.state === 'active' ? Split : ListTree} tone="violet" size="xs" />
      <Text label={data.label} sublabel={data.sublabel} compact={data.compact} />
      {data.compact && data.meta ? (
        <span className="font-mono text-[11px] tabular text-subtle-foreground">{data.meta}</span>
      ) : null}
    </Card>
  )
}

function ModelNode({ data }: NodeProps<TopoNodeData>) {
  return (
    <Card data={data}>
      <ProviderLogo provider={data.provider ?? 'vllm'} size={data.compact ? 14 : 16} framed />
      <Text label={data.label} sublabel={data.sublabel} compact={data.compact} mono />
    </Card>
  )
}

export const nodeTypes = {
  lane: LaneNode,
  entry: EntryNode,
  signal: SignalNode,
  projection: ProjectionNode,
  decision: DecisionNode,
  more: MoreNode,
  model: ModelNode,
}

function FlowEdge({
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
}: EdgeProps<TopoEdgeData>) {
  const [path] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
    curvature: 0.32,
  })
  const state = data?.state ?? 'idle'
  const color = toneFill[data?.tone ?? 'slate']
  return (
    <g className="pointer-events-none">
      <path
        d={path}
        fill="none"
        stroke={color}
        strokeWidth={state === 'active' ? 2.2 : 1.3}
        strokeOpacity={
          state === 'active'
            ? 0.95
            : state === 'secondary'
              ? 0.55
              : state === 'dimmed'
                ? 0.07
                : 0.32
        }
        strokeDasharray={state === 'secondary' ? '4 5' : undefined}
        style={{ transition: 'stroke-opacity 300ms, stroke-width 300ms' }}
      />
      {state === 'active' ? (
        <path
          d={path}
          fill="none"
          stroke="var(--surface)"
          strokeWidth={2.4}
          strokeDasharray="3 13"
          strokeLinecap="round"
          className="animate-flow-dash"
        />
      ) : null}
    </g>
  )
}

export const edgeTypes = { flow: FlowEdge }

/** React Flow restyled with design tokens so the canvas follows light and dark mode. */
export const topologyCss = `
.vsr-topology .react-flow__pane { cursor: grab; }
.vsr-topology .react-flow__pane.dragging { cursor: grabbing; }
.vsr-topology .react-flow__node { cursor: pointer; }
.vsr-topology .react-flow__node-lane { cursor: default; }
.vsr-topology .react-flow__node:focus-visible > div { outline: 2px solid var(--ring); outline-offset: 2px; border-radius: 12px; }
.vsr-topology .react-flow__handle { opacity: 0; width: 6px; height: 6px; min-width: 0; min-height: 0; border: 0; background: var(--border-strong); pointer-events: none; }
.vsr-topology .react-flow__controls { box-shadow: var(--shadow-md); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; background: var(--popover); }
.vsr-topology .react-flow__controls-button { width: 30px; height: 30px; padding: 0; background: var(--popover); border: 0; border-bottom: 1px solid var(--border); color: var(--muted-foreground); }
.vsr-topology .react-flow__controls-button:last-child { border-bottom: 0; }
.vsr-topology .react-flow__controls-button:hover { background: var(--accent); color: var(--foreground); }
.vsr-topology .react-flow__controls-button svg { fill: currentColor; max-width: 12px; max-height: 12px; }
.vsr-topology .react-flow__controls-button svg.lucide { fill: none; max-width: 14px; max-height: 14px; }
.vsr-topology .react-flow__minimap { background: var(--popover); border: 1px solid var(--border); border-radius: 12px; box-shadow: var(--shadow-md); overflow: hidden; }
.vsr-topology .react-flow__minimap-mask { fill: var(--surface-sunken); fill-opacity: 0.6; stroke: var(--border-strong); stroke-width: 1px; }
.vsr-topology .react-flow__minimap-node { stroke: none; }
.vsr-topology .react-flow__panel { margin: 12px; }
.vsr-topology .react-flow__attribution { display: none; }
`
