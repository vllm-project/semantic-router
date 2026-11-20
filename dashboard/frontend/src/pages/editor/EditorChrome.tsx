import { AnimatePresence, motion } from 'framer-motion'
import {
  AlignLeft,
  BookOpenText,
  CircleCheck,
  Code,
  Copy,
  Download,
  Eraser,
  FileCode2,
  Layers,
  LayoutGrid,
  MoreHorizontal,
  OctagonX,
  PanelRight,
  PanelRightClose,
  RefreshCw,
  ShieldCheck,
  TriangleAlert,
  Undo2,
} from 'lucide-react'
import {
  Button,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  Segmented,
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
  Separator,
  StatusDot,
  Tooltip,
  cn,
} from '../../ui'
import type { BuilderRoutingScope } from '../builderPageRoutingScopeSupport'

export type EditorView = 'visual' | 'dsl' | 'yaml'

const viewLabel: Record<EditorView, string> = {
  visual: 'Visual',
  dsl: 'Signal DSL',
  yaml: 'Compiled YAML',
}

export function ScopeSelect({
  scopes,
  value,
  stats,
  onChange,
}: {
  scopes: BuilderRoutingScope[]
  value: string
  stats: (scope: BuilderRoutingScope) => string
  onChange: (value: string) => void
}) {
  const current = scopes.find((scope) => scope.id === value)
  return (
    <Select value={value} onValueChange={onChange} disabled={scopes.length === 0}>
      <SelectTrigger
        size="sm"
        className="min-w-[148px] max-w-[260px] shrink-0 gap-2 xl:min-w-[176px]"
        aria-label="Recipe"
      >
        <Layers className="size-3.5 shrink-0 text-chart-1" />
        <span className="text-subtle-foreground max-xl:sr-only">Recipe</span>
        <SelectValue placeholder="Loading…">
          <span className="truncate font-mono text-[12.5px] font-medium">
            {current ? (current.recipeName ?? current.label) : 'Loading…'}
          </span>
        </SelectValue>
      </SelectTrigger>
      <SelectContent align="start" className="w-80">
        <SelectGroup>
          <SelectLabel>Scopes in this config</SelectLabel>
          {scopes.map((scope) => (
            <SelectItem key={scope.id} value={scope.id} className="py-2">
              <span className="flex flex-col gap-0.5">
                <span
                  className={cn(
                    'text-[12.5px] font-medium',
                    scope.recipeName ? 'font-mono' : 'font-sans',
                  )}
                >
                  {scope.recipeName ?? scope.label}
                </span>
                <span className="text-xs text-subtle-foreground">
                  {stats(scope)}
                  {scope.modelNames.length ? ` · ${scope.modelNames.join(', ')}` : ''}
                </span>
              </span>
            </SelectItem>
          ))}
        </SelectGroup>
      </SelectContent>
    </Select>
  )
}

interface EditorToolbarProps {
  scopes: BuilderRoutingScope[]
  scopeId: string
  scopeStats: (scope: BuilderRoutingScope) => string
  onScopeChange: (scopeId: string) => void
  view: EditorView
  onViewChange: (view: EditorView) => void
  compilerReady: boolean
  hasSource: boolean
  validating: boolean
  formatting: boolean
  unsaved: boolean
  showProblems: boolean
  guideOpen: boolean
  onValidate: () => void
  onFormat: () => void
  onDiscard: () => void
  onToggleProblems: () => void
  onToggleGuide: () => void
  onReloadFromRouter: () => void
  onCopyDsl: () => void
  onDownloadDsl: () => void
  onClear: () => void
}

export function EditorToolbar({
  scopes,
  scopeId,
  scopeStats,
  onScopeChange,
  view,
  onViewChange,
  compilerReady,
  hasSource,
  validating,
  formatting,
  unsaved,
  showProblems,
  guideOpen,
  onValidate,
  onFormat,
  onDiscard,
  onToggleProblems,
  onToggleGuide,
  onReloadFromRouter,
  onCopyDsl,
  onDownloadDsl,
  onClear,
}: EditorToolbarProps) {
  return (
    <div className="flex h-12 shrink-0 items-center gap-1.5 border-b border-border px-2.5 xl:gap-2">
      <ScopeSelect scopes={scopes} value={scopeId} stats={scopeStats} onChange={onScopeChange} />
      <Segmented
        aria-label="Editor view"
        value={view}
        onValueChange={onViewChange}
        options={[
          { value: 'visual', label: 'Visual', icon: <LayoutGrid /> },
          { value: 'dsl', label: 'DSL', icon: <Code /> },
          { value: 'yaml', label: 'YAML', icon: <FileCode2 /> },
        ]}
      />
      <Separator orientation="vertical" className="mx-0.5 h-5 max-md:hidden" />
      <Tooltip content="Run the compiler's checks on the working copy">
        <span className="max-md:hidden">
          <Button
            variant="ghost"
            size="sm"
            onClick={onValidate}
            loading={validating}
            disabled={!compilerReady || !hasSource}
          >
            {validating ? null : <ShieldCheck />}
            <span className="max-xl:sr-only">Validate</span>
          </Button>
        </span>
      </Tooltip>
      <Tooltip content="Canonical formatting · Ctrl+Shift+F in the DSL">
        <span className="max-md:hidden">
          <Button
            variant="ghost"
            size="sm"
            onClick={onFormat}
            loading={formatting}
            disabled={!compilerReady || !hasSource}
          >
            {formatting ? null : <AlignLeft />}
            <span className="max-xl:sr-only">Format</span>
          </Button>
        </span>
      </Tooltip>
      <div className="ml-auto flex items-center gap-1.5">
        <AnimatePresence initial={false} mode="wait">
          {unsaved ? (
            <motion.span
              key="dirty"
              initial={{ opacity: 0, y: 3 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="hidden items-center gap-1.5 whitespace-nowrap pr-1 text-xs text-muted-foreground xl:flex"
            >
              <span className="size-1.5 rounded-full bg-warning" />
              Unsaved changes
            </motion.span>
          ) : hasSource ? (
            <motion.span
              key="clean"
              initial={{ opacity: 0, y: 3 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="hidden items-center gap-1.5 whitespace-nowrap pr-1 text-xs text-muted-foreground xl:flex"
            >
              <CircleCheck className="size-3.5 text-success" />
              Matches the loaded config
            </motion.span>
          ) : null}
        </AnimatePresence>
        {unsaved ? (
          <Button variant="ghost" size="sm" onClick={onDiscard}>
            <Undo2 />
            <span className="max-lg:sr-only">Discard</span>
          </Button>
        ) : null}
        <Tooltip content={guideOpen ? 'Close the DSL guide' : 'DSL language guide'}>
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label="DSL language guide"
            aria-pressed={guideOpen}
            onClick={onToggleGuide}
          >
            <BookOpenText />
          </Button>
        </Tooltip>
        <DropdownMenu>
          <Tooltip content="More actions">
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon-sm" aria-label="More editor actions">
                <MoreHorizontal />
              </Button>
            </DropdownMenuTrigger>
          </Tooltip>
          <DropdownMenuContent align="end" className="w-60">
            <DropdownMenuLabel>Working copy</DropdownMenuLabel>
            <DropdownMenuItem onSelect={onReloadFromRouter} disabled={!compilerReady}>
              <RefreshCw />
              Reload from the Router
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={onCopyDsl} disabled={!hasSource}>
              <Copy />
              Copy DSL
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={onDownloadDsl} disabled={!hasSource}>
              <Download />
              Download DSL
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem onSelect={onClear} className="text-destructive-foreground">
              <Eraser />
              Start from an empty file
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
        <Tooltip content={showProblems ? 'Hide problems' : 'Show problems'}>
          <Button
            variant="ghost"
            size="icon-sm"
            className="max-xl:hidden"
            aria-label="Toggle problems panel"
            aria-pressed={showProblems}
            onClick={onToggleProblems}
          >
            {showProblems ? <PanelRightClose /> : <PanelRight />}
          </Button>
        </Tooltip>
      </div>
    </div>
  )
}

interface EditorStatusBarProps {
  errors: number
  warnings: number
  ready: boolean
  compilerError: string | null
  view: EditorView
  cursor: { line: number; column: number }
  entityCount: number
  lineCount: number
  scopeLabel: string
  unsaved: boolean
  onShowProblems: () => void
}

export function EditorStatusBar({
  errors,
  warnings,
  ready,
  compilerError,
  view,
  cursor,
  entityCount,
  lineCount,
  scopeLabel,
  unsaved,
  onShowProblems,
}: EditorStatusBarProps) {
  const StatusIcon = errors ? OctagonX : warnings ? TriangleAlert : CircleCheck
  return (
    <div className="flex h-7 shrink-0 items-center gap-4 border-t border-border bg-surface-subtle/70 px-3 text-[11.5px] text-muted-foreground">
      <button
        type="button"
        onClick={onShowProblems}
        className="flex items-center gap-1.5 rounded outline-none transition-colors hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/40"
      >
        <StatusIcon
          className={cn(
            'size-3.5',
            errors ? 'text-destructive' : warnings ? 'text-warning' : 'text-success',
          )}
        />
        {errors ? (
          <span>
            {errors} {errors === 1 ? 'error' : 'errors'}
          </span>
        ) : ready ? (
          <span>Valid</span>
        ) : (
          <span>Checking…</span>
        )}
        {warnings ? (
          <span>
            · {warnings} {warnings === 1 ? 'warning' : 'warnings'}
          </span>
        ) : null}
      </button>
      {view === 'dsl' ? (
        <span className="font-mono tabular">
          Ln {cursor.line}, Col {cursor.column}
        </span>
      ) : (
        <span className="tabular">
          {entityCount} {entityCount === 1 ? 'entity' : 'entities'}
        </span>
      )}
      <span className="tabular max-md:hidden">{lineCount} lines</span>
      <span className="max-md:hidden">Spaces: 2</span>
      <span className="ml-auto flex min-w-0 items-center gap-1.5">
        <Layers className="size-3 shrink-0 text-chart-1" />
        <span className="truncate font-mono">{scopeLabel}</span>
      </span>
      <span className="max-sm:hidden">{viewLabel[view]}</span>
      {unsaved ? (
        <span className="flex items-center gap-1.5 max-md:hidden">
          <span className="size-1.5 rounded-full bg-warning" />
          Unsaved
        </span>
      ) : null}
      <span className="flex items-center gap-1.5">
        <StatusDot tone={compilerError ? 'danger' : ready ? 'success' : 'warning'} />
        {compilerError ? 'Compiler unavailable' : ready ? 'Compiler ready' : 'Connecting…'}
      </span>
    </div>
  )
}
