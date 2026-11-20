import { useEffect, useState } from 'react'
import { Check, Plus, PlusCircle } from 'lucide-react'
import {
  Button,
  Dialog,
  DialogBody,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  Field,
  Input,
  Popover,
  PopoverContent,
  PopoverTrigger,
  Separator,
  Skeleton,
  cn,
} from '../../ui'
import { Notice } from './benchUi'
import { experimentApi, type Experiment } from './experimentApi'
import type { useExperimentCreation } from './useExperimentCreation'

/** Single-choice filter chip; the list loads when the menu opens. */
export function ExperimentFilter({
  selectedID,
  selectedName,
  canWrite,
  onSelect,
  onCreate,
}: {
  selectedID?: string
  selectedName?: string
  canWrite: boolean
  onSelect: (id?: string) => void
  onCreate: () => void
}) {
  const [open, setOpen] = useState(false)
  const [request, setRequest] = useState<{ cursor: number; sequence: number } | null>(null)
  const [items, setItems] = useState<Experiment[]>([])
  const [next, setNext] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [query, setQuery] = useState('')

  useEffect(() => {
    if (!request) return
    const controller = new AbortController()
    setLoading(true)
    setError('')
    experimentApi
      .list(request.cursor, controller.signal)
      .then((value) => {
        if (controller.signal.aborted) return
        setItems((previous) =>
          request.cursor === 0
            ? value.experiments
            : [
                ...previous,
                ...value.experiments.filter((item) => !previous.some((old) => old.id === item.id)),
              ],
        )
        setNext(value.next_cursor)
      })
      .catch((cause: unknown) => {
        if (!controller.signal.aborted)
          setError(cause instanceof Error ? cause.message : 'Could not read experiments.')
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
  }, [request])

  const visible = items.filter((item) => item.name.toLowerCase().includes(query.toLowerCase()))
  const label = selectedID ? (selectedName ?? 'Selected') : null
  return (
    <Popover
      open={open}
      onOpenChange={(value) => {
        setOpen(value)
        if (value) setRequest((current) => ({ cursor: 0, sequence: (current?.sequence ?? 0) + 1 }))
      }}
    >
      <PopoverTrigger asChild>
        <button
          type="button"
          className={cn(
            'inline-flex h-8 max-w-72 items-center gap-1.5 rounded-md border border-dashed border-border-strong bg-transparent px-2.5 text-[13px] font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-foreground',
            label && 'border-solid bg-surface text-foreground shadow-xs',
          )}
        >
          <PlusCircle className={cn('size-3.5 shrink-0', label && 'rotate-45')} aria-hidden />
          Experiment
          {label ? (
            <>
              <Separator orientation="vertical" className="mx-0.5 h-4" />
              <span className="truncate rounded bg-secondary px-1.5 text-xs">{label}</span>
            </>
          ) : null}
        </button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-72 p-0">
        <div className="border-b border-border p-1.5">
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Filter experiments…"
            aria-label="Filter experiments"
            className="h-8 w-full rounded-md bg-transparent px-2 text-[13px] outline-none placeholder:text-subtle-foreground"
          />
        </div>
        <div className="max-h-72 overflow-y-auto p-1" role="menu" aria-label="Experiments">
          {visible.map((item) => {
            const checked = item.id === selectedID
            return (
              <button
                key={item.id}
                type="button"
                role="menuitemradio"
                aria-checked={checked}
                onClick={() => {
                  onSelect(checked ? undefined : item.id)
                  setOpen(false)
                }}
                className="flex min-h-8 w-full items-center gap-2 rounded-md px-2 py-1 text-left text-[13px] hover:bg-accent"
              >
                <Check
                  aria-hidden
                  className={cn('size-3.5 shrink-0 text-primary', !checked && 'invisible')}
                />
                <span className="min-w-0 flex-1 truncate">{item.name}</span>
                {typeof item.run_count === 'number' ? (
                  <span className="font-mono text-[11px] text-subtle-foreground">
                    {item.run_count}
                  </span>
                ) : null}
              </button>
            )
          })}
          {loading ? (
            <div
              className="flex flex-col gap-1.5 p-2"
              role="status"
              aria-label="Loading experiments"
            >
              <Skeleton className="h-4 w-3/4" />
              <Skeleton className="h-4 w-1/2" />
            </div>
          ) : null}
          {!loading && error ? (
            <p role="alert" className="px-2 py-2 text-xs text-destructive-foreground">
              {error}
            </p>
          ) : null}
          {!loading && !error && items.length === 0 ? (
            <p className="px-2 py-2 text-xs text-muted-foreground">
              No experiments yet. An experiment groups a reference run and the recipe versions you
              test against it.
            </p>
          ) : null}
          {!loading && next !== null ? (
            <button
              type="button"
              onClick={() =>
                setRequest((current) => ({ cursor: next, sequence: (current?.sequence ?? 0) + 1 }))
              }
              className="h-8 w-full rounded-md px-2 text-left text-[13px] text-muted-foreground hover:bg-accent hover:text-foreground"
            >
              Show more experiments
            </button>
          ) : null}
        </div>
        {canWrite || selectedID ? (
          <div className="flex flex-col border-t border-border p-1">
            {canWrite ? (
              <button
                type="button"
                onClick={() => {
                  setOpen(false)
                  onCreate()
                }}
                className="flex h-8 w-full items-center gap-2 rounded-md px-2 text-[13px] hover:bg-accent"
              >
                <Plus className="size-3.5" aria-hidden />
                New experiment…
              </button>
            ) : null}
            {selectedID ? (
              <button
                type="button"
                onClick={() => {
                  onSelect(undefined)
                  setOpen(false)
                }}
                className="h-8 w-full rounded-md px-2 text-left text-[13px] text-muted-foreground hover:bg-accent hover:text-foreground"
              >
                Clear filter
              </button>
            ) : null}
          </div>
        ) : null}
      </PopoverContent>
    </Popover>
  )
}

export function ExperimentCreateDialog({
  open,
  onOpenChange,
  creation,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  creation: ReturnType<typeof useExperimentCreation>
}) {
  const [name, setName] = useState('')
  const { pendingRequest, storageError, submitting, error, notice } = creation
  return (
    <Dialog open={open} onOpenChange={(value) => !submitting && onOpenChange(value)}>
      <DialogContent size="sm">
        <DialogHeader>
          <DialogTitle>New experiment</DialogTitle>
          <DialogDescription>
            An experiment groups a single-model reference and the recipe versions you test against
            it. Creating one does not start an evaluation.
          </DialogDescription>
        </DialogHeader>
        <DialogBody className="flex flex-col gap-3">
          {storageError ? (
            <Notice tone="danger" as="p" role="alert" className="my-0">
              {storageError}
            </Notice>
          ) : pendingRequest ? (
            <Notice tone="warning" className="my-0">
              <strong>Check your previous creation request</strong>
              <p className="font-medium text-foreground">{pendingRequest.name}</p>
              <p>
                The response was not confirmed. Check the same request before creating another
                experiment.
              </p>
            </Notice>
          ) : (
            <Field label="Experiment name" htmlFor="experiment-name">
              <Input
                id="experiment-name"
                value={name}
                maxLength={160}
                autoFocus
                onChange={(event) => setName(event.target.value)}
                placeholder="Balance · reasoning and cost"
              />
            </Field>
          )}
          {error ? (
            <Notice tone="danger" as="p" role="alert" className="my-0">
              {error}
            </Notice>
          ) : null}
          {notice ? (
            <Notice tone="info" as="p" role="status" className="my-0">
              {notice}
            </Notice>
          ) : null}
        </DialogBody>
        <DialogFooter>
          <Button
            variant="ghost"
            size="sm"
            disabled={submitting}
            onClick={() => onOpenChange(false)}
          >
            Cancel
          </Button>
          <Button
            size="sm"
            disabled={submitting || !!storageError || (!pendingRequest && !name.trim())}
            loading={submitting}
            onClick={() =>
              void creation.create(name).then((created) => {
                if (created) {
                  setName('')
                  onOpenChange(false)
                }
              })
            }
          >
            {submitting ? null : <Plus aria-hidden />}
            {pendingRequest ? 'Check creation request' : 'Create experiment'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
