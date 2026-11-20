import { GitCompareArrows, Link2 } from 'lucide-react'
import {
  Button,
  DetailSheet,
  DetailSheetBody,
  DetailSheetFooter,
  DetailSheetHeader,
  IconTile,
  Skeleton,
  toast,
} from '../../ui'
import RunComparison from './RunComparison'
import { Notice } from './benchUi'
import type { Run } from './types'

/** Paired comparison of selected runs; its selection lives in the URL so it can be reopened. */
export default function CompareSheet({
  open,
  onClose,
  runs,
  loaded,
  readError,
  actorID,
}: {
  open: boolean
  onClose: () => void
  runs: Run[]
  loaded: boolean
  readError?: string
  actorID: string
}) {
  return (
    <DetailSheet open={open} onOpenChange={(value) => !value && onClose()} size="xl">
      <DetailSheetHeader
        icon={<IconTile icon={GitCompareArrows} tone="violet" size="lg" />}
        title="Compare runs"
        description="Paired results on the same frozen questions: a single-model reference against your recipe versions."
      />
      <DetailSheetBody className="px-5 py-4">
        {!open ? null : loaded ? (
          <RunComparison key={actorID} runs={runs} />
        ) : readError ? (
          <Notice tone="danger" as="p" role="status" className="my-0">
            Comparisons need the saved run inventory. It reconnects automatically.
          </Notice>
        ) : (
          <div role="status" className="flex flex-col gap-2.5">
            <span className="text-xs text-muted-foreground">Loading evaluation runs…</span>
            <Skeleton className="h-24 w-full" />
          </div>
        )}
      </DetailSheetBody>
      <DetailSheetFooter>
        <span className="text-xs text-muted-foreground">
          Costs are frozen per-token prices applied to recorded usage.
        </span>
        <div className="ml-auto flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              void navigator.clipboard?.writeText(window.location.href).catch(() => undefined)
              toast.success('Comparison link copied', {
                description: 'The link reopens these runs with the same selection.',
              })
            }}
          >
            <Link2 aria-hidden="true" />
            Copy link
          </Button>
          <Button variant="ghost" size="sm" onClick={onClose}>
            Close
          </Button>
        </div>
      </DetailSheetFooter>
    </DetailSheet>
  )
}
