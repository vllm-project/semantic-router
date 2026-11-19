import { useEffect, useId, useRef, useState } from 'react'
import { TriangleAlert } from 'lucide-react'

import {
  Button,
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogTitle,
  Input,
  Label,
  useRetainedValue,
} from '../ui'
import { focusInitialDialogControl } from '../components/dialogFocusSupport'
import { modelDeleteConfirmation } from './configPageModelInventory'

interface ModelDeleteDialogProps {
  modelNames: string[]
  pending: boolean
  onCancel: () => void
  onConfirm: () => void
}

/** Confirms deleting backend models; several at once need a typed confirmation. */
export default function ModelDeleteDialog({
  modelNames,
  pending,
  onCancel,
  onConfirm,
}: ModelDeleteDialogProps) {
  const confirmationId = useId()
  const contentRef = useRef<HTMLDivElement>(null)
  const [confirmation, setConfirmation] = useState('')
  const isOpen = modelNames.length > 0
  // Keep the names while the dialog animates out after a delete clears them.
  const names = useRetainedValue(isOpen ? modelNames : null) ?? modelNames
  const requiredConfirmation = modelDeleteConfirmation(names.length)
  const confirmationReady = names.length === 1 || confirmation === requiredConfirmation
  // Typed input is deliberate work; only the explicit Cancel button discards it.
  const dismissible = !pending && confirmation.length === 0

  useEffect(() => {
    setConfirmation('')
  }, [modelNames])

  return (
    <Dialog
      open={isOpen}
      onOpenChange={(open) => {
        if (!open && dismissible) onCancel()
      }}
    >
      <DialogContent
        ref={contentRef}
        size="sm"
        role="alertdialog"
        aria-modal="true"
        aria-busy={pending}
        hideClose
        onOpenAutoFocus={(event) => focusInitialDialogControl(event, contentRef.current)}
        onPointerDownOutside={(event) => {
          if (!dismissible) event.preventDefault()
        }}
      >
        <form
          className="flex min-h-0 flex-col"
          onSubmit={(event) => {
            event.preventDefault()
            if (!pending && confirmationReady) onConfirm()
          }}
        >
          <div className="flex min-h-0 flex-col gap-4 overflow-y-auto px-6 pb-5 pt-6">
            <div className="flex items-start gap-3.5">
              <span className="grid size-9 shrink-0 place-items-center rounded-full bg-destructive-soft text-destructive-foreground">
                <TriangleAlert className="size-[18px]" strokeWidth={2} />
              </span>
              <div className="flex min-w-0 flex-1 flex-col gap-1.5 pt-0.5">
                <DialogTitle className="leading-6">
                  Delete {names.length} {names.length === 1 ? 'model' : 'models'}?
                </DialogTitle>
                <DialogDescription className="leading-relaxed">
                  The selected models are not defaults and are not referenced by routing decisions.
                  This operation writes the configuration once and cannot be undone from the
                  dashboard.
                </DialogDescription>
              </div>
            </div>

            <div className="flex flex-wrap gap-1.5 rounded-lg border border-border bg-surface-subtle px-3 py-2.5">
              {names.slice(0, 6).map((name) => (
                <code
                  key={name}
                  className="max-w-full truncate rounded-[5px] border border-border bg-surface px-1.5 py-0.5 font-mono text-[12px]"
                >
                  {name}
                </code>
              ))}
              {names.length > 6 ? (
                <span className="self-center text-xs text-muted-foreground">
                  +{names.length - 6} more
                </span>
              ) : null}
            </div>

            {names.length > 1 ? (
              <div className="grid gap-2">
                <Label
                  htmlFor={confirmationId}
                  className="inline font-normal leading-5 text-muted-foreground"
                >
                  Type{' '}
                  <strong className="font-mono font-semibold text-foreground">
                    {requiredConfirmation}
                  </strong>{' '}
                  to confirm
                </Label>
                <Input
                  id={confirmationId}
                  value={confirmation}
                  onChange={(event) => setConfirmation(event.target.value)}
                  autoComplete="off"
                  spellCheck={false}
                  className="font-mono"
                  disabled={pending}
                  data-dialog-initial-focus
                />
              </div>
            ) : null}
          </div>

          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={onCancel}
              disabled={pending}
              data-dialog-initial-focus={names.length === 1 ? true : undefined}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              variant="destructive"
              disabled={pending || !confirmationReady}
              loading={pending}
            >
              {pending ? 'Deleting…' : `Delete ${names.length === 1 ? 'model' : 'models'}`}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
