import {
  type FormEvent,
  type ReactNode,
  type RefObject,
  useEffect,
  useId,
  useRef,
  useState,
} from 'react'
import { Info, TriangleAlert } from 'lucide-react'

import {
  Button,
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogTitle,
  Input,
  Label,
  cn,
} from '../ui'
import {
  focusInitialDialogControl,
  keepOpenForToastInteraction,
  restoreDialogFocus,
} from './dialogFocusSupport'

interface ConfirmDialogProps {
  isOpen: boolean
  title: string
  description: ReactNode
  confirmLabel?: string
  cancelLabel?: string
  /** Retained for API compatibility; dialogs no longer render an eyebrow. */
  eyebrow?: string
  details?: ReactNode
  errorMessage?: ReactNode
  errorDetails?: ReactNode
  pending?: boolean
  pendingLabel?: string
  tone?: 'danger' | 'warning' | 'neutral'
  confirmationText?: string
  returnFocusRef?: RefObject<HTMLElement | null>
  returnFocusMode?: 'fallback' | 'always'
  onCancel: () => void
  onConfirm: () => void | Promise<void>
}

const toneSurface: Record<NonNullable<ConfirmDialogProps['tone']>, string> = {
  danger: 'bg-destructive-soft text-destructive-foreground',
  warning: 'bg-warning-soft text-warning-foreground',
  neutral: 'bg-info-soft text-info-foreground',
}

export default function ConfirmDialog({
  isOpen,
  title,
  description,
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  details,
  errorMessage,
  errorDetails,
  pending = false,
  pendingLabel = 'Working…',
  tone = 'danger',
  confirmationText,
  returnFocusRef,
  returnFocusMode = 'fallback',
  onCancel,
  onConfirm,
}: ConfirmDialogProps) {
  const errorId = useId()
  const confirmationId = useId()
  const contentRef = useRef<HTMLDivElement>(null)
  const previousFocusRef = useRef<Element | null>(null)
  const [confirmation, setConfirmation] = useState('')
  const confirmationReady = !confirmationText || confirmation === confirmationText
  const ToneIcon = tone === 'neutral' ? Info : TriangleAlert

  useEffect(() => {
    if (isOpen) setConfirmation('')
  }, [isOpen, confirmationText])

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault()
    if (pending || !confirmationReady) return
    void onConfirm()
  }

  return (
    <Dialog
      open={isOpen}
      onOpenChange={(open) => {
        if (!open && !pending) onCancel()
      }}
    >
      <DialogContent
        ref={contentRef}
        size="sm"
        role="alertdialog"
        aria-modal="true"
        aria-describedby={errorMessage ? errorId : undefined}
        aria-busy={pending}
        hideClose
        onOpenAutoFocus={(event) => {
          previousFocusRef.current = document.activeElement
          focusInitialDialogControl(event, contentRef.current)
        }}
        onCloseAutoFocus={(event) =>
          restoreDialogFocus(
            event,
            previousFocusRef.current,
            returnFocusRef?.current,
            returnFocusMode,
          )
        }
        onInteractOutside={keepOpenForToastInteraction}
      >
        <form onSubmit={handleSubmit} className="flex min-h-0 flex-col">
          <div className="flex min-h-0 flex-col gap-4 overflow-y-auto px-6 pb-5 pt-6">
            <div className="flex items-start gap-3.5">
              <span
                data-tone-icon={tone === 'neutral' ? 'info' : 'alert'}
                className={cn(
                  'grid size-9 shrink-0 place-items-center rounded-full',
                  toneSurface[tone],
                )}
              >
                <ToneIcon className="size-[18px]" strokeWidth={2} />
              </span>
              <div className="flex min-w-0 flex-1 flex-col gap-1.5 pt-0.5">
                <DialogTitle className="leading-6 [overflow-wrap:anywhere]">{title}</DialogTitle>
                <DialogDescription asChild>
                  <div className="leading-relaxed [overflow-wrap:anywhere]">{description}</div>
                </DialogDescription>
              </div>
            </div>

            {details ? (
              <div className="rounded-lg border border-border bg-surface-subtle px-3.5 py-3 text-[13px] text-muted-foreground">
                {details}
              </div>
            ) : null}

            {errorMessage ? (
              <div
                id={errorId}
                role="alert"
                className="flex flex-col gap-1.5 rounded-lg border border-destructive/25 bg-destructive-soft px-3.5 py-3 text-[13px] text-foreground [&_details]:text-xs [&_details]:text-muted-foreground [&_summary]:cursor-pointer"
              >
                <div>{errorMessage}</div>
                {errorDetails}
              </div>
            ) : null}

            {confirmationText ? (
              <div className="grid gap-2">
                <Label
                  htmlFor={confirmationId}
                  className="inline font-normal leading-5 text-muted-foreground"
                >
                  Enter{' '}
                  <strong className="font-mono font-semibold text-foreground">
                    {confirmationText}
                  </strong>{' '}
                  to confirm.
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
              data-dialog-initial-focus={!confirmationText ? true : undefined}
            >
              {cancelLabel}
            </Button>
            <Button
              type="submit"
              variant={tone === 'neutral' ? 'default' : 'destructive'}
              disabled={pending || !confirmationReady}
              loading={pending}
            >
              {pending ? pendingLabel : confirmLabel}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
