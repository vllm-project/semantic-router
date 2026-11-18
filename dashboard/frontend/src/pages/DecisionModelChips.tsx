import { Image, Type, Video } from 'lucide-react'
import { Tooltip, cn } from '../ui'
import { QUESTION_TYPE_ORDER, questionTypeDetail, questionTypeLabel } from './decisionQuestionTypes'

const MODALITY_ICONS: Record<string, typeof Type> = { text: Type, image: Image, video: Video }

/** Every System One question type, struck through when the model does not answer it natively. */
export function CapabilityChips({
  supported,
  className,
}: {
  supported: readonly string[]
  className?: string
}) {
  return (
    <span className={cn('flex flex-wrap gap-1', className)}>
      {QUESTION_TYPE_ORDER.map((type) => {
        const on = supported.includes(type)
        return (
          <Tooltip
            key={type}
            content={on ? questionTypeDetail(type) : `No ${questionTypeLabel(type)} questions`}
          >
            <span
              className={cn(
                'inline-flex h-5 items-center rounded-[5px] border px-1.5 text-[11px] font-medium',
                on
                  ? 'border-chart-3/20 bg-chart-3/10 text-foreground'
                  : 'border-dashed border-border text-subtle-foreground/70 line-through decoration-subtle-foreground/40',
              )}
            >
              {questionTypeLabel(type)}
            </span>
          </Tooltip>
        )
      })}
    </span>
  )
}

export function ModalityIcons({ modalities }: { modalities: readonly string[] }) {
  if (!modalities.length) return null
  return (
    <span
      className="flex items-center gap-1 text-subtle-foreground"
      aria-label={`Reads ${modalities.join(', ')}`}
    >
      {modalities.map((modality) => {
        const Icon = MODALITY_ICONS[modality] ?? Type
        return (
          <Tooltip key={modality} content={`${modality} input`}>
            <span className="grid size-5 place-items-center rounded-[5px] bg-secondary">
              <Icon className="size-3" />
            </span>
          </Tooltip>
        )
      })}
    </span>
  )
}
