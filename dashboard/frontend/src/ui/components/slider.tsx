import { Slider as SliderPrimitive } from 'radix-ui'
import { cn } from '../lib/cn'
import { toneFill, type Tone } from '../patterns/data-display'

/** A single-value slider; the filled range takes the entity tone. */
export function Slider({
  value,
  onValueChange,
  min = 0,
  max = 1,
  step = 0.01,
  tone = 'blue',
  disabled,
  className,
  'aria-label': ariaLabel,
}: {
  value: number
  onValueChange: (value: number) => void
  min?: number
  max?: number
  step?: number
  tone?: Tone
  disabled?: boolean
  className?: string
  'aria-label': string
}) {
  return (
    <SliderPrimitive.Root
      value={[value]}
      onValueChange={([next]) => onValueChange(next)}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      className={cn(
        'relative flex h-5 w-full touch-none items-center select-none data-[disabled]:opacity-50',
        className,
      )}
    >
      <SliderPrimitive.Track className="relative h-1.5 grow overflow-hidden rounded-full bg-muted">
        <SliderPrimitive.Range
          className="absolute h-full rounded-full"
          style={{ background: toneFill[tone] }}
        />
      </SliderPrimitive.Track>
      <SliderPrimitive.Thumb
        aria-label={ariaLabel}
        className="block size-4 cursor-grab rounded-full border border-border-strong bg-surface shadow-sm transition-[box-shadow,transform] outline-none hover:shadow-md focus-visible:ring-[3px] focus-visible:ring-ring/40 active:scale-110 active:cursor-grabbing data-[disabled]:cursor-not-allowed"
      />
    </SliderPrimitive.Root>
  )
}
