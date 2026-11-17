import * as React from 'react'
import { Checkbox as CheckboxPrimitive, Label as LabelPrimitive, RadioGroup as RadioPrimitive, Switch as SwitchPrimitive } from 'radix-ui'
import { Check, Minus } from 'lucide-react'
import { cn } from '../lib/cn'

export const Label = React.forwardRef<
  React.ElementRef<typeof LabelPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof LabelPrimitive.Root>
>(({ className, ...props }, ref) => (
  <LabelPrimitive.Root
    ref={ref}
    data-slot="label"
    className={cn(
      'flex select-none items-center gap-2 text-[13px] font-medium leading-none text-foreground peer-disabled:cursor-not-allowed peer-disabled:opacity-50',
      className,
    )}
    {...props}
  />
))
Label.displayName = 'Label'

export const Checkbox = React.forwardRef<
  React.ElementRef<typeof CheckboxPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof CheckboxPrimitive.Root>
>(({ className, ...props }, ref) => (
  <CheckboxPrimitive.Root
    ref={ref}
    data-slot="checkbox"
    className={cn(
      'peer grid size-4 shrink-0 place-items-center rounded-[5px] border border-border-strong bg-surface shadow-xs outline-none transition-colors',
      'focus-visible:ring-[3px] focus-visible:ring-ring/40 disabled:cursor-not-allowed disabled:opacity-50',
      'data-[state=checked]:border-primary data-[state=checked]:bg-primary data-[state=checked]:text-primary-foreground',
      'data-[state=indeterminate]:border-primary data-[state=indeterminate]:bg-primary data-[state=indeterminate]:text-primary-foreground',
      className,
    )}
    {...props}
  >
    <CheckboxPrimitive.Indicator className="grid place-items-center">
      {props.checked === 'indeterminate' ? (
        <Minus className="size-3" strokeWidth={3} />
      ) : (
        <Check className="size-3" strokeWidth={3} />
      )}
    </CheckboxPrimitive.Indicator>
  </CheckboxPrimitive.Root>
))
Checkbox.displayName = 'Checkbox'

export const Switch = React.forwardRef<
  React.ElementRef<typeof SwitchPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof SwitchPrimitive.Root> & { size?: 'sm' | 'default' }
>(({ className, size = 'default', ...props }, ref) => (
  <SwitchPrimitive.Root
    ref={ref}
    data-slot="switch"
    className={cn(
      'peer inline-flex shrink-0 items-center rounded-full border border-transparent shadow-xs outline-none transition-colors',
      'focus-visible:ring-[3px] focus-visible:ring-ring/40 disabled:cursor-not-allowed disabled:opacity-50',
      'data-[state=checked]:bg-primary data-[state=unchecked]:bg-border-strong',
      size === 'sm' ? 'h-4 w-7' : 'h-5 w-9',
      className,
    )}
    {...props}
  >
    <SwitchPrimitive.Thumb
      className={cn(
        'pointer-events-none block rounded-full bg-white shadow-sm ring-0 transition-transform duration-200 ease-[cubic-bezier(0.16,1,0.3,1)]',
        size === 'sm'
          ? 'size-3 data-[state=checked]:translate-x-3.5 data-[state=unchecked]:translate-x-0.5'
          : 'size-4 data-[state=checked]:translate-x-[18px] data-[state=unchecked]:translate-x-0.5',
      )}
    />
  </SwitchPrimitive.Root>
))
Switch.displayName = 'Switch'

export const RadioGroup = React.forwardRef<
  React.ElementRef<typeof RadioPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof RadioPrimitive.Root>
>(({ className, ...props }, ref) => (
  <RadioPrimitive.Root ref={ref} data-slot="radio-group" className={cn('grid gap-2', className)} {...props} />
))
RadioGroup.displayName = 'RadioGroup'

export const RadioGroupItem = React.forwardRef<
  React.ElementRef<typeof RadioPrimitive.Item>,
  React.ComponentPropsWithoutRef<typeof RadioPrimitive.Item>
>(({ className, ...props }, ref) => (
  <RadioPrimitive.Item
    ref={ref}
    data-slot="radio-group-item"
    className={cn(
      'grid aspect-square size-4 shrink-0 place-items-center rounded-full border border-border-strong bg-surface shadow-xs outline-none transition-colors',
      'focus-visible:ring-[3px] focus-visible:ring-ring/40 data-[state=checked]:border-primary',
      className,
    )}
    {...props}
  >
    <RadioPrimitive.Indicator className="size-2 rounded-full bg-primary" />
  </RadioPrimitive.Item>
))
RadioGroupItem.displayName = 'RadioGroupItem'

export function Field({
  label,
  hint,
  error,
  htmlFor,
  children,
  className,
  optional,
}: {
  label: React.ReactNode
  hint?: React.ReactNode
  error?: React.ReactNode
  htmlFor?: string
  optional?: boolean
  children: React.ReactNode
  className?: string
}) {
  return (
    <div className={cn('grid content-start gap-1.5', className)}>
      <Label htmlFor={htmlFor}>
        {label}
        {optional ? <span className="font-normal text-subtle-foreground">Optional</span> : null}
      </Label>
      {children}
      {error ? (
        <p className="text-xs text-destructive-foreground">{error}</p>
      ) : hint ? (
        <p className="text-xs text-muted-foreground">{hint}</p>
      ) : null}
    </div>
  )
}
