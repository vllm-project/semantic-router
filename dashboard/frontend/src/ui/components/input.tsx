import * as React from 'react'
import { cn } from '../lib/cn'

export const inputBase =
  'flex w-full min-w-0 rounded-md border border-input bg-surface px-3 text-[13px] text-foreground shadow-xs outline-none transition-[border-color,box-shadow] duration-150 placeholder:text-subtle-foreground hover:border-border-strong focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/40 disabled:cursor-not-allowed disabled:opacity-50 aria-invalid:border-destructive aria-invalid:ring-destructive/20'

export const Input = React.forwardRef<
  HTMLInputElement,
  React.InputHTMLAttributes<HTMLInputElement>
>(({ className, type = 'text', ...props }, ref) => (
  <input
    ref={ref}
    type={type}
    data-slot="input"
    className={cn(inputBase, 'h-9', className)}
    {...props}
  />
))
Input.displayName = 'Input'

export interface InputGroupProps extends React.InputHTMLAttributes<HTMLInputElement> {
  leading?: React.ReactNode
  trailing?: React.ReactNode
  containerClassName?: string
}

export const InputGroup = React.forwardRef<HTMLInputElement, InputGroupProps>(
  ({ leading, trailing, className, containerClassName, ...props }, ref) => (
    <div className={cn('relative flex items-center', containerClassName)}>
      {leading ? (
        <span className="pointer-events-none absolute left-2.5 flex text-subtle-foreground [&_svg]:size-4">
          {leading}
        </span>
      ) : null}
      <input
        ref={ref}
        data-slot="input"
        className={cn(inputBase, 'h-9', leading && 'pl-8', trailing && 'pr-14', className)}
        {...props}
      />
      {trailing ? (
        <span className="absolute right-2 flex items-center gap-1">{trailing}</span>
      ) : null}
    </div>
  ),
)
InputGroup.displayName = 'InputGroup'

export const Textarea = React.forwardRef<
  HTMLTextAreaElement,
  React.TextareaHTMLAttributes<HTMLTextAreaElement>
>(({ className, ...props }, ref) => (
  <textarea
    ref={ref}
    data-slot="textarea"
    className={cn(inputBase, 'min-h-20 resize-y py-2 leading-relaxed', className)}
    {...props}
  />
))
Textarea.displayName = 'Textarea'
