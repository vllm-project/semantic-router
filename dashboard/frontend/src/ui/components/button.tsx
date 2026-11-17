import * as React from 'react'
import { Slot } from 'radix-ui'
import { cva, type VariantProps } from 'class-variance-authority'
import { Loader2 } from 'lucide-react'
import { cn } from '../lib/cn'

export const buttonVariants = cva(
  [
    'relative inline-flex shrink-0 items-center justify-center gap-1.5 whitespace-nowrap rounded-md text-[13px] font-medium',
    'transition-[color,background-color,border-color,box-shadow,transform] duration-150 ease-out',
    'outline-none focus-visible:ring-[3px] focus-visible:ring-ring focus-visible:outline-none',
    'disabled:pointer-events-none disabled:opacity-50 active:translate-y-px',
    "[&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",
  ],
  {
    variants: {
      variant: {
        default:
          'bg-primary text-primary-foreground shadow-xs hover:bg-primary-hover shadow-[inset_0_1px_0_oklch(1_0_0/0.14)]',
        secondary: 'bg-secondary text-secondary-foreground hover:bg-accent',
        outline:
          'border border-border bg-surface text-foreground shadow-xs hover:bg-accent hover:border-border-strong',
        ghost: 'text-muted-foreground hover:bg-accent hover:text-foreground',
        destructive:
          'bg-destructive text-white shadow-xs hover:bg-destructive/90 focus-visible:ring-destructive/40',
        'destructive-ghost': 'text-destructive-foreground hover:bg-destructive-soft',
        soft: 'bg-primary-soft text-primary-soft-foreground hover:bg-primary-soft/80',
        link: 'h-auto px-0 text-primary underline-offset-4 hover:underline active:translate-y-0',
      },
      size: {
        xs: 'h-7 px-2 text-xs [&_svg:not([class*=size-])]:size-3.5',
        sm: 'h-8 px-2.5',
        default: 'h-9 px-3.5',
        lg: 'h-10 px-4 text-sm',
        xl: 'h-11 px-5 text-sm',
        icon: 'size-9',
        'icon-sm': 'size-8',
        'icon-xs': 'size-7 [&_svg:not([class*=size-])]:size-3.5',
      },
    },
    defaultVariants: { variant: 'default', size: 'default' },
  },
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
  loading?: boolean
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    { className, variant, size, asChild = false, loading = false, children, disabled, ...props },
    ref,
  ) => {
    const Comp = asChild ? Slot.Root : 'button'
    return (
      <Comp
        ref={ref}
        data-slot="button"
        className={cn(buttonVariants({ variant, size }), className)}
        disabled={disabled || loading}
        {...props}
      >
        {asChild ? (
          children
        ) : (
          <>
            {loading ? <Loader2 className="animate-spin" /> : null}
            {children}
          </>
        )}
      </Comp>
    )
  },
)
Button.displayName = 'Button'
