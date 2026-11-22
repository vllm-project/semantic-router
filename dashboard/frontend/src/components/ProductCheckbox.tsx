import { forwardRef, type InputHTMLAttributes } from 'react'

import { cn } from '../ui'

type ProductCheckboxProps = Omit<InputHTMLAttributes<HTMLInputElement>, 'type'>

/**
 * Native checkbox drawn like the design-system Checkbox, for callers that rely
 * on input semantics (`event.target.checked`, form submission, `indeterminate`).
 */
const ProductCheckbox = forwardRef<HTMLInputElement, ProductCheckboxProps>(
  ({ className, ...props }, ref) => (
    <input
      {...props}
      ref={ref}
      type="checkbox"
      className={cn(
        'peer relative grid size-4 shrink-0 cursor-pointer appearance-none place-content-center rounded-[5px] border border-border-strong bg-surface shadow-xs outline-none transition-colors',
        "before:h-[7px] before:w-1 before:-translate-y-px before:rotate-45 before:scale-0 before:border-b-[1.75px] before:border-r-[1.75px] before:border-primary-foreground before:transition-transform before:content-['']",
        'checked:border-primary checked:bg-primary checked:before:scale-100',
        'indeterminate:border-primary indeterminate:bg-primary indeterminate:before:h-0 indeterminate:before:w-2 indeterminate:before:translate-y-0 indeterminate:before:rotate-0 indeterminate:before:scale-100 indeterminate:before:border-r-0',
        'focus-visible:ring-[3px] focus-visible:ring-ring/40 disabled:cursor-not-allowed disabled:opacity-50',
        className,
      )}
    />
  ),
)

ProductCheckbox.displayName = 'ProductCheckbox'

export default ProductCheckbox
