import { forwardRef, type SelectHTMLAttributes } from 'react'
import { ChevronDown } from 'lucide-react'

import { cn, inputBase } from '../ui'

interface NativeSelectProps extends Omit<SelectHTMLAttributes<HTMLSelectElement>, 'size'> {
  size?: 'sm' | 'default'
  containerClassName?: string
}

/**
 * Design-system styling for a native <select>. Use it where the platform
 * control matters: empty-string options, form semantics, or legacy surfaces
 * whose stacking context would hide a portalled listbox.
 */
const NativeSelect = forwardRef<HTMLSelectElement, NativeSelectProps>(
  ({ className, containerClassName, size = 'default', children, ...props }, ref) => (
    <span className={cn('relative inline-flex w-full min-w-0', containerClassName)}>
      <select
        ref={ref}
        data-slot="native-select"
        className={cn(
          inputBase,
          'cursor-pointer appearance-none pr-8 [&>option]:bg-popover [&>option]:text-popover-foreground',
          size === 'sm' ? 'h-8 text-xs' : 'h-9',
          className,
        )}
        {...props}
      >
        {children}
      </select>
      <ChevronDown
        aria-hidden="true"
        className="pointer-events-none absolute right-2.5 top-1/2 size-4 -translate-y-1/2 text-subtle-foreground"
      />
    </span>
  ),
)

NativeSelect.displayName = 'NativeSelect'

export default NativeSelect
