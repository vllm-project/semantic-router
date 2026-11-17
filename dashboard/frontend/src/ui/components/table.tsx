import * as React from 'react'
import { cn } from '../lib/cn'

export const Table = React.forwardRef<HTMLTableElement, React.HTMLAttributes<HTMLTableElement>>(
  ({ className, ...props }, ref) => (
    <div data-slot="table-container" className="relative w-full overflow-x-auto">
      <table
        ref={ref}
        data-slot="table"
        className={cn('w-full caption-bottom text-[13px]', className)}
        {...props}
      />
    </div>
  ),
)
Table.displayName = 'Table'

export function TableHeader({
  className,
  ...props
}: React.HTMLAttributes<HTMLTableSectionElement>) {
  return (
    <thead
      data-slot="table-header"
      className={cn('[&_tr]:border-b [&_tr]:hover:bg-transparent', className)}
      {...props}
    />
  )
}

export function TableBody({ className, ...props }: React.HTMLAttributes<HTMLTableSectionElement>) {
  return (
    <tbody
      data-slot="table-body"
      className={cn('[&_tr:last-child]:border-0', className)}
      {...props}
    />
  )
}

export const TableRow = React.forwardRef<
  HTMLTableRowElement,
  React.HTMLAttributes<HTMLTableRowElement>
>(({ className, ...props }, ref) => (
  <tr
    ref={ref}
    data-slot="table-row"
    className={cn(
      'border-b border-border transition-colors hover:bg-surface-subtle data-[state=selected]:bg-primary-soft/60',
      className,
    )}
    {...props}
  />
))
TableRow.displayName = 'TableRow'

export function TableHead({ className, ...props }: React.ThHTMLAttributes<HTMLTableCellElement>) {
  return (
    <th
      data-slot="table-head"
      className={cn(
        'h-9 whitespace-nowrap px-3 text-left align-middle text-xs font-medium text-muted-foreground first:pl-4 last:pr-4 [&:has([role=checkbox])]:w-10',
        className,
      )}
      {...props}
    />
  )
}

export function TableCell({ className, ...props }: React.TdHTMLAttributes<HTMLTableCellElement>) {
  return (
    <td
      data-slot="table-cell"
      className={cn('h-12 whitespace-nowrap px-3 align-middle first:pl-4 last:pr-4', className)}
      {...props}
    />
  )
}
