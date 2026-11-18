import { ChevronLeft, ChevronRight } from 'lucide-react'
import { Button } from '../../ui'
import { number } from './model'

export default function BenchPagination({
  label,
  total,
  page,
  pageSize,
  onChange,
}: {
  label: string
  total: number
  page: number
  pageSize: number
  onChange: (page: number) => void
}) {
  if (total <= pageSize) return null
  const pages = Math.ceil(total / pageSize)
  const current = Math.min(page, pages - 1)
  return (
    <nav
      className="my-3 flex flex-wrap items-center justify-between gap-3 text-xs text-muted-foreground"
      aria-label={`${label} pages`}
    >
      <span role="status" className="tabular-nums">
        {number(current * pageSize + 1)}–{number(Math.min((current + 1) * pageSize, total))} of{' '}
        {number(total)} · Page {current + 1} of {pages}
      </span>
      <div className="flex items-center gap-1.5">
        <Button
          variant="outline"
          size="xs"
          disabled={current === 0}
          onClick={() => onChange(current - 1)}
        >
          <ChevronLeft aria-hidden="true" /> Previous {label.toLowerCase()}
        </Button>
        <Button
          variant="outline"
          size="xs"
          disabled={current + 1 >= pages}
          onClick={() => onChange(current + 1)}
        >
          Next {label.toLowerCase()} <ChevronRight aria-hidden="true" />
        </Button>
      </div>
    </nav>
  )
}
