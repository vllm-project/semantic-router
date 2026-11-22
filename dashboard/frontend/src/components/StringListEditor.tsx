import { useId } from 'react'
import { Plus, X } from 'lucide-react'

import { Badge, Button, Input } from '../ui'

interface StructuredEditorStateProps {
  disabled?: boolean
  readOnly?: boolean
}

export interface StringListEditorProps extends StructuredEditorStateProps {
  value: readonly string[]
  onChange: (value: string[]) => void
  addLabel?: string
  emptyLabel?: string
  itemLabel?: string
  maxItems?: number
  placeholder?: string
  validateItem?: (value: string, index: number) => string | null
}

export function StringListEditor({
  value,
  onChange,
  addLabel = 'Add value',
  emptyLabel = 'No values configured.',
  itemLabel = 'Value',
  maxItems,
  placeholder,
  validateItem,
  disabled = false,
  readOnly = false,
}: StringListEditorProps) {
  const editorId = useId()
  const normalizedValue = value.filter((item): item is string => typeof item === 'string')
  const normalizedCounts = normalizedValue.reduce<Map<string, number>>((counts, item) => {
    const normalized = item.trim().toLocaleLowerCase()
    if (normalized) counts.set(normalized, (counts.get(normalized) ?? 0) + 1)
    return counts
  }, new Map())
  const canAdd =
    !disabled && !readOnly && (maxItems === undefined || normalizedValue.length < maxItems)

  if (readOnly) {
    return normalizedValue.length > 0 ? (
      <div className="flex flex-wrap gap-1.5" aria-label={`${itemLabel} values`}>
        {normalizedValue.map((item, index) => (
          <Badge key={`${item}-${index}`} tone="neutral" className="font-mono font-normal">
            {item}
          </Badge>
        ))}
      </div>
    ) : (
      <p className="text-xs text-muted-foreground">{emptyLabel}</p>
    )
  }

  return (
    <div className="flex flex-col gap-2">
      {normalizedValue.length > 0 ? (
        <div className="flex flex-col gap-2">
          {normalizedValue.map((item, index) => {
            const normalized = item.trim().toLocaleLowerCase()
            const isEmpty = !item.trim()
            const isDuplicate = Boolean(normalized && (normalizedCounts.get(normalized) ?? 0) > 1)
            const structuralError = isEmpty
              ? `${itemLabel} cannot be empty.`
              : isDuplicate
                ? `${itemLabel} must be unique.`
                : null
            const error = structuralError ?? validateItem?.(item, index) ?? null
            const inputId = `${editorId}-${index}`
            const errorId = `${inputId}-error`

            return (
              <div key={index} className="grid grid-cols-[minmax(0,1fr)_1.75rem] items-start gap-2">
                <div className="flex min-w-0 flex-col gap-1">
                  <label className="sr-only" htmlFor={inputId}>
                    {itemLabel} {index + 1}
                  </label>
                  <Input
                    id={inputId}
                    className="h-8"
                    value={item}
                    onChange={(event) => {
                      const next = [...normalizedValue]
                      next[index] = event.target.value
                      onChange(next)
                    }}
                    placeholder={placeholder}
                    disabled={disabled}
                    aria-invalid={error ? 'true' : undefined}
                    aria-describedby={error ? errorId : undefined}
                  />
                  {error ? (
                    <span id={errorId} className="text-xs text-destructive-foreground">
                      {error}
                    </span>
                  ) : null}
                </div>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon-xs"
                  className="mt-0.5 hover:bg-destructive-soft hover:text-destructive-foreground"
                  onClick={() =>
                    onChange(normalizedValue.filter((_, itemIndex) => itemIndex !== index))
                  }
                  disabled={disabled}
                  aria-label={`Remove ${itemLabel.toLocaleLowerCase()} ${index + 1}`}
                >
                  <X />
                </Button>
              </div>
            )
          })}
        </div>
      ) : (
        <p className="text-xs text-muted-foreground">{emptyLabel}</p>
      )}
      <Button
        type="button"
        variant="outline"
        size="xs"
        className="self-start"
        onClick={() => onChange([...normalizedValue, ''])}
        disabled={!canAdd}
      >
        <Plus />
        {addLabel}
      </Button>
    </div>
  )
}
