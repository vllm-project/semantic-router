import { useId, useState } from 'react'
import { Plus, X } from 'lucide-react'

import { Button, Input } from '../ui'

interface StructuredEditorStateProps {
  disabled?: boolean
  readOnly?: boolean
}

export interface KeyValueEditorProps extends StructuredEditorStateProps {
  value: Readonly<Record<string, string>>
  onChange: (value: Record<string, string>) => void
  addLabel?: string
  emptyLabel?: string
  keyLabel?: string
  keyPlaceholder?: string
  valueLabel?: string
  valuePlaceholder?: string
}

function nextAvailableKey(value: Readonly<Record<string, string>>): string {
  let index = 1
  while (value[`key_${index}`] !== undefined) index += 1
  return `key_${index}`
}

export function KeyValueEditor({
  value,
  onChange,
  addLabel = 'Add entry',
  emptyLabel = 'No entries configured.',
  keyLabel = 'Key',
  keyPlaceholder,
  valueLabel = 'Value',
  valuePlaceholder,
  disabled = false,
  readOnly = false,
}: KeyValueEditorProps) {
  const editorId = useId()
  const [keyErrors, setKeyErrors] = useState<Record<string, string>>({})
  const entries = Object.entries(value).filter(
    (entry): entry is [string, string] => typeof entry[1] === 'string',
  )

  const renameKey = (currentKey: string, nextKey: string) => {
    const trimmedKey = nextKey.trim()
    if (!trimmedKey) {
      setKeyErrors((current) => ({ ...current, [currentKey]: `${keyLabel} cannot be empty.` }))
      return
    }
    if (trimmedKey !== currentKey && value[trimmedKey] !== undefined) {
      setKeyErrors((current) => ({ ...current, [currentKey]: `${keyLabel} must be unique.` }))
      return
    }

    const nextEntries = entries.map(([key, entryValue]) =>
      key === currentKey ? [trimmedKey, entryValue] : [key, entryValue],
    )
    setKeyErrors((current) => {
      const next = { ...current }
      delete next[currentKey]
      return next
    })
    onChange(Object.fromEntries(nextEntries))
  }

  if (readOnly) {
    return entries.length > 0 ? (
      <dl className="divide-y divide-border overflow-hidden rounded-lg border border-border bg-surface">
        {entries.map(([key, entryValue]) => (
          <div
            key={key}
            className="grid grid-cols-[minmax(7rem,max-content)_minmax(0,1fr)] gap-4 px-3 py-2 text-[13px]"
          >
            <dt className="font-mono text-xs leading-5 text-muted-foreground">{key}</dt>
            <dd className="min-w-0 leading-5 text-foreground [overflow-wrap:anywhere]">
              {entryValue}
            </dd>
          </div>
        ))}
      </dl>
    ) : (
      <p className="text-xs text-muted-foreground">{emptyLabel}</p>
    )
  }

  return (
    <div className="flex flex-col gap-2">
      {entries.length > 0 ? (
        <div className="flex flex-col gap-2">
          <div
            aria-hidden="true"
            className="grid grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)_1.75rem] gap-2 text-xs font-medium text-subtle-foreground"
          >
            <span>{keyLabel}</span>
            <span>{valueLabel}</span>
          </div>
          {entries.map(([key, entryValue], index) => {
            const keyId = `${editorId}-key-${index}`
            const valueId = `${editorId}-value-${index}`
            const errorId = `${keyId}-error`
            const keyError = keyErrors[key]

            return (
              <div
                key={index}
                className="grid grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)_1.75rem] items-start gap-2"
              >
                <div className="flex min-w-0 flex-col gap-1">
                  <label className="sr-only" htmlFor={keyId}>
                    {keyLabel}
                  </label>
                  <Input
                    id={keyId}
                    className="h-8 font-mono text-xs"
                    value={key}
                    onChange={(event) => renameKey(key, event.target.value)}
                    placeholder={keyPlaceholder}
                    disabled={disabled}
                    aria-invalid={keyError ? 'true' : undefined}
                    aria-describedby={keyError ? errorId : undefined}
                  />
                  {keyError ? (
                    <span id={errorId} className="text-xs text-destructive-foreground">
                      {keyError}
                    </span>
                  ) : null}
                </div>
                <div className="min-w-0">
                  <label className="sr-only" htmlFor={valueId}>
                    {valueLabel}
                  </label>
                  <Input
                    id={valueId}
                    className="h-8"
                    value={entryValue}
                    onChange={(event) => onChange({ ...value, [key]: event.target.value })}
                    placeholder={valuePlaceholder}
                    disabled={disabled}
                  />
                </div>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon-xs"
                  className="mt-0.5 hover:bg-destructive-soft hover:text-destructive-foreground"
                  onClick={() =>
                    onChange(Object.fromEntries(entries.filter(([entryKey]) => entryKey !== key)))
                  }
                  disabled={disabled}
                  aria-label={`Remove ${keyLabel.toLocaleLowerCase()} ${key}`}
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
        onClick={() => onChange({ ...value, [nextAvailableKey(value)]: '' })}
        disabled={disabled}
      >
        <Plus />
        {addLabel}
      </Button>
    </div>
  )
}
