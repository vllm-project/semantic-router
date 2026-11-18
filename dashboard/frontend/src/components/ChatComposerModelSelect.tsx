import { useEffect, useId, useRef, useState, type KeyboardEvent as ReactKeyboardEvent } from 'react'
import { Check, ChevronDown, Layers, Search } from 'lucide-react'

import { cn } from '../ui'
import type { RouterModelOption } from '../utils/routerModelSelection'

interface ChatComposerModelSelectProps {
  disabled?: boolean
  models: RouterModelOption[]
  onChange: (model: string) => void
  value: string
}

/** The public model the playground sends to, from the models the router advertises. */
export default function ChatComposerModelSelect({
  disabled = false,
  models,
  onChange,
  value,
}: ChatComposerModelSelectProps) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const rootRef = useRef<HTMLDivElement>(null)
  const triggerRef = useRef<HTMLButtonElement>(null)
  const optionRefs = useRef<Array<HTMLButtonElement | null>>([])
  const listboxId = useId()
  const filteredModels = models.filter((model) => {
    const normalized = query.trim().toLowerCase()
    return (
      !normalized ||
      model.id.toLowerCase().includes(normalized) ||
      model.recipe?.toLowerCase().includes(normalized) ||
      model.description.toLowerCase().includes(normalized)
    )
  })

  const closeMenu = (restoreFocus = false) => {
    setOpen(false)
    if (restoreFocus) requestAnimationFrame(() => triggerRef.current?.focus())
  }

  const focusOption = (index: number) => {
    if (filteredModels.length === 0) return
    const normalizedIndex = (index + filteredModels.length) % filteredModels.length
    optionRefs.current[normalizedIndex]?.focus()
  }

  const handleOptionKeyDown = (event: ReactKeyboardEvent<HTMLButtonElement>, index: number) => {
    if (event.key === 'ArrowDown') {
      event.preventDefault()
      focusOption(index + 1)
    } else if (event.key === 'ArrowUp') {
      event.preventDefault()
      focusOption(index - 1)
    } else if (event.key === 'Home') {
      event.preventDefault()
      focusOption(0)
    } else if (event.key === 'End') {
      event.preventDefault()
      focusOption(filteredModels.length - 1)
    } else if (event.key === 'Escape') {
      event.preventDefault()
      closeMenu(true)
    }
  }

  useEffect(() => {
    if (!open) return
    const handlePointerDown = (event: MouseEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) closeMenu()
    }
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') closeMenu(true)
    }
    document.addEventListener('mousedown', handlePointerDown)
    document.addEventListener('keydown', handleKeyDown)
    return () => {
      document.removeEventListener('mousedown', handlePointerDown)
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [open])

  return (
    <div ref={rootRef} className="relative min-w-0">
      <button
        ref={triggerRef}
        type="button"
        className="flex h-8 min-w-0 max-w-full items-center gap-2 rounded-md px-2 text-[13px] font-medium outline-none transition-colors hover:bg-accent focus-visible:ring-[3px] focus-visible:ring-ring/40 disabled:pointer-events-none disabled:opacity-50 aria-expanded:bg-accent"
        aria-expanded={open}
        aria-controls={listboxId}
        aria-haspopup="listbox"
        aria-label={`Model: ${value || 'Choose model'}`}
        disabled={disabled || models.length === 0}
        onClick={() => {
          setQuery('')
          if (open) closeMenu()
          else setOpen(true)
        }}
        data-testid="playground-composer-model-select"
        title={value || 'Choose model'}
      >
        <Layers className="size-4 shrink-0 text-chart-1" />
        <span className="truncate font-mono text-[12.5px]">{value || 'Choose model'}</span>
        <ChevronDown
          className={cn(
            'size-3.5 shrink-0 text-subtle-foreground transition-transform duration-200',
            open && 'rotate-180',
          )}
        />
      </button>

      {open ? (
        <div className="absolute left-0 top-[calc(100%+6px)] z-40 flex w-80 max-w-[calc(100vw-2rem)] flex-col overflow-hidden rounded-xl border border-border bg-popover text-popover-foreground shadow-lg animate-in fade-in-0 zoom-in-95 slide-in-from-top-1">
          <div className="flex items-center justify-between gap-2 px-3 pb-1 pt-2.5">
            <span className="text-xs font-medium text-muted-foreground">Choose a model</span>
            <small className="text-[11px] text-subtle-foreground tabular">
              {models.length} models
            </small>
          </div>
          <div className="relative px-2 pb-1.5">
            <Search className="pointer-events-none absolute left-4.5 top-1/2 size-3.5 -translate-y-[60%] text-subtle-foreground" />
            <input
              className="h-8 w-full rounded-md border border-input bg-surface pl-7 pr-2 text-[13px] outline-none placeholder:text-subtle-foreground focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/40"
              type="search"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search models"
              aria-label="Search models"
              role="combobox"
              aria-controls={listboxId}
              aria-expanded="true"
              onKeyDown={(event) => {
                if (event.key === 'ArrowDown') {
                  event.preventDefault()
                  focusOption(0)
                } else if (event.key === 'ArrowUp') {
                  event.preventDefault()
                  focusOption(filteredModels.length - 1)
                }
              }}
              autoFocus
            />
          </div>
          <div
            id={listboxId}
            role="listbox"
            aria-label="Select routing model"
            className="flex max-h-72 flex-col gap-0.5 overflow-y-auto border-t border-border p-1.5"
          >
            {filteredModels.map((model, index) => {
              const active = model.id === value
              return (
                <button
                  ref={(element) => {
                    optionRefs.current[index] = element
                  }}
                  key={model.id}
                  type="button"
                  role="option"
                  aria-selected={active}
                  className={cn(
                    'flex min-h-8 items-center gap-2 rounded-md px-2 py-1.5 text-left outline-none transition-colors hover:bg-accent focus-visible:bg-accent',
                    active && 'bg-primary-soft/70',
                  )}
                  onKeyDown={(event) => handleOptionKeyDown(event, index)}
                  onClick={() => {
                    onChange(model.id)
                    closeMenu(true)
                  }}
                >
                  <Layers className="size-3.5 shrink-0 text-chart-1" />
                  <span className="min-w-0 flex-1 truncate font-mono text-[12.5px]">
                    {model.id}
                  </span>
                  {model.recipe ? (
                    <span
                      data-recipe-chip
                      className="max-w-32 shrink-0 truncate rounded-full border border-border px-1.5 text-[11px] leading-[18px] text-muted-foreground"
                    >
                      {model.recipe}
                    </span>
                  ) : null}
                  {active ? (
                    <Check className="size-3.5 shrink-0 text-primary" strokeWidth={2.6} />
                  ) : null}
                </button>
              )
            })}
            {filteredModels.length === 0 ? (
              <span className="px-2 py-3 text-center text-xs text-muted-foreground">
                No models match “{query}”.
              </span>
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  )
}
