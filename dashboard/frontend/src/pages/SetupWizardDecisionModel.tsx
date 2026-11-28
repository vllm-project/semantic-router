import { useState } from 'react'
import { motion } from 'framer-motion'
import { BrainCircuit, Check, ChevronDown, Clock, Cpu, TriangleAlert } from 'lucide-react'
import { Badge, Collapsible, CollapsibleContent, CollapsibleTrigger, IconTile, cn } from '../ui'
import type { DecisionModelName, DecisionModelOption } from './decisionModelSupport'
import { formatGiB, memoryShortfall } from './setupWizardSupport'

interface SetupDecisionModelSectionProps {
  value: DecisionModelName
  options: readonly DecisionModelOption[]
  recommended: readonly DecisionModelOption[]
  /** Labels of CPU models the catalog announces but the runtime does not serve yet. */
  upcoming?: readonly string[]
  cpuOnly: boolean
  /** Memory the container runtime gives the deployment, in GiB; null when unknown. */
  memoryGiB?: number | null
  onChange: (name: DecisionModelName) => void
}

function ModelChoice({
  option,
  active,
  recommended,
  cpuOnly,
  shortfall,
  onSelect,
}: {
  option: DecisionModelOption
  active: boolean
  recommended: boolean
  cpuOnly: boolean
  shortfall: string | null
  onSelect: () => void
}) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={active}
      data-choice
      onClick={onSelect}
      className={cn(
        'relative flex items-start gap-3.5 rounded-xl border bg-card p-4 text-left shadow-xs outline-none transition-[border-color,box-shadow] duration-200 focus-visible:ring-[3px] focus-visible:ring-ring/40',
        active
          ? 'border-primary shadow-[0_0_0_3px_var(--ring)]'
          : 'border-border hover:border-border-strong',
      )}
    >
      <span
        className={cn(
          'mt-3 grid size-4 shrink-0 place-items-center rounded-full border transition-colors',
          active ? 'border-primary' : 'border-border-strong',
        )}
      >
        {active ? (
          <motion.span
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: 'spring', bounce: 0.4, duration: 0.3 }}
            className="size-2 rounded-full bg-primary"
          />
        ) : null}
      </span>
      <IconTile icon={BrainCircuit} tone="green" />
      <span className="flex min-w-0 flex-1 flex-col gap-1.5">
        <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <span className="font-mono text-[13.5px] font-semibold">{option.label}</span>
          <span className="text-xs text-subtle-foreground">{option.family}</span>
          {recommended ? (
            <Badge size="sm" tone="brand">
              {cpuOnly ? <Cpu /> : null}
              {cpuOnly ? 'Recommended for CPU' : 'Recommended'}
            </Badge>
          ) : null}
        </span>
        <span className="text-[13px] leading-relaxed text-muted-foreground">{option.summary}</span>
        {shortfall ? (
          <span className="flex items-center gap-1.5 text-xs font-medium text-warning-foreground">
            <TriangleAlert className="size-3.5 shrink-0" aria-hidden />
            {shortfall}
          </span>
        ) : null}
        <span className="flex flex-wrap gap-1 pt-0.5">
          {option.questionTypes.map((type) => (
            <span
              key={type}
              className="inline-flex h-5 items-center rounded-md border border-border bg-surface-subtle px-1.5 text-[11px] font-medium text-muted-foreground"
            >
              {type}
            </span>
          ))}
        </span>
      </span>
      <span className="hidden shrink-0 pt-0.5 text-right text-[11.5px] text-subtle-foreground sm:block">
        {option.hardware}
      </span>
    </button>
  )
}

export function SetupDecisionModelSection({
  value,
  options,
  recommended,
  upcoming = [],
  cpuOnly,
  memoryGiB = null,
  onChange,
}: SetupDecisionModelSectionProps) {
  const recommendedNames = new Set(recommended.map((option) => option.name))
  const others = options.filter((option) => !recommendedNames.has(option.name))
  const selectedIsOther = others.some((option) => option.name === value)
  const [open, setOpen] = useState(selectedIsOther)
  const families = [...new Set(others.map((option) => option.family))]

  return (
    <section className="flex flex-col gap-4" aria-label="Decision model options">
      <p className="text-[13px] leading-relaxed text-muted-foreground">
        {cpuOnly
          ? recommended.length > 1
            ? 'This deployment shows no GPU platform, so the models that run well on CPU come first. '
            : 'This deployment shows no GPU platform, so the recommended model is one that runs well on CPU. '
          : null}
        {memoryGiB !== null
          ? `Models that need more than this host's ${formatGiB(memoryGiB)} of memory are marked. `
          : null}
        You can switch later on the Decision models page.
      </p>
      <div role="radiogroup" aria-label="Decision model" className="flex flex-col gap-2.5">
        {recommended.map((option) => (
          <ModelChoice
            key={option.name}
            option={option}
            active={option.name === value}
            recommended
            cpuOnly={cpuOnly}
            shortfall={memoryShortfall(option, memoryGiB)}
            onSelect={() => onChange(option.name)}
          />
        ))}
        {others.length ? (
          <Collapsible
            open={open || selectedIsOther}
            onOpenChange={setOpen}
            className="rounded-xl border border-border bg-surface-subtle/50"
          >
            <CollapsibleTrigger asChild>
              <button
                type="button"
                className="flex h-11 w-full items-center gap-2 rounded-xl px-4 text-left text-[13px] font-medium outline-none hover:bg-accent/40 focus-visible:ring-[3px] focus-visible:ring-ring/40"
              >
                Other decision models
                <span className="text-xs font-normal text-subtle-foreground">{others.length}</span>
                <ChevronDown
                  className={cn(
                    'ml-auto size-4 text-subtle-foreground transition-transform duration-200',
                    (open || selectedIsOther) && 'rotate-180',
                  )}
                />
              </button>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className="flex flex-col gap-3 border-t border-border p-3">
                {families.map((family) => (
                  <div key={family} className="flex flex-col gap-1">
                    <span className="px-1 text-[11px] font-medium uppercase tracking-[0.06em] text-subtle-foreground">
                      {family}
                    </span>
                    {others
                      .filter((option) => option.family === family)
                      .map((option) => {
                        const active = option.name === value
                        const shortfall = memoryShortfall(option, memoryGiB)
                        return (
                          <button
                            key={option.name}
                            type="button"
                            role="radio"
                            aria-checked={active}
                            onClick={() => onChange(option.name)}
                            className={cn(
                              'flex items-center gap-3 rounded-lg border px-3 py-2 text-left outline-none transition-colors focus-visible:ring-[3px] focus-visible:ring-ring/40',
                              active
                                ? 'border-primary bg-primary-soft/50'
                                : 'border-transparent hover:bg-accent/60',
                            )}
                          >
                            <span
                              className={cn(
                                'grid size-4 shrink-0 place-items-center rounded-full border',
                                active ? 'border-primary' : 'border-border-strong',
                              )}
                            >
                              {active ? (
                                <Check className="size-2.5 text-primary" strokeWidth={4} />
                              ) : null}
                            </span>
                            <span className="min-w-0 flex-1 truncate font-mono text-[12.5px]">
                              {option.label}
                            </span>
                            <span
                              className={cn(
                                'flex shrink-0 items-center gap-1 text-xs',
                                shortfall
                                  ? 'font-medium text-warning-foreground'
                                  : 'text-subtle-foreground',
                              )}
                              title={shortfall ?? undefined}
                            >
                              {shortfall ? (
                                <TriangleAlert className="size-3.5" aria-label={shortfall} />
                              ) : null}
                              {option.hardware}
                            </span>
                          </button>
                        )
                      })}
                  </div>
                ))}
              </div>
            </CollapsibleContent>
          </Collapsible>
        ) : null}
      </div>
      {cpuOnly && upcoming.length ? (
        <p className="flex items-start gap-1.5 text-xs leading-relaxed text-subtle-foreground">
          <Clock className="mt-0.5 size-3.5 shrink-0" aria-hidden />
          <span>
            Coming to the model runtime for CPU:{' '}
            <span className="font-mono text-muted-foreground">{upcoming.join(', ')}</span>. They
            appear here once the runtime serves them.
          </span>
        </p>
      ) : null}
    </section>
  )
}
