import React from 'react'
import { Pencil, X } from 'lucide-react'

import { Button, KeyValueGrid, SectionLabel, cn } from '../ui'

export interface ViewField {
  label: string
  value: React.ReactNode
  fullWidth?: boolean
}

export interface ViewSection {
  title?: string
  fields: ViewField[]
}

export interface ViewPanelAction {
  label: string
  onClick: () => void
  tone?: 'secondary' | 'primary'
  disabled?: boolean
}

interface ViewPanelProps {
  title: string
  sections: ViewSection[]
  onClose?: () => void
  closeLabel?: string
  onEdit?: () => void
  actions?: ViewPanelAction[]
  variant?: 'modal' | 'page'
}

function isEmptyValue(value: React.ReactNode) {
  return value === null || value === undefined || value === false || value === ''
}

/** Read-only label/value sections shared by the detail sheet and standalone panels. */
export function ViewSections({ sections }: { sections: ViewSection[] }) {
  return (
    <div className="flex flex-col divide-y divide-border">
      {sections.map((section, sectionIndex) => (
        <section key={sectionIndex} className="flex flex-col gap-3.5 py-5 first:pt-0 last:pb-0">
          {section.title ? <SectionLabel>{section.title}</SectionLabel> : null}
          <KeyValueGrid
            className="[&_dd]:[overflow-wrap:anywhere] [&_dd]:leading-relaxed"
            items={section.fields.map((field) => ({
              label: field.label,
              span: field.fullWidth,
              value: isEmptyValue(field.value) ? (
                <span className="text-subtle-foreground">—</span>
              ) : (
                field.value
              ),
            }))}
          />
        </section>
      ))}
    </div>
  )
}

/** Footer buttons of a detail view: custom actions, then Edit as the primary action. */
export function ViewPanelActions({
  actions = [],
  onEdit,
  onClose,
  closeLabel,
}: {
  actions?: ViewPanelAction[]
  onEdit?: () => void
  onClose?: () => void
  closeLabel?: string
}) {
  return (
    <>
      {actions.map((action) => (
        <Button
          key={action.label}
          type="button"
          variant={action.tone === 'primary' ? 'default' : 'outline'}
          onClick={action.onClick}
          disabled={action.disabled}
        >
          {action.label}
        </Button>
      ))}
      {onClose && closeLabel ? (
        <Button type="button" variant="ghost" onClick={onClose}>
          {closeLabel}
        </Button>
      ) : null}
      {onEdit ? (
        <Button type="button" onClick={onEdit}>
          <Pencil />
          Edit
        </Button>
      ) : null}
    </>
  )
}

const ViewPanel: React.FC<ViewPanelProps> = ({
  title,
  sections,
  onClose,
  closeLabel = 'Close',
  onEdit,
  actions = [],
  variant = 'modal',
}) => {
  const hasFooter = actions.length > 0 || Boolean(onEdit)

  return (
    <div
      className={cn(
        'flex min-h-0 flex-col overflow-hidden rounded-xl border border-border bg-card text-card-foreground',
        variant === 'page' && 'mx-auto w-full max-w-[1120px] shadow-xs',
      )}
    >
      <div className="flex items-start justify-between gap-4 border-b border-border px-5 py-4">
        <h2 className="min-w-0 text-base font-semibold leading-6 tracking-[-0.01em] [overflow-wrap:anywhere]">
          {title}
        </h2>
        {onClose ? (
          <Button
            type="button"
            variant="ghost"
            size="icon-xs"
            className="-mr-1.5 shrink-0"
            aria-label={closeLabel}
            onClick={onClose}
          >
            <X />
          </Button>
        ) : null}
      </div>

      <div className={cn('min-h-0 flex-1 px-5 py-5', variant === 'modal' && 'overflow-y-auto')}>
        <ViewSections sections={sections} />
      </div>

      {hasFooter ? (
        <div className="flex flex-wrap items-center justify-end gap-2 border-t border-border bg-surface-subtle/70 px-5 py-3">
          <ViewPanelActions actions={actions} onEdit={onEdit} />
        </div>
      ) : null}
    </div>
  )
}

export default ViewPanel
