import * as React from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowRight, CornerDownLeft, Moon, Sun } from 'lucide-react'
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
  CommandShortcut,
  Kbd,
} from '../ui'
import { navigation, type NavItem } from './navigation'
import { useShell } from './shell-context'
import { useTheme } from './theme'

export interface CommandEntry {
  id: string
  label: string
  hint?: string
  icon?: React.ReactNode
  keywords?: string[]
  to?: string
  onSelect?: () => void
  shortcut?: string
}

export interface CommandSection {
  heading: string
  entries: CommandEntry[]
}

export function CommandMenu({
  sections = [],
  isItemVisible = () => true,
}: {
  sections?: CommandSection[]
  isItemVisible?: (item: NavItem) => boolean
}) {
  const { commandOpen, setCommandOpen } = useShell()
  const { resolved, toggle } = useTheme()
  const navigate = useNavigate()

  const run = (entry: CommandEntry) => {
    setCommandOpen(false)
    if (entry.onSelect) entry.onSelect()
    else if (entry.to) navigate(entry.to)
  }

  return (
    <CommandDialog open={commandOpen} onOpenChange={setCommandOpen}>
      <CommandInput placeholder="Search pages, recipes, decisions, models…" />
      <CommandList>
        <CommandEmpty>No results. Try a model, decision or page name.</CommandEmpty>
        {navigation.map((group) => {
          const items = group.items.filter(isItemVisible)
          if (items.length === 0) return null
          return (
            <CommandGroup key={group.id} heading={group.label ?? 'Start'}>
              {items.map((item) => (
                <CommandItem
                  key={item.id}
                  value={`${item.label} ${item.description} ${(item.keywords ?? []).join(' ')}`}
                  onSelect={() => run({ id: item.id, label: item.label, to: item.to })}
                >
                  <item.icon />
                  <span>{item.label}</span>
                  <span className="truncate text-xs text-subtle-foreground">
                    {item.description}
                  </span>
                  <ArrowRight className="ml-auto size-3.5 opacity-0 transition-opacity [[data-selected=true]_&]:opacity-100" />
                </CommandItem>
              ))}
            </CommandGroup>
          )
        })}
        {sections.map((section) => (
          <React.Fragment key={section.heading}>
            <CommandSeparator />
            <CommandGroup heading={section.heading}>
              {section.entries.map((entry) => (
                <CommandItem
                  key={entry.id}
                  value={`${entry.label} ${entry.hint ?? ''} ${(entry.keywords ?? []).join(' ')}`}
                  onSelect={() => run(entry)}
                >
                  {entry.icon}
                  <span className="truncate">{entry.label}</span>
                  {entry.hint ? (
                    <span className="truncate text-xs text-subtle-foreground">{entry.hint}</span>
                  ) : null}
                  {entry.shortcut ? <CommandShortcut>{entry.shortcut}</CommandShortcut> : null}
                </CommandItem>
              ))}
            </CommandGroup>
          </React.Fragment>
        ))}
        <CommandSeparator />
        <CommandGroup heading="Preferences">
          <CommandItem
            value="toggle theme dark light mode appearance"
            onSelect={() => {
              setCommandOpen(false)
              toggle()
            }}
          >
            {resolved === 'dark' ? <Sun /> : <Moon />}
            Switch to {resolved === 'dark' ? 'light' : 'dark'} mode
            <CommandShortcut>
              <Kbd>⇧</Kbd>
              <Kbd>D</Kbd>
            </CommandShortcut>
          </CommandItem>
        </CommandGroup>
      </CommandList>
      <div className="flex items-center gap-4 border-t border-border bg-surface-subtle/70 px-4 py-2 text-[11px] text-subtle-foreground">
        <span className="flex items-center gap-1.5">
          <Kbd>↑</Kbd>
          <Kbd>↓</Kbd>
          Navigate
        </span>
        <span className="flex items-center gap-1.5">
          <Kbd>
            <CornerDownLeft className="size-3" />
          </Kbd>
          Open
        </span>
        <span className="flex items-center gap-1.5">
          <Kbd>esc</Kbd>
          Close
        </span>
      </div>
    </CommandDialog>
  )
}
