import { Check, ChevronDown } from 'lucide-react'

import {
  Button,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '../ui'
import { formatDateTime } from '../utils/dateTime'

import { HarnessLogo } from './InsightsHarness'
import { runHeadline } from './insightsRunSupport'
import type { SessionScope } from './insightsSessionScope'

/** The sessions a harness id names when callers under different keys chose the same id. */
export function SessionChoice({ scope }: { scope: SessionScope }) {
  if (scope.matches.length < 2) return null
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="xs" className="h-6 px-1.5 text-xs text-muted-foreground">
          {scope.matches.length} sessions use this id
          <ChevronDown />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-80">
        {scope.matches.map((run) => (
          <DropdownMenuItem key={run.session_id} onSelect={() => scope.choose(run.session_id)}>
            <HarnessLogo harness={run.harness} size={14} />
            <span className="min-w-0 flex-1 truncate">{runHeadline(run)}</span>
            <span className="shrink-0 text-xs text-subtle-foreground tabular">
              {formatDateTime(run.started_at)}
            </span>
            {run.session_id === scope.id ? <Check className="size-3.5" /> : null}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
