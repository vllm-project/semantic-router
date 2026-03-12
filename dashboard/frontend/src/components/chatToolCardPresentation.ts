import type { ToolCall } from '../tools'

const TOOL_LABELS: Record<string, string> = {
  claw_create_team: 'Build Team',
  claw_create_worker: 'Hire Talent',
  claw_delete_team: 'Delete Team',
  claw_delete_worker: 'Remove Talent',
  claw_get_team: 'Team Details',
  claw_get_worker: 'Talent Profile',
  claw_list_teams: 'Browse Teams',
  claw_list_workers: 'Browse Talent',
  claw_update_team: 'Update Team',
  claw_update_worker: 'Update Talent',
}

const STATUS_LABELS: Record<ToolCall['status'], string> = {
  pending: 'Queued',
  running: 'Running',
  completed: 'Done',
  failed: 'Failed',
}

function readStringField(args: Record<string, unknown> | null, key: string) {
  const value = args?.[key]
  return typeof value === 'string' ? value.trim() : ''
}

export function getToolDisplayName(toolName: string) {
  return TOOL_LABELS[toolName] || toolName
}

export function getToolStatusLabel(status: ToolCall['status']) {
  return STATUS_LABELS[status]
}

export function getToolSummary(toolName: string, args: Record<string, unknown> | null, isClawTool: boolean) {
  const name = readStringField(args, 'name')
  const role = readStringField(args, 'role')
  const team = readStringField(args, 'team_name') || readStringField(args, 'team')
  const query = readStringField(args, 'query')
  const url = readStringField(args, 'url')

  if (toolName === 'claw_create_worker') {
    return [name, role].filter(Boolean).join(' · ') || 'Preparing a talent profile'
  }

  if (toolName === 'claw_create_team') {
    return team || name || 'Preparing a team hire plan'
  }

  if (query) {
    return `"${query}"`
  }

  if (url) {
    try {
      return new URL(url).hostname
    } catch {
      return url
    }
  }

  if (team) {
    return team
  }

  if (name || role) {
    return [name, role].filter(Boolean).join(' · ')
  }

  return isClawTool ? 'HireClaw control action' : 'Tool execution'
}
