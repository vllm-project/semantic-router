import {
  compareByName,
  roleLabel,
  sanitizeLookupKey,
  type MentionOption,
  type TeamProfile,
  type WorkerProfile,
} from './clawRoomChatSupport'

export interface ClawRoomMemberProfile {
  id: string
  displayName: string
  isLeader: boolean
  roleText: string
  vibeText: string
}

interface ClawRoomTeamView {
  memberResumeProfiles: ClawRoomMemberProfile[]
  mentionOptions: MentionOption[]
  teamBriefText: string
  teamWorkers: WorkerProfile[]
  workerLookup: Map<string, WorkerProfile>
}

export function buildClawRoomTeamView(
  selectedTeam: TeamProfile | null,
  workers: readonly WorkerProfile[],
  selectedTeamId: string,
): ClawRoomTeamView {
  const teamWorkers = workers
    .filter((worker) => worker.teamId === selectedTeamId)
    .sort(compareByName)
  let leaderWorker: WorkerProfile | null = null

  if (selectedTeam?.leaderId) {
    leaderWorker = teamWorkers.find((worker) => worker.name === selectedTeam.leaderId) || null
  }
  if (!leaderWorker) {
    leaderWorker = teamWorkers.find((worker) => roleLabel(worker.roleKind) === 'leader') || null
  }

  const workerLookup = new Map<string, WorkerProfile>()
  for (const worker of teamWorkers) {
    const keys = [worker.name, worker.agentName]
    for (const key of keys) {
      const normalized = sanitizeLookupKey(key)
      if (!normalized || workerLookup.has(normalized)) continue
      workerLookup.set(normalized, worker)
    }
  }

  const mentionOptions: MentionOption[] = []
  const seen = new Set<string>()
  const allDescription =
    teamWorkers.length > 0
      ? `All claws in this team (${teamWorkers.length})`
      : 'All claws in this team'
  mentionOptions.push({ token: '@all', description: allDescription })
  seen.add('@all')

  const leaderDescription = leaderWorker
    ? `Leader alias (${leaderWorker.agentName || leaderWorker.name})`
    : 'Leader alias'
  mentionOptions.push({ token: '@leader', description: leaderDescription })
  seen.add('@leader')

  for (const worker of teamWorkers) {
    if (leaderWorker && worker.name === leaderWorker.name) continue
    const token = `@${worker.name}`
    if (seen.has(token)) continue
    seen.add(token)
    mentionOptions.push({
      token,
      description: worker.agentName || roleLabel(worker.roleKind),
    })
  }

  const leaderRoleText = leaderWorker?.agentRole || selectedTeam?.role || 'Team Leader'
  const memberResumeProfiles = teamWorkers.map((worker) => {
    const isLeader =
      selectedTeam?.leaderId === worker.name || roleLabel(worker.roleKind) === 'leader'
    return {
      id: worker.name,
      isLeader,
      displayName: worker.agentName || worker.name,
      roleText: worker.agentRole || (isLeader ? leaderRoleText : 'Team Worker'),
      vibeText: worker.agentVibe || selectedTeam?.vibe || 'Execution-focused',
    }
  })
  memberResumeProfiles.sort((left, right) => {
    if (left.isLeader !== right.isLeader) return left.isLeader ? -1 : 1
    return left.displayName.localeCompare(right.displayName)
  })

  let teamBriefText =
    'Use @leader to delegate work. Workers can also report progress or blockers back via @leader.'
  if (selectedTeam?.description?.trim()) teamBriefText = selectedTeam.description.trim()
  else if (selectedTeam?.principal?.trim()) teamBriefText = selectedTeam.principal.trim()
  else if (leaderWorker?.agentPrinciples?.trim()) {
    teamBriefText = leaderWorker.agentPrinciples.trim()
  }

  return {
    memberResumeProfiles,
    mentionOptions,
    teamBriefText,
    teamWorkers,
    workerLookup,
  }
}
