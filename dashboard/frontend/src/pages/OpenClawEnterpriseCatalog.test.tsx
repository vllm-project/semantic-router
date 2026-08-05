import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import ClawRoomSidebar from '../components/ClawRoomSidebar'
import { StatusTab } from './OpenClawStatusTab'
import type { OpenClawStatus, TeamProfile } from './OpenClawPageSupport'
import { TeamTab } from './OpenClawTeamTab'
import { SkillsStep } from './OpenClawWorkerProvisionWizard'

const teams: TeamProfile[] = Array.from({ length: 25 }, (_, index) => ({
  id: `team-${String(index + 1).padStart(2, '0')}`,
  name: `Team ${String(index + 1).padStart(2, '0')}`,
}))

const containers: OpenClawStatus[] = Array.from({ length: 22 }, (_, index) => ({
  running: index % 3 !== 0,
  containerName: `worker-${String(index + 1).padStart(2, '0')}`,
  gatewayUrl: '',
  port: 18_789 + index,
  healthy: index % 3 === 1,
  error: '',
  teamId: teams[index % teams.length].id,
  teamName: teams[index % teams.length].name,
  agentName: `Agent ${index + 1}`,
}))

describe('OpenClaw enterprise catalogs', () => {
  it('bounds team and container views while reporting the full client collection', () => {
    const teamMarkup = renderToStaticMarkup(
      createElement(TeamTab, {
        teams,
        teamsLoading: false,
        containers,
        onTeamsUpdated: vi.fn(),
        readOnly: true,
      }),
    )
    expect(teamMarkup).toContain('1–12 of 25 teams')
    expect(teamMarkup).toContain('Team 12')
    expect(teamMarkup).not.toContain('Team 13')

    const statusMarkup = renderToStaticMarkup(
      createElement(StatusTab, {
        containers,
        statusLoading: false,
        onRefresh: vi.fn(),
        readOnly: true,
      }),
    )
    expect(statusMarkup).toContain('1–15 of 22 containers')
    expect(statusMarkup.match(/containerTableName/g)).toHaveLength(15)
    expect(statusMarkup).not.toContain('>worker-03<')
  })

  it('uses checkbox-backed structured controls for worker skills', () => {
    const markup = renderToStaticMarkup(
      createElement(SkillsStep, {
        skills: [
          {
            id: 'routing',
            name: 'Semantic routing',
            description: 'Select the best model path.',
            emoji: '↗',
            category: 'Routing',
            builtin: true,
          },
        ],
        skillsLoading: false,
        skillsError: '',
        selectedSkills: ['routing'],
        toggleSkill: vi.fn(),
        onRetry: vi.fn(),
      }),
    )

    expect(markup).toContain('type="checkbox"')
    expect(markup).toContain('checked=""')
    expect(markup).toContain('1 skill selected')
  })

  it('bounds room navigation to eight rows with explicit count and paging', () => {
    const rooms = Array.from({ length: 18 }, (_, index) => ({
      id: `room-${String(index + 1).padStart(2, '0')}`,
      name: `Room ${String(index + 1).padStart(2, '0')}`,
    }))
    const markup = renderToStaticMarkup(
      createElement(ClawRoomSidebar, {
        creatingRoom: false,
        deletingRoomId: null,
        managementDisabled: true,
        memberProfiles: [],
        newRoomName: '',
        onChangeNewRoomName: vi.fn(),
        onCreateRoom: vi.fn(),
        onDeleteRoom: vi.fn(),
        onSelectRoom: vi.fn(),
        onSelectTeam: vi.fn(),
        rooms,
        selectedRoom: rooms[0],
        selectedRoomId: rooms[0].id,
        selectedTeam: { id: 'team-01', name: 'Team 01' },
        selectedTeamId: 'team-01',
        teamBriefText: '',
        teams: [{ id: 'team-01', name: 'Team 01' }],
      }),
    )

    expect(markup).toContain('8/page')
    expect(markup).toContain('1–8 of 18')
    expect(markup).toContain('Room 08')
    expect(markup).not.toContain('Room 09')
  })
})
