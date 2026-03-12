import type { FormEvent } from 'react'

import styles from './ClawRoomChat.module.css'

interface TeamOption {
  id: string
  name: string
  role?: string
  vibe?: string
}

interface RoomOption {
  id: string
  name: string
}

interface MemberProfile {
  id: string
  alias: string
  displayName: string
  isLeader: boolean
  principlesText: string
  roleText: string
  vibeText: string
}

interface ClawRoomSidebarProps {
  creatingRoom: boolean
  deletingRoomId: string | null
  managementDisabled: boolean
  memberProfiles: MemberProfile[]
  mentionHints: string[]
  newRoomName: string
  onChangeNewRoomName: (value: string) => void
  onCreateRoom: (event: FormEvent<HTMLFormElement>) => void
  onDeleteRoom: (room: RoomOption) => void
  onInsertMention: (alias: string) => void
  onSelectRoom: (roomId: string) => void
  onSelectTeam: (teamId: string) => void
  onSetLeader: (workerId: string) => void
  rooms: RoomOption[]
  selectedRoom: RoomOption | null
  selectedRoomId: string
  selectedTeam: TeamOption | null
  selectedTeamId: string
  settingLeaderId: string | null
  teamBriefText: string
  teams: TeamOption[]
}

const OPENCLAW_LOGO_SRC = '/openclaw.svg'

export default function ClawRoomSidebar({
  creatingRoom,
  deletingRoomId,
  managementDisabled,
  memberProfiles,
  mentionHints,
  newRoomName,
  onChangeNewRoomName,
  onCreateRoom,
  onDeleteRoom,
  onInsertMention,
  onSelectRoom,
  onSelectTeam,
  onSetLeader,
  rooms,
  selectedRoom,
  selectedRoomId,
  selectedTeam,
  selectedTeamId,
  settingLeaderId,
  teamBriefText,
  teams,
}: ClawRoomSidebarProps) {
  const memberSubtitle = `${memberProfiles.length} ${memberProfiles.length === 1 ? 'claw' : 'claws'}${
    memberProfiles.some(profile => profile.isLeader) ? ' · leader first' : ' · no leader set'
  }`

  return (
    <aside className={styles.sidebar} data-testid="claw-room-sidebar">
      <div className={styles.sidebarSection}>
        <div className={styles.selectGroup}>
          <label className={styles.label} htmlFor="claw-team-select">Team</label>
          <select
            id="claw-team-select"
            className={styles.select}
            value={selectedTeamId}
            onChange={event => onSelectTeam(event.target.value)}
          >
            {teams.length === 0 && <option value="">No teams</option>}
            {teams.map(team => (
              <option key={team.id} value={team.id}>
                {team.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className={styles.sidebarSection}>
        <div className={styles.roomsHeader}>
          <div>
            <div className={styles.roomsTitle}>Team</div>
            <div className={styles.roomsSubtitle}>{selectedTeam?.name || 'Select a team'}</div>
          </div>
        </div>

        {selectedTeam ? (
          <div className={styles.metaCard}>
            {(selectedTeam.role || selectedTeam.vibe) && (
              <div className={styles.metaInline}>
                {selectedTeam.role ? <span className={styles.metaPill}>{selectedTeam.role}</span> : null}
                {selectedTeam.vibe ? <span className={styles.metaPill}>{selectedTeam.vibe}</span> : null}
              </div>
            )}
            <div className={styles.metaBrief}>{teamBriefText}</div>
            <div className={styles.metaSubtle}>
              {selectedRoom ? (
                <>
                  Active room <code>{selectedRoom.name}</code>
                </>
              ) : (
                'Create or select a room to start.'
              )}
            </div>
          </div>
        ) : (
          <div className={styles.sidebarEmpty}>Select a team to browse rooms and members.</div>
        )}
      </div>

      <div className={styles.sidebarSection}>
        <div className={styles.roomsHeader}>
          <div>
            <div className={styles.roomsTitle}>Rooms</div>
            <div className={styles.roomsSubtitle}>{selectedTeam?.name || 'Select a team'}</div>
          </div>
        </div>

        <form className={styles.createRoomForm} onSubmit={event => onCreateRoom(event)}>
          <input
            type="text"
            className={styles.createRoomInput}
            value={newRoomName}
            onChange={event => onChangeNewRoomName(event.target.value)}
            placeholder="New room name (optional)"
            disabled={managementDisabled || !selectedTeamId || creatingRoom}
          />
          <button
            type="submit"
            className={styles.createRoomButton}
            disabled={managementDisabled || !selectedTeamId || creatingRoom}
          >
            {creatingRoom ? 'Creating...' : 'Create'}
          </button>
        </form>

        <div className={styles.roomList}>
          {rooms.length === 0 ? (
            <div className={styles.sidebarEmpty}>No room yet.</div>
          ) : (
            rooms.map(room => {
              const active = room.id === selectedRoomId
              return (
                <div
                  key={room.id}
                  className={`${styles.roomItem} ${active ? styles.roomItemActive : ''}`}
                  onClick={() => onSelectRoom(room.id)}
                  onKeyDown={event => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault()
                      onSelectRoom(room.id)
                    }
                  }}
                  role="button"
                  tabIndex={0}
                >
                  <div className={styles.roomItemBody}>
                    <span className={styles.roomName}>{room.name}</span>
                    <span className={styles.roomId}>{room.id}</span>
                  </div>
                  <button
                    type="button"
                    className={styles.roomDeleteButton}
                    onClick={event => {
                      event.stopPropagation()
                      onDeleteRoom(room)
                    }}
                    disabled={managementDisabled || deletingRoomId === room.id}
                    title="Delete room"
                    aria-label="Delete room"
                  >
                    {deletingRoomId === room.id ? '…' : '✕'}
                  </button>
                </div>
              )
            })
          )}
        </div>
      </div>

      <div className={styles.sidebarSection}>
        <div className={styles.memberQueueHeading}>
          <span className={styles.memberQueueTitle}>Members</span>
          <span className={styles.memberQueueSubtitle}>{memberSubtitle}</span>
        </div>

        {memberProfiles.length === 0 ? (
          <div className={styles.memberQueueEmpty}>No workers in this team yet</div>
        ) : (
          <div className={styles.memberResumeList}>
            {memberProfiles.map(profile => (
              <article
                key={profile.id}
                className={`${styles.memberResumeItem} ${profile.isLeader ? styles.memberResumeItemLeader : ''}`}
              >
                <div className={styles.memberResumeHead}>
                  <span className={styles.memberResumeIdentity}>
                    <img
                      src={OPENCLAW_LOGO_SRC}
                      alt={profile.isLeader ? 'leader logo' : 'worker logo'}
                      className={styles.metaLogo}
                    />
                    <span className={styles.memberResumeName}>{profile.displayName}</span>
                  </span>
                  <span className={profile.isLeader ? styles.memberResumeRoleLeader : styles.memberResumeRoleWorker}>
                    {profile.isLeader ? 'LEADER' : 'WORKER'}
                  </span>
                </div>
                <div className={styles.resumeAliasRow}>
                  <button
                    type="button"
                    className={styles.quickMentionButton}
                    onClick={() => onInsertMention(profile.alias)}
                  >
                    {profile.alias}
                  </button>
                </div>
                <div className={`${styles.memberResumeFactGrid} ${styles.memberResumeFactGridCompact}`}>
                  <div className={styles.memberResumeFact}>
                    <span className={styles.memberResumeFactLabel}>Role</span>
                    <span className={styles.memberResumeFactValue}>{profile.roleText}</span>
                  </div>
                  <div className={styles.memberResumeFact}>
                    <span className={styles.memberResumeFactLabel}>Style</span>
                    <span className={styles.memberResumeFactValue}>{profile.vibeText}</span>
                  </div>
                </div>
                {profile.principlesText ? (
                  <div className={styles.memberResumeNarrative}>{profile.principlesText}</div>
                ) : null}
                {!profile.isLeader ? (
                  <button
                    type="button"
                    className={styles.memberPromoteButton}
                    onClick={() => onSetLeader(profile.id)}
                    disabled={managementDisabled || settingLeaderId === profile.id}
                  >
                    {settingLeaderId === profile.id ? 'Setting…' : 'Set as leader'}
                  </button>
                ) : null}
              </article>
            ))}
          </div>
        )}
      </div>

      <div className={styles.sidebarSection}>
        <div className={styles.teamGuide}>
          Collaboration tip: start with <code>@leader</code> for delegation.
        </div>
        <div className={styles.mentionsHint}>
          Mention hints: {mentionHints.join(' ')}
        </div>
      </div>
    </aside>
  )
}
