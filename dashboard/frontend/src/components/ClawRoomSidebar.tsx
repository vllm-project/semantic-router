import { useState, type FormEvent } from 'react'

import styles from './ClawRoomChat.module.css'
import ClawRoomTeamDetailsModal from './ClawRoomTeamDetailsModal'

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
  displayName: string
  isLeader: boolean
  roleText: string
  vibeText: string
}

interface ClawRoomSidebarProps {
  creatingRoom: boolean
  deletingRoomId: string | null
  managementDisabled: boolean
  memberProfiles: MemberProfile[]
  newRoomName: string
  onChangeNewRoomName: (value: string) => void
  onCreateRoom: (event: FormEvent<HTMLFormElement>) => void
  onDeleteRoom: (room: RoomOption) => void
  onSelectRoom: (roomId: string) => void
  onSelectTeam: (teamId: string) => void
  rooms: RoomOption[]
  selectedRoom: RoomOption | null
  selectedRoomId: string
  selectedTeam: TeamOption | null
  selectedTeamId: string
  teamBriefText: string
  teams: TeamOption[]
}

export default function ClawRoomSidebar({
  creatingRoom,
  deletingRoomId,
  managementDisabled,
  memberProfiles,
  newRoomName,
  onChangeNewRoomName,
  onCreateRoom,
  onDeleteRoom,
  onSelectRoom,
  onSelectTeam,
  rooms,
  selectedRoom,
  selectedRoomId,
  selectedTeam,
  selectedTeamId,
  teamBriefText,
  teams,
}: ClawRoomSidebarProps) {
  const [isTeamDetailsOpen, setIsTeamDetailsOpen] = useState(false)
  const memberSubtitle = `${memberProfiles.length} ${memberProfiles.length === 1 ? 'claw' : 'claws'}`

  return (
    <>
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

          <button
            type="button"
            className={styles.teamDetailsButton}
            onClick={() => setIsTeamDetailsOpen(true)}
            disabled={!selectedTeam}
            data-testid="claw-room-team-details-button"
          >
            <span className={styles.teamDetailsButtonHead}>
              <span className={styles.teamDetailsButtonLabel}>Team details</span>
              <span className={styles.teamDetailsButtonIcon} aria-hidden="true">↗</span>
            </span>
            <span className={styles.teamDetailsButtonText}>
              {selectedTeam ? 'Open the team summary, active room, and member roster in a centered dialog.' : 'Select a team first.'}
            </span>
            {selectedTeam ? (
              <span className={styles.teamDetailsButtonMeta}>
                {memberSubtitle} · {selectedRoom ? selectedRoom.name : 'No active room'}
              </span>
            ) : null}
          </button>
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

      </aside>

      <ClawRoomTeamDetailsModal
        isOpen={isTeamDetailsOpen}
        memberProfiles={memberProfiles}
        onClose={() => setIsTeamDetailsOpen(false)}
        selectedRoom={selectedRoom}
        selectedTeam={selectedTeam}
        teamBriefText={teamBriefText}
      />
    </>
  )
}
