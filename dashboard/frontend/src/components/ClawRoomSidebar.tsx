import { useDeferredValue, useEffect, useMemo, useState, type FormEvent } from 'react'

import styles from './ClawRoomChat.module.css'
import ClawRoomTeamDetailsModal from './ClawRoomTeamDetailsModal'
import {
  filterAndSortOpenClawRooms,
  getOpenClawPageCount,
  getOpenClawVisibleRange,
  OPENCLAW_ROOMS_PAGE_SIZE,
  paginateOpenClawItems,
  type RoomCatalogFilter,
  type RoomCatalogSort,
} from '../utils/openClawCatalogSupport'

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
  roomsError?: string | null
  roomsLoading?: boolean
  selectedRoom: RoomOption | null
  selectedRoomId: string
  selectedTeam: TeamOption | null
  selectedTeamId: string
  teamBriefText: string
  teams: TeamOption[]
  onRetryRooms?: () => void
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
  roomsError,
  roomsLoading = false,
  selectedRoom,
  selectedRoomId,
  selectedTeam,
  selectedTeamId,
  teamBriefText,
  teams,
  onRetryRooms,
}: ClawRoomSidebarProps) {
  const [isTeamDetailsOpen, setIsTeamDetailsOpen] = useState(false)
  const [roomSearch, setRoomSearch] = useState('')
  const [roomFilter, setRoomFilter] = useState<RoomCatalogFilter>('all')
  const [roomSort, setRoomSort] = useState<RoomCatalogSort>('selected-first')
  const [roomPage, setRoomPage] = useState(1)
  const deferredRoomSearch = useDeferredValue(roomSearch)
  const memberSubtitle = `${memberProfiles.length} ${memberProfiles.length === 1 ? 'claw' : 'claws'}`
  const filteredRooms = useMemo(
    () =>
      filterAndSortOpenClawRooms(rooms, selectedRoomId, deferredRoomSearch, roomFilter, roomSort),
    [deferredRoomSearch, roomFilter, roomSort, rooms, selectedRoomId],
  )
  const pageCount = getOpenClawPageCount(filteredRooms.length, OPENCLAW_ROOMS_PAGE_SIZE)
  const safePage = Math.min(roomPage, pageCount)
  const visibleRooms = paginateOpenClawItems(filteredRooms, safePage, OPENCLAW_ROOMS_PAGE_SIZE)
  const roomRange = getOpenClawVisibleRange(
    filteredRooms.length,
    safePage,
    OPENCLAW_ROOMS_PAGE_SIZE,
  )

  useEffect(() => setRoomPage(1), [deferredRoomSearch, roomFilter, roomSort, selectedTeamId])
  useEffect(() => {
    if (roomPage > pageCount) setRoomPage(pageCount)
  }, [pageCount, roomPage])

  return (
    <>
      <aside className={styles.sidebar} data-testid="claw-room-sidebar">
        <div className={styles.sidebarSection}>
          <div className={styles.selectGroup}>
            <label className={styles.label} htmlFor="claw-team-select">
              Team
            </label>
            <select
              id="claw-team-select"
              className={styles.select}
              value={selectedTeamId}
              onChange={(event) => onSelectTeam(event.target.value)}
            >
              {teams.length === 0 && <option value="">No teams</option>}
              {teams.map((team) => (
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
              <span className={styles.teamDetailsButtonIcon} aria-hidden="true">
                ↗
              </span>
            </span>
            <span className={styles.teamDetailsButtonText}>
              {selectedTeam
                ? 'Open the team summary, active room, and member roster in a centered dialog.'
                : 'Select a team first.'}
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

          <form className={styles.createRoomForm} onSubmit={(event) => onCreateRoom(event)}>
            <input
              type="text"
              className={styles.createRoomInput}
              value={newRoomName}
              onChange={(event) => onChangeNewRoomName(event.target.value)}
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

          <div className={styles.roomCatalogControls}>
            <label>
              <span>Search rooms</span>
              <input
                type="search"
                value={roomSearch}
                onChange={(event) => setRoomSearch(event.target.value)}
                placeholder="Name or ID"
              />
            </label>
            <div className={styles.roomCatalogSelects}>
              <label>
                <span>View</span>
                <select
                  value={roomFilter}
                  onChange={(event) => setRoomFilter(event.target.value as RoomCatalogFilter)}
                >
                  <option value="all">All</option>
                  <option value="selected">Selected</option>
                  <option value="other">Other</option>
                </select>
              </label>
              <label>
                <span>Sort</span>
                <select
                  value={roomSort}
                  onChange={(event) => setRoomSort(event.target.value as RoomCatalogSort)}
                >
                  <option value="selected-first">Selected first</option>
                  <option value="name-asc">Name A–Z</option>
                  <option value="id-asc">ID A–Z</option>
                </select>
              </label>
            </div>
          </div>
          <div className={styles.roomCatalogMeta} aria-live="polite">
            <span>
              {filteredRooms.length} of {rooms.length} rooms
            </span>
            <span>{OPENCLAW_ROOMS_PAGE_SIZE}/page</span>
          </div>

          {roomsError ? (
            <div className={styles.roomCatalogError} role="alert">
              <span>{roomsError}</span>
              {onRetryRooms ? (
                <button type="button" onClick={onRetryRooms}>
                  Retry
                </button>
              ) : null}
            </div>
          ) : null}

          <div className={styles.roomList}>
            {roomsLoading && rooms.length === 0 ? (
              <div className={styles.sidebarEmpty} role="status">
                Loading rooms…
              </div>
            ) : visibleRooms.length === 0 ? (
              <div className={styles.sidebarEmpty}>
                {rooms.length === 0 ? 'No room yet.' : 'No rooms match the current filters.'}
              </div>
            ) : (
              visibleRooms.map((room) => {
                const active = room.id === selectedRoomId
                return (
                  <div
                    key={room.id}
                    className={`${styles.roomItem} ${active ? styles.roomItemActive : ''}`}
                  >
                    <button
                      type="button"
                      className={styles.roomSelectButton}
                      onClick={() => onSelectRoom(room.id)}
                      aria-current={active ? 'page' : undefined}
                    >
                      <span className={styles.roomName}>{room.name}</span>
                      <span className={styles.roomId}>{room.id}</span>
                    </button>
                    <button
                      type="button"
                      className={styles.roomDeleteButton}
                      onClick={(event) => {
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
          {filteredRooms.length > 0 ? (
            <div className={styles.roomPagination}>
              <span>
                {roomRange.start}–{roomRange.end} of {filteredRooms.length}
              </span>
              <div>
                <button
                  type="button"
                  onClick={() => setRoomPage(safePage - 1)}
                  disabled={safePage === 1}
                  aria-label="Previous rooms page"
                >
                  ‹
                </button>
                <span>
                  {safePage}/{pageCount}
                </span>
                <button
                  type="button"
                  onClick={() => setRoomPage(safePage + 1)}
                  disabled={safePage === pageCount}
                  aria-label="Next rooms page"
                >
                  ›
                </button>
              </div>
            </div>
          ) : null}
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
