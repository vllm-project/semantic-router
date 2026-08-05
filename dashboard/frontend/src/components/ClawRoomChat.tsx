import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type FormEvent,
  type KeyboardEvent,
  type ReactNode,
} from 'react'
import styles from './ClawRoomChat.module.css'
import { useReadonly } from '../contexts/ReadonlyContext'
import {
  createLatestOpenClawRequest,
  fetchOpenClawJSON,
  getOpenClawErrorMessage,
  type LatestOpenClawRequest,
} from '../utils/openClawRequestSupport'
import ClawRoomMentionMenu from './ClawRoomMentionMenu'
import ClawRoomSidebar from './ClawRoomSidebar'
import ClawRoomTranscript from './ClawRoomTranscript'
import ClawRoomTransportStatus from './ClawRoomTransportStatus'
import ConfirmDialog from './ConfirmDialog'
import { buildStreamingEntries } from './clawRoomStreamingUi'
import { buildClawRoomTeamView } from './clawRoomTeamViewSupport'
import {
  compareByCreatedAt,
  compareByName,
  findMentionRange,
  type MentionAutocompleteState,
  type MentionOption,
  parseJSON,
  sanitizeLookupKey,
  type SenderVisual,
  type TeamProfile,
  type RoomEntry,
  type RoomMessage,
  type WorkerProfile,
  type WSInboundMessage,
} from './clawRoomChatSupport'
import { useClawRoomTransport } from './useClawRoomTransport'

interface ClawRoomChatProps {
  isSidebarOpen?: boolean
  createRoomRequestToken?: number
  inputModeControls?: ReactNode
}

const ClawRoomChat = ({
  isSidebarOpen = true,
  createRoomRequestToken = 0,
  inputModeControls,
}: ClawRoomChatProps) => {
  const { isReadonly, isLoading: readonlyLoading } = useReadonly()
  const [teams, setTeams] = useState<TeamProfile[]>([])
  const [workers, setWorkers] = useState<WorkerProfile[]>([])
  const [rooms, setRooms] = useState<RoomEntry[]>([])
  const [roomsLoading, setRoomsLoading] = useState(false)
  const [roomsError, setRoomsError] = useState<string | null>(null)
  const [messages, setMessages] = useState<RoomMessage[]>([])
  const [selectedTeamId, setSelectedTeamId] = useState('')
  const [selectedRoomId, setSelectedRoomId] = useState('')
  const [draft, setDraft] = useState('')
  const [loading, setLoading] = useState(true)
  const [posting, setPosting] = useState(false)
  const [creatingRoom, setCreatingRoom] = useState(false)
  const [deletingRoomId, setDeletingRoomId] = useState<string | null>(null)
  const [roomPendingDelete, setRoomPendingDelete] = useState<Pick<RoomEntry, 'id' | 'name'> | null>(
    null,
  )
  const [newRoomName, setNewRoomName] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [mentionAutocomplete, setMentionAutocomplete] = useState<MentionAutocompleteState | null>(
    null,
  )

  const endRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement | null>(null)
  const lastCreateRoomRequestTokenRef = useRef(0)
  const roomsRequestRef = useRef<LatestOpenClawRequest | null>(null)
  if (!roomsRequestRef.current) roomsRequestRef.current = createLatestOpenClawRequest()

  const selectedTeam = useMemo(
    () => teams.find((team) => team.id === selectedTeamId) || null,
    [teams, selectedTeamId],
  )
  const selectedRoom = useMemo(
    () => rooms.find((room) => room.id === selectedRoomId) || null,
    [rooms, selectedRoomId],
  )
  const managementDisabled = readonlyLoading || isReadonly

  const { memberResumeProfiles, mentionOptions, teamBriefText, workerLookup } = useMemo(
    () => buildClawRoomTeamView(selectedTeam, workers, selectedTeamId),
    [selectedTeam, selectedTeamId, workers],
  )

  const upsertMessage = useCallback((message: RoomMessage) => {
    setMessages((prev) => {
      const index = prev.findIndex((existing) => existing.id === message.id)
      if (index >= 0) {
        const next = [...prev]
        next[index] = message
        next.sort(compareByCreatedAt)
        return next
      }
      const next = [...prev, message]
      next.sort(compareByCreatedAt)
      return next
    })
  }, [])

  const computeMentionAutocomplete = useCallback(
    (value: string, caret: number): MentionAutocompleteState | null => {
      const range = findMentionRange(value, caret)
      if (!range) {
        return null
      }
      const filtered = mentionOptions.filter((option) =>
        option.token.slice(1).toLowerCase().startsWith(range.query),
      )
      if (filtered.length === 0) {
        return null
      }
      return {
        ...range,
        options: filtered,
        activeIndex: 0,
      }
    },
    [mentionOptions],
  )

  const refreshMentionAutocomplete = useCallback(
    (value: string, caret: number) => {
      setMentionAutocomplete((previous) => {
        const next = computeMentionAutocomplete(value, caret)
        if (!next) {
          return null
        }
        if (
          previous &&
          previous.start === next.start &&
          previous.end === next.end &&
          previous.query === next.query
        ) {
          return {
            ...next,
            activeIndex: Math.min(previous.activeIndex, next.options.length - 1),
          }
        }
        return next
      })
    },
    [computeMentionAutocomplete],
  )

  const fetchTeamsAndWorkers = useCallback(async () => {
    const [teamsResp, workersResp] = await Promise.all([
      fetch('/api/openclaw/teams'),
      fetch('/api/openclaw/workers'),
    ])

    if (!teamsResp.ok) {
      throw new Error(`Failed to load teams: ${teamsResp.status}`)
    }
    if (!workersResp.ok) {
      throw new Error(`Failed to load workers: ${workersResp.status}`)
    }

    const teamsData = await parseJSON<TeamProfile[]>(teamsResp)
    const workersData = await parseJSON<WorkerProfile[]>(workersResp)

    const sortedTeams = [...(Array.isArray(teamsData) ? teamsData : [])].sort(compareByName)
    const sortedWorkers = [...(Array.isArray(workersData) ? workersData : [])].sort(compareByName)

    setTeams(sortedTeams)
    setWorkers(sortedWorkers)

    setSelectedTeamId((prev) => {
      if (prev && sortedTeams.some((team) => team.id === prev)) {
        return prev
      }
      return sortedTeams[0]?.id || ''
    })
  }, [])

  const fetchRooms = useCallback(async (teamId: string) => {
    if (!teamId) {
      roomsRequestRef.current?.cancel()
      setRooms([])
      setSelectedRoomId('')
      setRoomsError(null)
      return
    }

    await roomsRequestRef.current?.run(
      (signal) =>
        fetchOpenClawJSON<RoomEntry[]>(
          `/api/openclaw/rooms?teamId=${encodeURIComponent(teamId)}`,
          {},
          signal,
        ),
      {
        onStart: () => {
          setRoomsLoading(true)
          setRoomsError(null)
        },
        onSuccess: (data) => {
          const nextRooms = (Array.isArray(data) ? data : []).sort(compareByName)
          setRooms(nextRooms)
          setSelectedRoomId((prev) => {
            if (prev && nextRooms.some((room) => room.id === prev)) return prev
            return nextRooms[0]?.id || ''
          })
        },
        onError: (requestError) => {
          setRoomsError(getOpenClawErrorMessage(requestError, 'Failed to load rooms.'))
        },
        onFinish: () => setRoomsLoading(false),
      },
    )
  }, [])

  const fetchMessages = useCallback(async (roomId: string) => {
    if (!roomId) {
      setMessages([])
      return
    }

    const resp = await fetch(`/api/openclaw/rooms/${encodeURIComponent(roomId)}/messages?limit=300`)
    if (!resp.ok) {
      throw new Error(`Failed to load messages: ${resp.status}`)
    }
    const data = await parseJSON<RoomMessage[]>(resp)
    const nextMessages = (Array.isArray(data) ? data : []).sort(compareByCreatedAt)
    setMessages(nextMessages)
  }, [])

  const {
    streamingMessages,
    streamingToolTraces,
    streamingParticipants,
    transportMode,
    wsConnected,
    wsRef,
  } = useClawRoomTransport({
    selectedRoomId,
    fetchMessages,
    setMessages,
    setError,
    upsertMessage,
  })

  const streamingEntries = useMemo(
    () => buildStreamingEntries(messages, streamingMessages, streamingToolTraces),
    [messages, streamingMessages, streamingToolTraces],
  )

  useEffect(() => {
    let mounted = true

    const load = async () => {
      setLoading(true)
      try {
        await fetchTeamsAndWorkers()
        if (mounted) {
          setError(null)
        }
      } catch (err) {
        if (!mounted) return
        const message = err instanceof Error ? err.message : 'Failed to load Claw room context'
        setError(message)
      } finally {
        if (mounted) {
          setLoading(false)
        }
      }
    }

    void load()

    return () => {
      mounted = false
    }
  }, [fetchTeamsAndWorkers])

  useEffect(() => {
    if (!selectedTeamId) {
      setRooms([])
      setSelectedRoomId('')
      return
    }

    void fetchRooms(selectedTeamId)
  }, [fetchRooms, selectedTeamId])

  useEffect(() => () => roomsRequestRef.current?.cancel(), [])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamingEntries])

  useEffect(() => {
    if (!selectedRoomId) {
      setMentionAutocomplete(null)
    }
  }, [selectedRoomId])

  useEffect(() => {
    const element = inputRef.current
    if (!element) {
      return
    }
    if (!draft.trim()) {
      setMentionAutocomplete(null)
      return
    }
    const caret = element.selectionStart ?? draft.length
    refreshMentionAutocomplete(draft, caret)
  }, [draft, mentionOptions, refreshMentionAutocomplete])

  const handleSend = useCallback(async () => {
    if (!selectedRoomId) return
    const content = draft.trim()
    if (!content || posting) return

    setPosting(true)
    try {
      // Try WebSocket first if connected
      if (wsConnected && wsRef.current?.readyState === WebSocket.OPEN) {
        const wsMessage: WSInboundMessage = {
          type: 'send_message',
          content,
          senderType: 'user',
          senderName: 'You',
          senderId: 'playground-user',
        }
        wsRef.current.send(JSON.stringify(wsMessage))
        setDraft('')
        setMentionAutocomplete(null)
        setError(null)
      } else {
        // Fallback to HTTP POST
        const resp = await fetch(
          `/api/openclaw/rooms/${encodeURIComponent(selectedRoomId)}/messages`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              content,
              senderType: 'user',
              senderName: 'You',
              senderId: 'playground-user',
            }),
          },
        )
        if (!resp.ok) {
          const body = await resp.text()
          throw new Error(body || `Send failed (${resp.status})`)
        }
        const created = await parseJSON<RoomMessage>(resp)
        upsertMessage(created)
        setDraft('')
        setMentionAutocomplete(null)
        setError(null)
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to send room message'
      setError(message)
    } finally {
      setPosting(false)
    }
  }, [draft, posting, selectedRoomId, upsertMessage, wsConnected, wsRef])

  const handleCreateRoom = useCallback(
    async (event?: FormEvent<HTMLFormElement>) => {
      event?.preventDefault()
      if (managementDisabled || !selectedTeamId || creatingRoom) {
        return
      }

      setCreatingRoom(true)
      try {
        const payload: { teamId: string; name?: string } = {
          teamId: selectedTeamId,
        }
        const roomName = newRoomName.trim()
        if (roomName) {
          payload.name = roomName
        }

        const resp = await fetch('/api/openclaw/rooms', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        })
        if (!resp.ok) {
          const body = await resp.text()
          throw new Error(body || `Create room failed (${resp.status})`)
        }

        const created = await parseJSON<RoomEntry>(resp)
        setNewRoomName('')
        if (created?.id) {
          setSelectedRoomId(created.id)
        }
        await fetchRooms(selectedTeamId)
        setError(null)
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to create room'
        setError(message)
      } finally {
        setCreatingRoom(false)
      }
    },
    [creatingRoom, fetchRooms, managementDisabled, newRoomName, selectedTeamId],
  )

  useEffect(() => {
    if (createRoomRequestToken <= lastCreateRoomRequestTokenRef.current) {
      return
    }
    lastCreateRoomRequestTokenRef.current = createRoomRequestToken
    void handleCreateRoom()
  }, [createRoomRequestToken, handleCreateRoom])

  const handleDeleteRoom = useCallback(async () => {
    if (managementDisabled || !roomPendingDelete?.id || deletingRoomId) {
      return
    }
    const room = roomPendingDelete

    setDeletingRoomId(room.id)
    try {
      const resp = await fetch(`/api/openclaw/rooms/${encodeURIComponent(room.id)}`, {
        method: 'DELETE',
      })
      if (!resp.ok) {
        const body = await resp.text()
        throw new Error(body || `Delete room failed (${resp.status})`)
      }

      if (selectedRoomId === room.id) {
        setSelectedRoomId('')
        setMessages([])
      }
      await fetchRooms(selectedTeamId)
      setError(null)
      setRoomPendingDelete(null)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to delete room'
      setError(message)
    } finally {
      setDeletingRoomId(null)
    }
  }, [
    deletingRoomId,
    fetchRooms,
    managementDisabled,
    roomPendingDelete,
    selectedRoomId,
    selectedTeamId,
  ])

  const handleDraftChange = useCallback(
    (event: ChangeEvent<HTMLTextAreaElement>) => {
      const { value, selectionStart } = event.target
      setDraft(value)
      refreshMentionAutocomplete(value, selectionStart ?? value.length)
    },
    [refreshMentionAutocomplete],
  )

  const syncMentionByCursor = useCallback(() => {
    const element = inputRef.current
    if (!element) {
      return
    }
    refreshMentionAutocomplete(draft, element.selectionStart ?? draft.length)
  }, [draft, refreshMentionAutocomplete])

  const selectMentionOption = useCallback(
    (option: MentionOption) => {
      if (!mentionAutocomplete) {
        return
      }

      const nextDraft = `${draft.slice(0, mentionAutocomplete.start)}${option.token} ${draft.slice(mentionAutocomplete.end)}`
      const nextCaret = mentionAutocomplete.start + option.token.length + 1
      setDraft(nextDraft)
      setMentionAutocomplete(null)

      requestAnimationFrame(() => {
        const element = inputRef.current
        if (!element) return
        element.focus()
        element.setSelectionRange(nextCaret, nextCaret)
      })
    },
    [draft, mentionAutocomplete],
  )

  const handleDraftKeyDown = useCallback(
    (event: KeyboardEvent<HTMLTextAreaElement>) => {
      if (mentionAutocomplete && mentionAutocomplete.options.length > 0) {
        if (event.key === 'ArrowDown') {
          event.preventDefault()
          setMentionAutocomplete((previous) => {
            if (!previous) return previous
            return {
              ...previous,
              activeIndex: (previous.activeIndex + 1) % previous.options.length,
            }
          })
          return
        }

        if (event.key === 'ArrowUp') {
          event.preventDefault()
          setMentionAutocomplete((previous) => {
            if (!previous) return previous
            return {
              ...previous,
              activeIndex:
                (previous.activeIndex - 1 + previous.options.length) % previous.options.length,
            }
          })
          return
        }

        if (event.key === 'Escape') {
          event.preventDefault()
          setMentionAutocomplete(null)
          return
        }

        if (event.key === 'Tab' || (event.key === 'Enter' && !event.shiftKey)) {
          event.preventDefault()
          const option = mentionAutocomplete.options[mentionAutocomplete.activeIndex]
          if (option) {
            selectMentionOption(option)
          }
          return
        }
      }

      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault()
        void handleSend()
      }
    },
    [handleSend, mentionAutocomplete, selectMentionOption],
  )

  const resolveSenderVisual = useCallback(
    (message: RoomMessage): SenderVisual => {
      if (message.senderType === 'user') {
        return {
          displayName: message.senderName || 'You',
          roleLabel: 'USER',
        }
      }

      if (message.senderType === 'system') {
        return {
          displayName: message.senderName || 'ClawOS',
          roleLabel: 'SYSTEM',
        }
      }

      const lookupByID = workerLookup.get(sanitizeLookupKey(message.senderId))
      const lookupByName = workerLookup.get(sanitizeLookupKey(message.senderName))
      const worker = lookupByID || lookupByName

      const displayName = worker?.agentName || message.senderName || message.senderId || 'Claw'

      return {
        displayName,
        roleLabel: message.senderType === 'leader' ? 'LEADER' : 'WORKER',
      }
    },
    [workerLookup],
  )

  const containerClassName = `${styles.container} ${isSidebarOpen ? styles.containerSidebarOpen : ''}`

  if (loading) {
    return (
      <div className={containerClassName}>
        <div className={styles.loadingShell} aria-live="polite">
          <div className={styles.loadingTopRow}>
            <div className={`${styles.loadingTitle} ${styles.loadingPulse}`} />
            <div className={`${styles.loadingBadge} ${styles.loadingPulse}`} />
          </div>
          <div className={styles.loadingSubtitle}>Loading Claw room context...</div>

          <div className={styles.loadingLayout}>
            {isSidebarOpen && (
              <aside className={styles.loadingSidebar}>
                <div className={`${styles.loadingLine} ${styles.loadingPulse}`} />
                <div className={`${styles.loadingLineWide} ${styles.loadingPulse}`} />
                <div className={`${styles.loadingLine} ${styles.loadingPulse}`} />
                <div className={`${styles.loadingRoomItem} ${styles.loadingPulse}`} />
                <div className={`${styles.loadingRoomItem} ${styles.loadingPulse}`} />
                <div className={`${styles.loadingRoomItem} ${styles.loadingPulse}`} />
              </aside>
            )}

            <section className={styles.loadingChat}>
              <div className={styles.loadingChatHeader}>
                <div className={`${styles.loadingLineWide} ${styles.loadingPulse}`} />
                <div className={styles.loadingChipRow}>
                  <div className={`${styles.loadingChip} ${styles.loadingPulse}`} />
                  <div className={`${styles.loadingChip} ${styles.loadingPulse}`} />
                  <div className={`${styles.loadingChip} ${styles.loadingPulse}`} />
                </div>
              </div>
              <div className={styles.loadingMessages}>
                <div className={`${styles.loadingBubbleWide} ${styles.loadingPulse}`} />
                <div className={`${styles.loadingBubble} ${styles.loadingPulse}`} />
                <div className={`${styles.loadingBubbleWide} ${styles.loadingPulse}`} />
              </div>
              <div className={`${styles.loadingInput} ${styles.loadingPulse}`} />
            </section>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className={containerClassName}>
      {error && <div className={styles.error}>{error}</div>}

      <div className={styles.layout}>
        {isSidebarOpen && (
          <ClawRoomSidebar
            creatingRoom={creatingRoom}
            deletingRoomId={deletingRoomId}
            managementDisabled={managementDisabled}
            memberProfiles={memberResumeProfiles}
            newRoomName={newRoomName}
            onChangeNewRoomName={setNewRoomName}
            onCreateRoom={(event) => void handleCreateRoom(event)}
            onDeleteRoom={(room) => {
              if (!managementDisabled && !deletingRoomId) setRoomPendingDelete(room)
            }}
            onSelectRoom={setSelectedRoomId}
            onSelectTeam={setSelectedTeamId}
            rooms={rooms}
            roomsError={roomsError}
            roomsLoading={roomsLoading}
            selectedRoom={selectedRoom}
            selectedRoomId={selectedRoomId}
            selectedTeam={selectedTeam}
            selectedTeamId={selectedTeamId}
            teamBriefText={teamBriefText}
            teams={teams}
            onRetryRooms={() => void fetchRooms(selectedTeamId)}
          />
        )}

        <section className={styles.chatPanel}>
          <header className={styles.chatHeader} data-testid="claw-room-header">
            <div className={styles.chatTitleWrap}>
              <h3 className={styles.chatTitle}>
                {selectedRoom?.name || selectedTeam?.name || 'No room selected'}
              </h3>
              {selectedRoomId && (
                <ClawRoomTransportStatus transportMode={transportMode} wsConnected={wsConnected} />
              )}
            </div>
          </header>

          <div className={styles.messages} data-testid="claw-room-transcript">
            <ClawRoomTranscript
              messages={messages}
              resolveSenderVisual={resolveSenderVisual}
              selectedRoomId={selectedRoomId}
              streamingEntries={streamingEntries}
              streamingMessages={streamingMessages}
              streamingToolTraces={streamingToolTraces}
              streamingParticipants={streamingParticipants}
            />
            <div ref={endRef} />
          </div>

          <div className={styles.inputArea}>
            <div className={styles.inputStack}>
              <div className={styles.inputShell}>
                <textarea
                  ref={inputRef}
                  className={styles.input}
                  value={draft}
                  onChange={handleDraftChange}
                  onClick={syncMentionByCursor}
                  onKeyUp={syncMentionByCursor}
                  onKeyDown={handleDraftKeyDown}
                  placeholder="@all to mention everyone, @leader to assign tasks, or @worker-name"
                  rows={1}
                  disabled={!selectedRoomId || posting}
                />
                <div className={styles.inputActionsRow}>
                  {inputModeControls && (
                    <div className={styles.inputModeControls}>{inputModeControls}</div>
                  )}
                  <button
                    type="button"
                    className={styles.sendButton}
                    onClick={() => void handleSend()}
                    disabled={!selectedRoomId || posting || !draft.trim()}
                    title="Send message"
                    aria-label="Send message"
                  >
                    <svg
                      width="16"
                      height="16"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2.5"
                    >
                      <path
                        d="M12 19V5M5 12l7-7 7 7"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                  </button>
                </div>
              </div>

              {mentionAutocomplete && (
                <ClawRoomMentionMenu
                  mentionAutocomplete={mentionAutocomplete}
                  onSelect={selectMentionOption}
                  onActiveIndexChange={(index) => {
                    setMentionAutocomplete((previous) => {
                      if (!previous) return previous
                      return { ...previous, activeIndex: index }
                    })
                  }}
                />
              )}
            </div>
          </div>
        </section>
      </div>

      <ConfirmDialog
        isOpen={Boolean(roomPendingDelete)}
        title="Delete room"
        description={
          <>
            Delete <strong>{roomPendingDelete?.name}</strong> and remove its conversation history?
          </>
        }
        confirmLabel="Delete room"
        pending={Boolean(deletingRoomId)}
        onCancel={() => {
          if (!deletingRoomId) setRoomPendingDelete(null)
        }}
        onConfirm={handleDeleteRoom}
      />
    </div>
  )
}

export default ClawRoomChat
