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
import MarkdownRenderer from './MarkdownRenderer'
import styles from './ClawRoomChat.module.css'
import { useReadonly } from '../contexts/ReadonlyContext'
import ClawRoomSidebar from './ClawRoomSidebar'
import ClawRoomMessageMeta from './ClawRoomMessageMeta'
import {
  compareByCreatedAt,
  compareByName,
  findMentionRange,
  formatMessageTime,
  type MentionAutocompleteState,
  type MentionOption,
  parseJSON,
  roleLabel,
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
  const [messages, setMessages] = useState<RoomMessage[]>([])
  const [selectedTeamId, setSelectedTeamId] = useState('')
  const [selectedRoomId, setSelectedRoomId] = useState('')
  const [draft, setDraft] = useState('')
  const [loading, setLoading] = useState(true)
  const [posting, setPosting] = useState(false)
  const [creatingRoom, setCreatingRoom] = useState(false)
  const [deletingRoomId, setDeletingRoomId] = useState<string | null>(null)
  const [newRoomName, setNewRoomName] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [mentionAutocomplete, setMentionAutocomplete] = useState<MentionAutocompleteState | null>(null)

  const endRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement | null>(null)
  const lastCreateRoomRequestTokenRef = useRef(0)

  const selectedTeam = useMemo(() => teams.find(team => team.id === selectedTeamId) || null, [teams, selectedTeamId])
  const selectedRoom = useMemo(() => rooms.find(room => room.id === selectedRoomId) || null, [rooms, selectedRoomId])
  const managementDisabled = readonlyLoading || isReadonly

  const teamWorkers = useMemo(() => {
    return workers
      .filter(worker => worker.teamId === selectedTeamId)
      .sort(compareByName)
  }, [workers, selectedTeamId])

  const leaderWorker = useMemo(() => {
    if (selectedTeam?.leaderId) {
      const explicitLeader = teamWorkers.find(worker => worker.name === selectedTeam.leaderId)
      if (explicitLeader) {
        return explicitLeader
      }
    }
    return teamWorkers.find(worker => roleLabel(worker.roleKind) === 'leader') || null
  }, [selectedTeam?.leaderId, teamWorkers])

  const workerLookup = useMemo(() => {
    const map = new Map<string, WorkerProfile>()
    for (const worker of teamWorkers) {
      const keys = [worker.name, worker.agentName]
      for (const key of keys) {
        const normalized = sanitizeLookupKey(key)
        if (!normalized || map.has(normalized)) {
          continue
        }
        map.set(normalized, worker)
      }
    }
    return map
  }, [teamWorkers])

  const mentionOptions = useMemo<MentionOption[]>(() => {
    const entries: MentionOption[] = []
    const seen = new Set<string>()

    const allDesc =
      teamWorkers.length > 0
        ? `All claws in this team (${teamWorkers.length})`
        : 'All claws in this team'
    entries.push({ token: '@all', description: allDesc })
    seen.add('@all')

    const leaderDesc = leaderWorker
      ? `Leader alias (${leaderWorker.agentName || leaderWorker.name})`
      : 'Leader alias'
    entries.push({ token: '@leader', description: leaderDesc })
    seen.add('@leader')

    for (const worker of teamWorkers) {
      if (leaderWorker && worker.name === leaderWorker.name) {
        continue
      }
      const token = `@${worker.name}`
      if (seen.has(token)) {
        continue
      }
      seen.add(token)
      entries.push({
        token,
        description: worker.agentName || roleLabel(worker.roleKind),
      })
    }
    return entries
  }, [leaderWorker, teamWorkers])

  const leaderRoleText = leaderWorker?.agentRole || selectedTeam?.role || 'Team Leader'
  const memberResumeProfiles = useMemo(() => {
    const profiles = teamWorkers.map(worker => {
      const isLeader = selectedTeam?.leaderId === worker.name || roleLabel(worker.roleKind) === 'leader'
      return {
        id: worker.name,
        isLeader,
        displayName: worker.agentName || worker.name,
        roleText: worker.agentRole || (isLeader ? leaderRoleText : 'Team Worker'),
        vibeText: worker.agentVibe || selectedTeam?.vibe || 'Execution-focused',
      }
    })

    profiles.sort((a, b) => {
      if (a.isLeader !== b.isLeader) {
        return a.isLeader ? -1 : 1
      }
      return a.displayName.localeCompare(b.displayName)
    })

    return profiles
  }, [leaderRoleText, selectedTeam?.leaderId, selectedTeam?.vibe, teamWorkers])
  const teamBriefText = useMemo(() => {
    if (selectedTeam?.description?.trim()) {
      return selectedTeam.description.trim()
    }
    if (selectedTeam?.principal?.trim()) {
      return selectedTeam.principal.trim()
    }
    if (leaderWorker?.agentPrinciples?.trim()) {
      return leaderWorker.agentPrinciples.trim()
    }
    return 'Use @leader to delegate work. Workers can also report progress or blockers back via @leader.'
  }, [leaderWorker?.agentPrinciples, selectedTeam?.description, selectedTeam?.principal])

  const upsertMessage = useCallback((message: RoomMessage) => {
    setMessages(prev => {
      const index = prev.findIndex(existing => existing.id === message.id)
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
      const filtered = mentionOptions.filter(option =>
        option.token.slice(1).toLowerCase().startsWith(range.query)
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
    [mentionOptions]
  )

  const refreshMentionAutocomplete = useCallback(
    (value: string, caret: number) => {
      setMentionAutocomplete(previous => {
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
    [computeMentionAutocomplete]
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

    setSelectedTeamId(prev => {
      if (prev && sortedTeams.some(team => team.id === prev)) {
        return prev
      }
      return sortedTeams[0]?.id || ''
    })
  }, [])

  const fetchRooms = useCallback(async (teamId: string) => {
    if (!teamId) {
      setRooms([])
      setSelectedRoomId('')
      return
    }

    const resp = await fetch(`/api/openclaw/rooms?teamId=${encodeURIComponent(teamId)}`)
    if (!resp.ok) {
      throw new Error(`Failed to load rooms: ${resp.status}`)
    }

    const data = await parseJSON<RoomEntry[]>(resp)
    const nextRooms = (Array.isArray(data) ? data : []).sort(compareByName)
    setRooms(nextRooms)

    setSelectedRoomId(prev => {
      if (prev && nextRooms.some(room => room.id === prev)) {
        return prev
      }
      return nextRooms[0]?.id || ''
    })
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
    wsConnected,
    wsRef,
  } = useClawRoomTransport({
    selectedRoomId,
    fetchMessages,
    setMessages,
    setError,
    upsertMessage,
  })
  void streamingMessages

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

    let mounted = true

    const loadRooms = async () => {
      try {
        await fetchRooms(selectedTeamId)
        if (mounted) {
          setError(null)
        }
      } catch (err) {
        if (!mounted) return
        const message = err instanceof Error ? err.message : 'Failed to load rooms'
        setError(message)
      }
    }

    void loadRooms()

    return () => {
      mounted = false
    }
  }, [fetchRooms, selectedTeamId])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

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
        const resp = await fetch(`/api/openclaw/rooms/${encodeURIComponent(selectedRoomId)}/messages`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            content,
            senderType: 'user',
            senderName: 'You',
            senderId: 'playground-user',
          }),
        })
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

  const handleCreateRoom = useCallback(async (event?: FormEvent<HTMLFormElement>) => {
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
  }, [creatingRoom, fetchRooms, managementDisabled, newRoomName, selectedTeamId])

  useEffect(() => {
    if (createRoomRequestToken <= lastCreateRoomRequestTokenRef.current) {
      return
    }
    lastCreateRoomRequestTokenRef.current = createRoomRequestToken
    void handleCreateRoom()
  }, [createRoomRequestToken, handleCreateRoom])

  const handleDeleteRoom = useCallback(async (room: Pick<RoomEntry, 'id' | 'name'>) => {
    if (managementDisabled || !room?.id || deletingRoomId) {
      return
    }
    const ok = window.confirm(`Delete room "${room.name}"?`)
    if (!ok) {
      return
    }

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
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to delete room'
      setError(message)
    } finally {
      setDeletingRoomId(null)
    }
  }, [deletingRoomId, fetchRooms, managementDisabled, selectedRoomId, selectedTeamId])

  const handleDraftChange = useCallback((event: ChangeEvent<HTMLTextAreaElement>) => {
    const { value, selectionStart } = event.target
    setDraft(value)
    refreshMentionAutocomplete(value, selectionStart ?? value.length)
  }, [refreshMentionAutocomplete])

  const syncMentionByCursor = useCallback(() => {
    const element = inputRef.current
    if (!element) {
      return
    }
    refreshMentionAutocomplete(draft, element.selectionStart ?? draft.length)
  }, [draft, refreshMentionAutocomplete])

  const selectMentionOption = useCallback((option: MentionOption) => {
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
  }, [draft, mentionAutocomplete])

  const handleDraftKeyDown = useCallback((event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (mentionAutocomplete && mentionAutocomplete.options.length > 0) {
      if (event.key === 'ArrowDown') {
        event.preventDefault()
        setMentionAutocomplete(previous => {
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
        setMentionAutocomplete(previous => {
          if (!previous) return previous
          return {
            ...previous,
            activeIndex: (previous.activeIndex - 1 + previous.options.length) % previous.options.length,
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
  }, [handleSend, mentionAutocomplete, selectMentionOption])

  const resolveSenderVisual = useCallback((message: RoomMessage): SenderVisual => {
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
  }, [workerLookup])

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
            onCreateRoom={event => void handleCreateRoom(event)}
            onDeleteRoom={room => void handleDeleteRoom(room)}
            onSelectRoom={setSelectedRoomId}
            onSelectTeam={setSelectedTeamId}
            rooms={rooms}
            selectedRoom={selectedRoom}
            selectedRoomId={selectedRoomId}
            selectedTeam={selectedTeam}
            selectedTeamId={selectedTeamId}
            teamBriefText={teamBriefText}
            teams={teams}
          />
        )}

        <section className={styles.chatPanel}>
          <header className={styles.chatHeader} data-testid="claw-room-header">
            <div className={styles.chatTitleWrap}>
              <h3 className={styles.chatTitle}>{selectedRoom?.name || selectedTeam?.name || 'No room selected'}</h3>
              {selectedRoomId && (
                <span
                  className={`${styles.chatTitleStatus} ${wsConnected ? styles.wsConnected : styles.wsDisconnected}`}
                  title={wsConnected ? 'WebSocket connected' : 'WebSocket disconnected (using fallback)'}
                >
                  {wsConnected ? '● Live' : '○ Reconnecting...'}
                </span>
              )}
            </div>
          </header>

          <div className={styles.messages} data-testid="claw-room-transcript">
            {!selectedRoomId ? (
              <div className={styles.stateHint}>Select a room from the left panel.</div>
            ) : messages.length === 0 ? (
              <div className={styles.stateHint}>No messages yet. Start the conversation.</div>
            ) : (
              messages.map(message => {
                const isUser = message.senderType === 'user'
                const isSystem = message.senderType === 'system'
                const isLeader = message.senderType === 'leader'
                const isWorker = message.senderType === 'worker'
                const senderVisual = resolveSenderVisual(message)
                return (
                  <div
                    key={message.id}
                    className={`${styles.messageRow} ${isUser ? styles.messageRowUser : styles.messageRowAgent}`}
                    data-room-message-id={message.id}
                    data-room-message-role={message.senderType}
                  >
                    <div className={styles.messageMain}>
                      <ClawRoomMessageMeta
                        displayName={senderVisual.displayName}
                        isLeader={isLeader}
                        isUser={isUser}
                        isWorker={isWorker}
                        roleLabel={senderVisual.roleLabel}
                        timestamp={formatMessageTime(message.createdAt)}
                      />
                      <div
                        className={`${styles.messageBubble} ${isUser ? styles.messageBubbleUser : styles.messageBubbleAgent} ${isSystem ? styles.messageBubbleSystem : ''}`}
                        data-room-message-content
                      >
                        <div className={styles.messageMarkdown}>
                          <MarkdownRenderer content={message.content} />
                        </div>
                      </div>
                    </div>
                  </div>
                )
              })
            )}
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
                    <div className={styles.inputModeControls}>
                      {inputModeControls}
                    </div>
                  )}
                  <button
                    type="button"
                    className={styles.sendButton}
                    onClick={() => void handleSend()}
                    disabled={!selectedRoomId || posting || !draft.trim()}
                    title="Send message"
                    aria-label="Send message"
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                      <path d="M12 19V5M5 12l7-7 7 7" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </button>
                </div>
              </div>

              {mentionAutocomplete && mentionAutocomplete.options.length > 0 && (
                <div className={styles.mentionMenu} role="listbox" aria-label="Mention suggestions">
                  {mentionAutocomplete.options.map((option, index) => {
                    const isActive = mentionAutocomplete.activeIndex === index
                    return (
                      <button
                        key={option.token}
                        type="button"
                        className={`${styles.mentionItem} ${isActive ? styles.mentionItemActive : ''}`}
                        onMouseDown={event => {
                          event.preventDefault()
                          selectMentionOption(option)
                        }}
                        onMouseEnter={() => {
                          setMentionAutocomplete(previous => {
                            if (!previous) return previous
                            return { ...previous, activeIndex: index }
                          })
                        }}
                      >
                        <span className={styles.mentionToken}>{option.token}</span>
                        <span className={styles.mentionDescription}>{option.description}</span>
                      </button>
                    )
                  })}
                </div>
              )}
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}

export default ClawRoomChat
