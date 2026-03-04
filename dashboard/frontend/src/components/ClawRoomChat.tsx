import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type FormEvent,
  type KeyboardEvent,
} from 'react'
import MarkdownRenderer from './MarkdownRenderer'
import styles from './ClawRoomChat.module.css'

interface TeamProfile {
  id: string
  name: string
  vibe?: string
  role?: string
  principal?: string
  description?: string
  leaderId?: string
}

interface WorkerProfile {
  name: string
  teamId?: string
  agentName?: string
  agentEmoji?: string
  agentRole?: string
  agentVibe?: string
  agentPrinciples?: string
  roleKind?: string
}

interface RoomEntry {
  id: string
  teamId: string
  name: string
  createdAt?: string
  updatedAt?: string
}

interface RoomMessage {
  id: string
  roomId: string
  teamId: string
  senderType: 'user' | 'leader' | 'worker' | 'system'
  senderId?: string
  senderName: string
  content: string
  mentions?: string[]
  createdAt: string
  metadata?: Record<string, string>
}

interface RoomStreamEvent {
  type: string
  roomId: string
  message?: RoomMessage
}

interface MentionOption {
  token: string
  description: string
}

interface MentionAutocompleteState {
  start: number
  end: number
  query: string
  options: MentionOption[]
  activeIndex: number
}

interface SenderVisual {
  displayName: string
  roleLabel: string
  avatar: string
}

interface ClawRoomChatProps {
  isSidebarOpen?: boolean
  createRoomRequestToken?: number
}

const parseJSON = async <T,>(resp: Response): Promise<T> => {
  const text = await resp.text()
  if (!text.trim()) {
    return {} as T
  }
  return JSON.parse(text) as T
}

const roleLabel = (roleKind: string | undefined): 'leader' | 'worker' => {
  if (typeof roleKind === 'string' && roleKind.trim().toLowerCase() === 'leader') {
    return 'leader'
  }
  return 'worker'
}

const compareByName = <T extends { name?: string }>(a: T, b: T): number => {
  return (a.name || '').localeCompare(b.name || '')
}

const compareByCreatedAt = (a: RoomMessage, b: RoomMessage): number => {
  const aTime = Date.parse(a.createdAt)
  const bTime = Date.parse(b.createdAt)
  if (Number.isNaN(aTime) || Number.isNaN(bTime)) {
    return a.createdAt.localeCompare(b.createdAt)
  }
  return aTime - bTime
}

const mentionQueryPattern = /^@[a-zA-Z0-9_.-]*$/

const findMentionRange = (text: string, caret: number): { start: number; end: number; query: string } | null => {
  if (!text || caret < 0 || caret > text.length) {
    return null
  }

  let start = caret
  while (start > 0) {
    const ch = text[start - 1]
    if (/\s/.test(ch)) break
    start -= 1
  }

  const token = text.slice(start, caret)
  if (!token.startsWith('@') || !mentionQueryPattern.test(token)) {
    return null
  }

  return {
    start,
    end: caret,
    query: token.slice(1).toLowerCase(),
  }
}

const formatMessageTime = (raw: string): string => {
  const time = new Date(raw)
  if (Number.isNaN(time.getTime())) {
    return '--:--'
  }
  return time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

const sanitizeLookupKey = (value: string | undefined): string => {
  return (value || '').trim().toLowerCase()
}

const firstGlyph = (value: string): string => {
  const input = value.trim()
  if (!input) {
    return '🤖'
  }
  return input[0]?.toUpperCase() || '🤖'
}

const ClawRoomChat = ({
  isSidebarOpen = true,
  createRoomRequestToken = 0,
}: ClawRoomChatProps) => {
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
  const [settingLeaderId, setSettingLeaderId] = useState<string | null>(null)
  const [mentionAutocomplete, setMentionAutocomplete] = useState<MentionAutocompleteState | null>(null)

  const endRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement | null>(null)
  const sourceRef = useRef<EventSource | null>(null)
  const reconnectTimerRef = useRef<number | null>(null)
  const lastCreateRoomRequestTokenRef = useRef(0)

  const selectedTeam = useMemo(
    () => teams.find(team => team.id === selectedTeamId) || null,
    [teams, selectedTeamId]
  )

  const selectedRoom = useMemo(
    () => rooms.find(room => room.id === selectedRoomId) || null,
    [rooms, selectedRoomId]
  )

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

  const workerMentionTokens = useMemo(() => {
    return teamWorkers
      .filter(worker => worker.name !== leaderWorker?.name)
      .map(worker => `@${worker.name}`)
  }, [leaderWorker?.name, teamWorkers])

  const quickMentionTokens = useMemo(() => {
    return Array.from(new Set(['@leader', ...workerMentionTokens]))
  }, [workerMentionTokens])

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

  const mentionHints = useMemo(() => mentionOptions.map(option => option.token), [mentionOptions])

  const leaderRoleText = leaderWorker?.agentRole || selectedTeam?.role || 'Team Leader'
  const leaderVibeText = leaderWorker?.agentVibe || selectedTeam?.vibe || 'Coordination-driven'
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
    if (!selectedRoomId) {
      setMessages([])
      if (sourceRef.current) {
        sourceRef.current.close()
        sourceRef.current = null
      }
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current)
        reconnectTimerRef.current = null
      }
      return
    }

    let mounted = true

    const loadMessages = async () => {
      try {
        await fetchMessages(selectedRoomId)
        if (mounted) {
          setError(null)
        }
      } catch (err) {
        if (!mounted) return
        const message = err instanceof Error ? err.message : 'Failed to load messages'
        setError(message)
      }
    }

    const connect = () => {
      if (!mounted) return
      if (sourceRef.current) {
        sourceRef.current.close()
      }

      const source = new EventSource(`/api/openclaw/rooms/${encodeURIComponent(selectedRoomId)}/stream`)
      sourceRef.current = source

      source.addEventListener('message', ((event: MessageEvent<string>) => {
        try {
          const payload = JSON.parse(event.data) as RoomStreamEvent
          if (payload.message) {
            upsertMessage(payload.message)
          }
        } catch {
          // ignore malformed stream events
        }
      }) as EventListener)

      source.onerror = () => {
        source.close()
        if (!mounted) return
        if (reconnectTimerRef.current !== null) {
          window.clearTimeout(reconnectTimerRef.current)
        }
        reconnectTimerRef.current = window.setTimeout(connect, 1500)
      }
    }

    void loadMessages()
    connect()

    return () => {
      mounted = false
      if (sourceRef.current) {
        sourceRef.current.close()
        sourceRef.current = null
      }
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current)
        reconnectTimerRef.current = null
      }
    }
  }, [fetchMessages, selectedRoomId, upsertMessage])

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
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to send room message'
      setError(message)
    } finally {
      setPosting(false)
    }
  }, [draft, posting, selectedRoomId, upsertMessage])

  const handleCreateRoom = useCallback(async (event?: FormEvent<HTMLFormElement>) => {
    event?.preventDefault()
    if (!selectedTeamId || creatingRoom) {
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
  }, [creatingRoom, fetchRooms, newRoomName, selectedTeamId])

  useEffect(() => {
    if (createRoomRequestToken <= lastCreateRoomRequestTokenRef.current) {
      return
    }
    lastCreateRoomRequestTokenRef.current = createRoomRequestToken
    void handleCreateRoom()
  }, [createRoomRequestToken, handleCreateRoom])

  const handleDeleteRoom = useCallback(async (room: RoomEntry) => {
    if (!room?.id || deletingRoomId) {
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
  }, [deletingRoomId, fetchRooms, selectedRoomId, selectedTeamId])

  const handleSetLeader = useCallback(async (workerName: string) => {
    if (!workerName) return
    setSettingLeaderId(workerName)
    try {
      const resp = await fetch(`/api/openclaw/workers/${encodeURIComponent(workerName)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ roleKind: 'leader' }),
      })
      if (!resp.ok) {
        const body = await resp.text()
        throw new Error(body || `Failed to update leader (${resp.status})`)
      }
      await fetchTeamsAndWorkers()
      setError(null)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to set leader'
      setError(message)
    } finally {
      setSettingLeaderId(null)
    }
  }, [fetchTeamsAndWorkers])

  const handleInsertMention = useCallback((token: string) => {
    if (!token) return
    setDraft(previous => {
      const base = previous.trim().length === 0 ? '' : `${previous}${previous.endsWith(' ') ? '' : ' '}`
      const next = `${base}${token} `
      requestAnimationFrame(() => {
        const element = inputRef.current
        if (!element) return
        element.focus()
        element.setSelectionRange(next.length, next.length)
      })
      return next
    })
    setMentionAutocomplete(null)
  }, [])

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
        avatar: '🧑',
      }
    }

    if (message.senderType === 'system') {
      return {
        displayName: message.senderName || 'ClawOS',
        roleLabel: 'SYSTEM',
        avatar: '⚙️',
      }
    }

    const lookupByID = workerLookup.get(sanitizeLookupKey(message.senderId))
    const lookupByName = workerLookup.get(sanitizeLookupKey(message.senderName))
    const worker = lookupByID || lookupByName

    const displayName = worker?.agentName || message.senderName || message.senderId || 'Claw'
    const avatar = worker?.agentEmoji?.trim() || (message.senderType === 'leader' ? '🧠' : firstGlyph(displayName))

    return {
      displayName,
      roleLabel: message.senderType === 'leader' ? 'LEADER' : 'WORKER',
      avatar,
    }
  }, [workerLookup])

  if (loading) {
    return (
      <div className={styles.container}>
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
    <div className={styles.container}>
      {error && <div className={styles.error}>{error}</div>}

      <div className={styles.layout}>
        {isSidebarOpen && (
          <aside className={styles.sidebar}>
            <div className={styles.selectGroup}>
              <label className={styles.label} htmlFor="claw-team-select">Team</label>
              <select
                id="claw-team-select"
                className={styles.select}
                value={selectedTeamId}
                onChange={event => setSelectedTeamId(event.target.value)}
              >
                {teams.length === 0 && <option value="">No teams</option>}
                {teams.map(team => (
                  <option key={team.id} value={team.id}>
                    {team.name}
                  </option>
                ))}
              </select>
            </div>

            <div className={styles.roomsHeader}>
              <div>
                <div className={styles.roomsTitle}>Rooms</div>
                <div className={styles.roomsSubtitle}>{selectedTeam?.name || 'Select a team'}</div>
              </div>
            </div>

            <form className={styles.createRoomForm} onSubmit={event => void handleCreateRoom(event)}>
              <input
                type="text"
                className={styles.createRoomInput}
                value={newRoomName}
                onChange={event => setNewRoomName(event.target.value)}
                placeholder="New room name (optional)"
                disabled={!selectedTeamId || creatingRoom}
              />
              <button
                type="submit"
                className={styles.createRoomButton}
                disabled={!selectedTeamId || creatingRoom}
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
                      onClick={() => setSelectedRoomId(room.id)}
                      onKeyDown={event => {
                        if (event.key === 'Enter' || event.key === ' ') {
                          event.preventDefault()
                          setSelectedRoomId(room.id)
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
                          void handleDeleteRoom(room)
                        }}
                        disabled={deletingRoomId === room.id}
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

            <div className={styles.mentionsHint}>
              Mention hints: {mentionHints.join(' ')}
            </div>
          </aside>
        )}

        <section className={styles.chatPanel}>
          <header className={styles.chatHeader}>
            <div className={styles.chatTitleWrap}>
              <h3 className={styles.chatTitle}>{selectedRoom?.name || 'No room selected'}</h3>
              <span className={styles.chatSubtitle}>{selectedTeam?.name || 'No team selected'}</span>
            </div>

            <div className={styles.metaGrid}>
              <div className={styles.metaCard}>
                <span className={styles.metaLabel}>Team Brief</span>
                <span className={styles.metaValue}>{selectedTeam?.name || 'No team selected'}</span>
                <span className={styles.metaSubtle}>
                  {selectedRoom ? `Room · ${selectedRoom.name}` : 'Create or select a room to start'}
                </span>
                {(selectedTeam?.role || selectedTeam?.vibe) && (
                  <div className={styles.metaInline}>
                    {selectedTeam?.role && <span className={styles.metaPill}>{selectedTeam.role}</span>}
                    {selectedTeam?.vibe && <span className={styles.metaPill}>{selectedTeam.vibe}</span>}
                  </div>
                )}
                <div className={styles.metaBrief}>{teamBriefText}</div>
              </div>

              <div className={styles.metaCard}>
                <span className={styles.metaLabel}>Leader</span>
                {leaderWorker ? (
                  <>
                    <span className={styles.metaValue}>
                      {leaderWorker.agentEmoji || '🧠'} {leaderWorker.agentName || leaderWorker.name}
                    </span>
                    <span className={styles.metaSubtle}>
                      <code>@leader</code> alias · <code>@{leaderWorker.name}</code>
                    </span>
                    <div className={styles.metaInline}>
                      <span className={styles.metaPill}>{leaderRoleText}</span>
                      <span className={styles.metaPill}>{leaderVibeText}</span>
                    </div>
                  </>
                ) : (
                  <>
                    <span className={styles.metaValue}>Leader not set</span>
                    <span className={styles.metaSubtle}>Use member chips below to assign one</span>
                  </>
                )}
              </div>

              <div className={styles.metaCard}>
                <span className={styles.metaLabel}>Members</span>
                <span className={styles.metaValue}>
                  {teamWorkers.length} {teamWorkers.length === 1 ? 'claw' : 'claws'}
                </span>
                <span className={styles.metaSubtle}>
                  {leaderWorker
                    ? `${Math.max(teamWorkers.length - 1, 0)} worker${teamWorkers.length - 1 === 1 ? '' : 's'} + 1 leader`
                    : 'No leader assigned yet'}
                </span>
                {quickMentionTokens.length > 0 && (
                  <div className={styles.quickMentionRow}>
                    {quickMentionTokens.map(token => (
                      <button
                        key={token}
                        type="button"
                        className={styles.quickMentionButton}
                        onClick={() => handleInsertMention(token)}
                      >
                        {token}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div className={styles.participantStrip}>
              {teamWorkers.length === 0 ? (
                <span className={styles.emptyParticipants}>No workers in this team yet</span>
              ) : (
                teamWorkers.map(worker => {
                  const isLeader = selectedTeam?.leaderId === worker.name || roleLabel(worker.roleKind) === 'leader'
                  const display = worker.agentName || worker.name
                  return (
                    <div key={worker.name} className={styles.participantChip}>
                      <span className={styles.participantAvatar}>{worker.agentEmoji || firstGlyph(display)}</span>
                      <span className={styles.participantName}>@{worker.name}</span>
                      <span className={isLeader ? styles.leaderBadge : styles.workerBadge}>
                        {isLeader ? 'leader' : 'worker'}
                      </span>
                      {!isLeader && (
                        <button
                          type="button"
                          className={styles.leaderAction}
                          onClick={() => void handleSetLeader(worker.name)}
                          disabled={settingLeaderId === worker.name}
                          title="Set as leader"
                        >
                          {settingLeaderId === worker.name ? '...' : '👑'}
                        </button>
                      )}
                    </div>
                  )
                })
              )}
            </div>

            <div className={styles.teamGuide}>
              Collaboration tip: start with <code>@leader</code> for delegation; members should report progress back via <code>@leader</code> or @{leaderWorker?.name || 'leader'}.
            </div>
          </header>

          <div className={styles.messages}>
            {!selectedRoomId ? (
              <div className={styles.stateHint}>Select a room from the left panel.</div>
            ) : messages.length === 0 ? (
              <div className={styles.stateHint}>No messages yet. Start the conversation.</div>
            ) : (
              messages.map(message => {
                const isUser = message.senderType === 'user'
                const isSystem = message.senderType === 'system'
                const senderVisual = resolveSenderVisual(message)
                return (
                  <div
                    key={message.id}
                    className={`${styles.messageRow} ${isUser ? styles.messageRowUser : styles.messageRowAgent}`}
                  >
                    <div className={styles.messageAvatar}>{senderVisual.avatar}</div>
                    <div className={styles.messageMain}>
                      <div className={styles.messageMeta}>
                        <span className={styles.senderName}>{senderVisual.displayName}</span>
                        <span className={styles.senderType}>{senderVisual.roleLabel}</span>
                        <span className={styles.timestamp}>{formatMessageTime(message.createdAt)}</span>
                      </div>
                      <div
                        className={`${styles.messageBubble} ${isUser ? styles.messageBubbleUser : styles.messageBubbleAgent} ${isSystem ? styles.messageBubbleSystem : ''}`}
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
                  placeholder="Type message... use @leader to assign/report, or @worker-name"
                  rows={2}
                  disabled={!selectedRoomId || posting}
                />
                <button
                  type="button"
                  className={styles.sendButton}
                  onClick={() => void handleSend()}
                  disabled={!selectedRoomId || posting || !draft.trim()}
                >
                  {posting ? '…' : '➤'}
                </button>
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
