package handlers

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"
)

type ClawRoomEntry struct {
	ID        string `json:"id"`
	TeamID    string `json:"teamId"`
	Name      string `json:"name"`
	CreatedAt string `json:"createdAt"`
	UpdatedAt string `json:"updatedAt"`
}

type ClawRoomMessage struct {
	ID         string            `json:"id"`
	RoomID     string            `json:"roomId"`
	TeamID     string            `json:"teamId"`
	SenderType string            `json:"senderType"`
	SenderID   string            `json:"senderId,omitempty"`
	SenderName string            `json:"senderName"`
	Content    string            `json:"content"`
	Mentions   []string          `json:"mentions,omitempty"`
	CreatedAt  string            `json:"createdAt"`
	Metadata   map[string]string `json:"metadata,omitempty"`
}

type clawRoomPayload struct {
	ID     string `json:"id"`
	TeamID string `json:"teamId"`
	Name   string `json:"name"`
}

type clawRoomMessagePayload struct {
	Content    string `json:"content"`
	SenderType string `json:"senderType,omitempty"`
	SenderID   string `json:"senderId,omitempty"`
	SenderName string `json:"senderName,omitempty"`
}

type clawRoomStreamEvent struct {
	Type    string           `json:"type"`
	RoomID  string           `json:"roomId"`
	Message *ClawRoomMessage `json:"message,omitempty"`
}

var roomMentionPattern = regexp.MustCompile(`@([a-zA-Z0-9_.-]+)`)

const (
	roomAutomationProcessedAtKey = "automationProcessedAt"
	roomIDDynamicSuffixBytes     = 2
	roomIDDynamicMaxAttempts     = 12
)

func (h *OpenClawHandler) loadRooms() ([]ClawRoomEntry, error) {
	lines, err := h.wf.ListOpenClawRoomJSON()
	if err != nil {
		return nil, err
	}
	out := make([]ClawRoomEntry, 0, len(lines))
	for _, line := range lines {
		var r ClawRoomEntry
		if err := json.Unmarshal([]byte(line), &r); err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, nil
}

func (h *OpenClawHandler) saveRooms(rooms []ClawRoomEntry) error {
	rows := make([][2]string, 0, len(rooms))
	for i := range rooms {
		b, err := json.Marshal(rooms[i])
		if err != nil {
			return err
		}
		if strings.TrimSpace(rooms[i].ID) == "" {
			return fmt.Errorf("room entry missing id")
		}
		rows = append(rows, [2]string{rooms[i].ID, string(b)})
	}
	return h.wf.ReplaceOpenClawRooms(rows)
}

func (h *OpenClawHandler) loadRoomMessages(roomID string) ([]ClawRoomMessage, error) {
	lines, err := h.wf.ListOpenClawRoomMessages(roomID)
	if err != nil {
		return nil, err
	}
	out := make([]ClawRoomMessage, 0, len(lines))
	for _, line := range lines {
		var m ClawRoomMessage
		if err := json.Unmarshal([]byte(line), &m); err != nil {
			return nil, err
		}
		out = append(out, m)
	}
	return out, nil
}

func (h *OpenClawHandler) saveRoomMessages(roomID string, messages []ClawRoomMessage) error {
	rows := make([][2]string, 0, len(messages))
	for i := range messages {
		if strings.TrimSpace(messages[i].ID) == "" {
			return fmt.Errorf("room message missing id")
		}
		b, err := json.Marshal(messages[i])
		if err != nil {
			return err
		}
		rows = append(rows, [2]string{messages[i].ID, string(b)})
	}
	return h.wf.ReplaceOpenClawRoomMessages(roomID, rows)
}

func findRoomByID(rooms []ClawRoomEntry, roomID string) *ClawRoomEntry {
	for i := range rooms {
		if rooms[i].ID == roomID {
			return &rooms[i]
		}
	}
	return nil
}

func defaultRoomNameForTeam(teamName string) string {
	trimmed := strings.TrimSpace(teamName)
	if trimmed == "" {
		return "Team Room"
	}
	return fmt.Sprintf("%s Room", trimmed)
}

func defaultRoomIDForTeam(teamID string) string {
	return sanitizeRoomID("team-" + teamID)
}

func buildRoomIDWithDynamicSuffix(base string) string {
	normalizedBase := sanitizeRoomID(base)
	if normalizedBase == "" {
		normalizedBase = "room"
	}

	suffix := sanitizeRoomID(generateToken(roomIDDynamicSuffixBytes))
	if suffix == "" {
		suffix = strconv.FormatInt(time.Now().UTC().UnixNano()%1_000_000, 10)
	}
	return sanitizeRoomID(fmt.Sprintf("%s-%s", normalizedBase, suffix))
}

func nextAvailableRoomID(base string, rooms []ClawRoomEntry) string {
	normalizedBase := sanitizeRoomID(base)
	if normalizedBase == "" {
		normalizedBase = "room"
	}
	for attempt := 0; attempt < roomIDDynamicMaxAttempts; attempt++ {
		candidate := buildRoomIDWithDynamicSuffix(normalizedBase)
		if candidate != "" && findRoomByID(rooms, candidate) == nil {
			return candidate
		}
	}

	fallback := generateRoomEntityID(normalizedBase)
	if findRoomByID(rooms, fallback) == nil {
		return fallback
	}
	return generateRoomEntityID("room")
}

func (h *OpenClawHandler) ensureDefaultRoomLocked(team TeamEntry) (ClawRoomEntry, error) {
	rooms, err := h.loadRooms()
	if err != nil {
		return ClawRoomEntry{}, err
	}
	for _, room := range rooms {
		if room.TeamID == team.ID {
			return room, nil
		}
	}
	now := time.Now().UTC().Format(time.RFC3339)
	created := ClawRoomEntry{
		ID:        defaultRoomIDForTeam(team.ID),
		TeamID:    team.ID,
		Name:      defaultRoomNameForTeam(team.Name),
		CreatedAt: now,
		UpdatedAt: now,
	}
	rooms = append(rooms, created)
	sort.Slice(rooms, func(i, j int) bool { return rooms[i].Name < rooms[j].Name })
	if err := h.saveRooms(rooms); err != nil {
		return ClawRoomEntry{}, err
	}
	return created, nil
}

func (h *OpenClawHandler) deleteRoomsForTeamLocked(teamID string) error {
	rooms, err := h.loadRooms()
	if err != nil {
		return err
	}
	filtered := rooms[:0]
	removedRoomIDs := make([]string, 0)
	for _, room := range rooms {
		if room.TeamID == teamID {
			removedRoomIDs = append(removedRoomIDs, room.ID)
			continue
		}
		filtered = append(filtered, room)
	}
	if len(removedRoomIDs) == 0 {
		return nil
	}
	if err := h.saveRooms(filtered); err != nil {
		return err
	}
	if err := h.wf.DeleteOpenClawMessagesForRooms(removedRoomIDs); err != nil {
		return err
	}
	for _, roomID := range removedRoomIDs {
		_ = os.Remove(h.roomMessagesPath(roomID))
		h.roomSSEClients.Delete(roomID)
		h.roomSSELastEvent.Delete(roomID)
		h.roomAutomationMu.Delete(roomID)
	}
	return nil
}

func normalizeRoomSenderType(raw string) string {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "leader":
		return "leader"
	case "worker":
		return "worker"
	case "system":
		return "system"
	default:
		return "user"
	}
}

func extractMentions(content string) []string {
	seen := map[string]bool{}
	mentions := make([]string, 0)
	for _, match := range roomMentionPattern.FindAllStringSubmatch(content, -1) {
		if len(match) < 2 {
			continue
		}
		token := strings.ToLower(strings.TrimSpace(match[1]))
		if token == "" || seen[token] {
			continue
		}
		seen[token] = true
		mentions = append(mentions, token)
	}
	return mentions
}

func generateRoomEntityID(prefix string) string {
	return fmt.Sprintf("%s-%d-%s", prefix, time.Now().UTC().UnixNano(), generateToken(3))
}

func newRoomMessage(room ClawRoomEntry, senderType, senderID, senderName, content string, metadata map[string]string) ClawRoomMessage {
	if strings.TrimSpace(senderName) == "" {
		senderName = "Unknown"
	}
	return ClawRoomMessage{
		ID:         generateRoomEntityID("room-msg"),
		RoomID:     room.ID,
		TeamID:     room.TeamID,
		SenderType: normalizeRoomSenderType(senderType),
		SenderID:   strings.TrimSpace(senderID),
		SenderName: strings.TrimSpace(senderName),
		Content:    strings.TrimSpace(content),
		Mentions:   extractMentions(content),
		CreatedAt:  time.Now().UTC().Format(time.RFC3339),
		Metadata:   metadata,
	}
}

func cloneRoomEventWithMessage(event clawRoomStreamEvent) clawRoomStreamEvent {
	if event.Message == nil {
		return event
	}
	copyMsg := *event.Message
	event.Message = &copyMsg
	return event
}

func (h *OpenClawHandler) roomClientMap(roomID string) *sync.Map {
	if existing, ok := h.roomSSEClients.Load(roomID); ok {
		return existing.(*sync.Map)
	}
	clients := &sync.Map{}
	actual, _ := h.roomSSEClients.LoadOrStore(roomID, clients)
	return actual.(*sync.Map)
}

func (h *OpenClawHandler) publishRoomEvent(roomID string, event clawRoomStreamEvent) {
	event.RoomID = roomID
	h.roomSSELastEvent.Store(roomID, cloneRoomEventWithMessage(event))

	clients := h.roomClientMap(roomID)
	clients.Range(func(_, value any) bool {
		ch, ok := value.(chan clawRoomStreamEvent)
		if !ok {
			return true
		}
		select {
		case ch <- cloneRoomEventWithMessage(event):
		default:
		}
		return true
	})
}

func writeSSE(w http.ResponseWriter, flusher http.Flusher, eventName string, payload any) {
	data, err := json.Marshal(payload)
	if err != nil {
		log.Printf("openclaw: failed to marshal room SSE payload: %v", err)
		return
	}
	_, _ = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", eventName, data)
	flusher.Flush()
}

func (h *OpenClawHandler) appendRoomMessage(roomID string, message ClawRoomMessage) error {
	if strings.TrimSpace(message.ID) == "" {
		return fmt.Errorf("room message missing id")
	}
	b, err := json.Marshal(message)
	if err != nil {
		return err
	}
	if err := h.wf.AppendOpenClawRoomMessage(roomID, message.ID, string(b)); err != nil {
		return err
	}

	// Broadcast to WebSocket clients (also handles SSE backward compatibility)
	h.publishRoomWSEvent(roomID, WSOutboundMessage{
		Type:    WSTypeNewMessage,
		Message: &message,
	})

	// Keep SSE event for backward compatibility
	h.publishRoomEvent(roomID, clawRoomStreamEvent{Type: "message", Message: &message})
	return nil
}

func teamWorkers(entries []ContainerEntry, teamID string) []ContainerEntry {
	workers := make([]ContainerEntry, 0)
	for _, entry := range entries {
		if entry.TeamID != teamID {
			continue
		}
		entry.RoleKind = normalizeRoleKind(entry.RoleKind)
		workers = append(workers, enrichContainerIdentity(entry))
	}
	sort.Slice(workers, func(i, j int) bool { return workers[i].Name < workers[j].Name })
	return workers
}

func resolveTeamLeader(team TeamEntry, workers []ContainerEntry) *ContainerEntry {
	leaderID := strings.TrimSpace(team.LeaderID)
	if leaderID != "" {
		for i := range workers {
			if workers[i].Name == leaderID {
				return &workers[i]
			}
		}
	}
	for i := range workers {
		if normalizeRoleKind(workers[i].RoleKind) == "leader" {
			return &workers[i]
		}
	}
	return nil
}

func workerAliases(worker ContainerEntry) []string {
	aliases := []string{strings.ToLower(strings.TrimSpace(worker.Name))}
	if alias := strings.ToLower(strings.TrimSpace(sanitizeRoomID(worker.AgentName))); alias != "" {
		aliases = append(aliases, alias)
	}
	return aliases
}

func resolveMentionTargetsWithFallback(
	mentions []string,
	team TeamEntry,
	workers []ContainerEntry,
	defaultToLeader bool,
) []ContainerEntry {
	if len(workers) == 0 {
		return nil
	}

	leader := resolveTeamLeader(team, workers)
	lookup := map[string]ContainerEntry{}
	for _, worker := range workers {
		for _, alias := range workerAliases(worker) {
			lookup[alias] = worker
		}
	}

	picked := map[string]ContainerEntry{}
	for _, mention := range mentions {
		token := strings.ToLower(strings.TrimSpace(mention))
		if token == "" {
			continue
		}
		if token == "all" {
			for _, worker := range workers {
				picked[worker.Name] = worker
			}
			continue
		}
		if token == "leader" {
			if leader != nil {
				picked[leader.Name] = *leader
			}
			continue
		}
		if worker, ok := lookup[token]; ok {
			picked[worker.Name] = worker
		}
	}

	if len(picked) == 0 && defaultToLeader && leader != nil {
		picked[leader.Name] = *leader
	}

	out := make([]ContainerEntry, 0, len(picked))
	for _, worker := range picked {
		out = append(out, worker)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

func stripLeadingMentions(content string) string {
	rawRunes := []rune(strings.TrimSpace(content))
	if len(rawRunes) == 0 {
		return ""
	}

	// Remove addressing prefixes like:
	// "@a @b task", "@leader，安排一下", "@x,@y do this"
	i := 0
	consumedMention := false
	for i < len(rawRunes) {
		if rawRunes[i] != '@' {
			break
		}

		j := i + 1
		for j < len(rawRunes) {
			ch := rawRunes[j]
			if unicode.IsLetter(ch) || unicode.IsDigit(ch) || ch == '_' || ch == '.' || ch == '-' {
				j++
				continue
			}
			break
		}

		// Invalid mention token, keep original content.
		if j == i+1 {
			return string(rawRunes)
		}

		consumedMention = true
		i = j
		for i < len(rawRunes) {
			ch := rawRunes[i]
			if unicode.IsSpace(ch) || ch == ',' || ch == ';' || ch == '，' || ch == '；' || ch == '、' {
				i++
				continue
			}
			break
		}
	}

	if !consumedMention {
		return string(rawRunes)
	}

	trimmed := strings.TrimSpace(string(rawRunes[i:]))
	if trimmed == "" {
		return string(rawRunes)
	}
	return trimmed
}

func buildTeamMentionGuide(team TeamEntry, workers []ContainerEntry, self ContainerEntry) string {
	if len(workers) == 0 {
		return "No teammates registered."
	}

	sorted := append([]ContainerEntry(nil), workers...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Name < sorted[j].Name })

	leader := resolveTeamLeader(team, workers)
	lines := make([]string, 0, len(sorted)+5)
	if leader != nil {
		leaderRole := strings.TrimSpace(leader.AgentRole)
		if leaderRole == "" {
			leaderRole = "leader"
		}
		lines = append(
			lines,
			fmt.Sprintf(
				"Leader aliases: @leader and @%s = %s (%s)",
				leader.Name,
				workerDisplayName(*leader),
				leaderRole,
			),
		)
		if leader.Name == self.Name {
			lines = append(lines, "You are the leader. Delegate with @worker-id mentions and keep the team aligned.")
			lines = append(lines, "Hard rule: do not delegate with @mentions until the user gives an explicit executable task.")
		} else {
			lines = append(
				lines,
				fmt.Sprintf("Your leader is @leader (same as @%s).", leader.Name),
			)
			lines = append(lines, "Hard rule: workers cannot use @mentions. Report progress in plain text without @leader or @worker-id.")
		}
	} else {
		lines = append(lines, "No leader is assigned yet. Coordinate directly with @worker-id aliases.")
	}

	lines = append(lines, "Team member aliases for delegation:")
	for _, member := range sorted {
		roleKind := normalizeRoleKind(member.RoleKind)
		if roleKind != "leader" {
			roleKind = "worker"
		}
		roleText := strings.TrimSpace(member.AgentRole)
		if roleText == "" {
			roleText = roleKind
		}
		displayName := workerDisplayName(member)
		line := fmt.Sprintf("- @%s = %s (%s)", member.Name, displayName, roleText)
		if member.Name == self.Name {
			line += " [you]"
		}
		lines = append(lines, line)
	}
	lines = append(lines, "Only leader can delegate with @worker-id, and only after explicit user task confirmation.")
	return strings.Join(lines, "\n")
}

type openAIChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

func (h *OpenClawHandler) RoomsHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			teamID := sanitizeTeamID(r.URL.Query().Get("teamId"))

			if teamID != "" {
				h.mu.Lock()
				teams, err := h.loadTeams()
				if err == nil {
					if team := findTeamByID(teams, teamID); team != nil {
						if _, ensureErr := h.ensureDefaultRoomLocked(*team); ensureErr != nil {
							log.Printf("openclaw: failed to ensure default room for team %s: %v", teamID, ensureErr)
						}
					}
				}
				h.mu.Unlock()
			}

			h.mu.RLock()
			rooms, err := h.loadRooms()
			h.mu.RUnlock()
			if err != nil {
				writeJSONError(w, fmt.Sprintf("Failed to load rooms: %v", err), http.StatusInternalServerError)
				return
			}

			filtered := make([]ClawRoomEntry, 0, len(rooms))
			for _, room := range rooms {
				if teamID != "" && room.TeamID != teamID {
					continue
				}
				filtered = append(filtered, room)
			}
			sort.Slice(filtered, func(i, j int) bool { return filtered[i].Name < filtered[j].Name })
			w.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(w).Encode(filtered); err != nil {
				log.Printf("openclaw: rooms encode error: %v", err)
			}
		case http.MethodPost:
			if !h.canManageOpenClaw() {
				h.writeReadOnlyError(w)
				return
			}
			var req clawRoomPayload
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				writeJSONError(w, fmt.Sprintf("invalid request body: %v", err), http.StatusBadRequest)
				return
			}
			teamID := sanitizeTeamID(req.TeamID)
			if teamID == "" {
				writeJSONError(w, "teamId is required", http.StatusBadRequest)
				return
			}
			roomName := strings.TrimSpace(req.Name)
			if roomName == "" {
				roomName = defaultRoomNameForTeam(teamID)
			}
			requestedRoomID := sanitizeRoomID(req.ID)

			h.mu.Lock()
			defer h.mu.Unlock()
			teams, err := h.loadTeams()
			if err != nil {
				writeJSONError(w, fmt.Sprintf("Failed to load teams: %v", err), http.StatusInternalServerError)
				return
			}
			if findTeamByID(teams, teamID) == nil {
				writeJSONError(w, fmt.Sprintf("team %q not found", teamID), http.StatusNotFound)
				return
			}
			rooms, err := h.loadRooms()
			if err != nil {
				writeJSONError(w, fmt.Sprintf("Failed to load rooms: %v", err), http.StatusInternalServerError)
				return
			}

			roomID := requestedRoomID
			if roomID == "" {
				baseRoomID := sanitizeRoomID(roomName)
				if baseRoomID == "" {
					baseRoomID = defaultRoomIDForTeam(teamID)
				}
				roomID = nextAvailableRoomID(baseRoomID, rooms)
			}
			if roomID == "" {
				roomID = generateRoomEntityID("room")
			}

			if requestedRoomID != "" && findRoomByID(rooms, roomID) != nil {
				writeJSONError(w, fmt.Sprintf("room %q already exists", roomID), http.StatusConflict)
				return
			}
			if findRoomByID(rooms, roomID) != nil {
				roomID = nextAvailableRoomID(roomID, rooms)
			}
			if roomID == "" || findRoomByID(rooms, roomID) != nil {
				writeJSONError(w, "failed to generate unique room id", http.StatusInternalServerError)
				return
			}

			now := time.Now().UTC().Format(time.RFC3339)
			created := ClawRoomEntry{ID: roomID, TeamID: teamID, Name: roomName, CreatedAt: now, UpdatedAt: now}
			rooms = append(rooms, created)
			sort.Slice(rooms, func(i, j int) bool { return rooms[i].Name < rooms[j].Name })
			if err := h.saveRooms(rooms); err != nil {
				writeJSONError(w, fmt.Sprintf("Failed to save rooms: %v", err), http.StatusInternalServerError)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusCreated)
			if err := json.NewEncoder(w).Encode(created); err != nil {
				log.Printf("openclaw: create room encode error: %v", err)
			}
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	}
}

func (h *OpenClawHandler) RoomByIDHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		rest := strings.Trim(strings.TrimPrefix(r.URL.Path, "/api/openclaw/rooms/"), "/")
		if rest == "" {
			writeJSONError(w, "room id required in path", http.StatusBadRequest)
			return
		}

		parts := strings.Split(rest, "/")
		roomID := sanitizeRoomID(parts[0])
		if roomID == "" {
			writeJSONError(w, "room id is invalid", http.StatusBadRequest)
			return
		}

		if len(parts) == 1 {
			switch r.Method {
			case http.MethodGet:
				h.mu.RLock()
				rooms, err := h.loadRooms()
				h.mu.RUnlock()
				if err != nil {
					writeJSONError(w, fmt.Sprintf("Failed to load rooms: %v", err), http.StatusInternalServerError)
					return
				}
				room := findRoomByID(rooms, roomID)
				if room == nil {
					writeJSONError(w, "room not found", http.StatusNotFound)
					return
				}
				w.Header().Set("Content-Type", "application/json")
				if err := json.NewEncoder(w).Encode(room); err != nil {
					log.Printf("openclaw: room encode error: %v", err)
				}
			case http.MethodDelete:
				if !h.canManageOpenClaw() {
					h.writeReadOnlyError(w)
					return
				}

				h.mu.Lock()
				rooms, err := h.loadRooms()
				if err != nil {
					h.mu.Unlock()
					writeJSONError(w, fmt.Sprintf("Failed to load rooms: %v", err), http.StatusInternalServerError)
					return
				}

				index := -1
				var removed ClawRoomEntry
				for i := range rooms {
					if rooms[i].ID == roomID {
						index = i
						removed = rooms[i]
						break
					}
				}
				if index < 0 {
					h.mu.Unlock()
					writeJSONError(w, "room not found", http.StatusNotFound)
					return
				}

				rooms = append(rooms[:index], rooms[index+1:]...)
				if err := h.saveRooms(rooms); err != nil {
					h.mu.Unlock()
					writeJSONError(w, fmt.Sprintf("Failed to save rooms: %v", err), http.StatusInternalServerError)
					return
				}

				_ = os.Remove(h.roomMessagesPath(roomID))
				h.roomSSEClients.Delete(roomID)
				h.roomSSELastEvent.Delete(roomID)
				h.roomAutomationMu.Delete(roomID)
				h.mu.Unlock()

				w.Header().Set("Content-Type", "application/json")
				if err := json.NewEncoder(w).Encode(map[string]any{
					"deleted": true,
					"roomId":  roomID,
					"room":    removed,
				}); err != nil {
					log.Printf("openclaw: delete room encode error: %v", err)
				}
			default:
				http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			}
			return
		}

		sub := strings.ToLower(strings.TrimSpace(parts[1]))
		switch sub {
		case "messages":
			h.handleRoomMessages(w, r, roomID)
		case "stream":
			h.handleRoomStream(w, r, roomID)
		case "ws":
			h.handleRoomWebSocket(w, r, roomID)
		default:
			http.NotFound(w, r)
		}
	}
}

func (h *OpenClawHandler) handleRoomMessages(w http.ResponseWriter, r *http.Request, roomID string) {
	switch r.Method {
	case http.MethodGet:
		h.mu.RLock()
		rooms, roomErr := h.loadRooms()
		messages, msgErr := h.loadRoomMessages(roomID)
		h.mu.RUnlock()
		if roomErr != nil {
			writeJSONError(w, fmt.Sprintf("Failed to load rooms: %v", roomErr), http.StatusInternalServerError)
			return
		}
		if msgErr != nil {
			writeJSONError(w, fmt.Sprintf("Failed to load room messages: %v", msgErr), http.StatusInternalServerError)
			return
		}
		if findRoomByID(rooms, roomID) == nil {
			writeJSONError(w, "room not found", http.StatusNotFound)
			return
		}

		limit := 200
		if raw := strings.TrimSpace(r.URL.Query().Get("limit")); raw != "" {
			if n, err := strconv.Atoi(raw); err == nil && n > 0 {
				if n > 1000 {
					n = 1000
				}
				limit = n
			}
		}
		if len(messages) > limit {
			messages = messages[len(messages)-limit:]
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(messages); err != nil {
			log.Printf("openclaw: room messages encode error: %v", err)
		}
	case http.MethodPost:
		if !h.canSendRoomMessages() {
			h.writeReadOnlyError(w)
			return
		}
		var req clawRoomMessagePayload
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSONError(w, fmt.Sprintf("invalid request body: %v", err), http.StatusBadRequest)
			return
		}
		content := strings.TrimSpace(req.Content)
		if content == "" {
			writeJSONError(w, "content is required", http.StatusBadRequest)
			return
		}

		h.mu.RLock()
		rooms, err := h.loadRooms()
		h.mu.RUnlock()
		if err != nil {
			writeJSONError(w, fmt.Sprintf("Failed to load rooms: %v", err), http.StatusInternalServerError)
			return
		}
		room := findRoomByID(rooms, roomID)
		if room == nil {
			writeJSONError(w, "room not found", http.StatusNotFound)
			return
		}

		senderType := normalizeRoomSenderType(req.SenderType)
		if senderType != "user" && senderType != "leader" && senderType != "worker" && senderType != "system" {
			senderType = "user"
		}
		senderName := strings.TrimSpace(req.SenderName)
		if senderName == "" {
			switch senderType {
			case "leader":
				senderName = "Leader"
			case "worker":
				senderName = "Worker"
			case "system":
				senderName = "System"
			default:
				senderName = "You"
			}
		}
		senderID := strings.TrimSpace(req.SenderID)
		if senderID == "" {
			switch senderType {
			case "user":
				senderID = "playground-user"
			case "leader", "worker":
				senderID = sanitizeContainerName(senderName)
			}
		}

		created := newRoomMessage(*room, senderType, senderID, senderName, content, nil)
		if err := h.appendRoomMessage(room.ID, created); err != nil {
			writeJSONError(w, fmt.Sprintf("Failed to save room message: %v", err), http.StatusInternalServerError)
			return
		}

		if senderType != "system" {
			go h.processRoomUserMessage(room.ID, created.ID)
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		if err := json.NewEncoder(w).Encode(created); err != nil {
			log.Printf("openclaw: room message encode error: %v", err)
		}
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

func (h *OpenClawHandler) handleRoomStream(w http.ResponseWriter, r *http.Request, roomID string) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	h.mu.RLock()
	rooms, err := h.loadRooms()
	h.mu.RUnlock()
	if err != nil {
		writeJSONError(w, fmt.Sprintf("Failed to load rooms: %v", err), http.StatusInternalServerError)
		return
	}
	if findRoomByID(rooms, roomID) == nil {
		writeJSONError(w, "room not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	clientID := generateRoomEntityID("room-client")
	clientChan := make(chan clawRoomStreamEvent, 16)
	clients := h.roomClientMap(roomID)
	clients.Store(clientID, clientChan)
	defer func() {
		clients.Delete(clientID)
		close(clientChan)
	}()

	writeSSE(w, flusher, "connected", map[string]string{"roomId": roomID})
	if lastAny, ok := h.roomSSELastEvent.Load(roomID); ok {
		if lastEvent, ok := lastAny.(clawRoomStreamEvent); ok {
			writeSSE(w, flusher, lastEvent.Type, lastEvent)
		}
	}

	heartbeat := time.NewTicker(15 * time.Second)
	defer heartbeat.Stop()

	ctx := r.Context()
	for {
		select {
		case <-ctx.Done():
			return
		case <-heartbeat.C:
			_, _ = fmt.Fprintf(w, ": heartbeat\n\n")
			flusher.Flush()
		case event, ok := <-clientChan:
			if !ok {
				return
			}
			writeSSE(w, flusher, event.Type, event)
		}
	}
}
