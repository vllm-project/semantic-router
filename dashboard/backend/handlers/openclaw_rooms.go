package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
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
	data, err := os.ReadFile(h.roomsPath())
	if err != nil {
		if os.IsNotExist(err) {
			return []ClawRoomEntry{}, nil
		}
		return nil, err
	}
	var rooms []ClawRoomEntry
	if err := json.Unmarshal(data, &rooms); err != nil {
		return nil, err
	}
	return rooms, nil
}

func (h *OpenClawHandler) saveRooms(rooms []ClawRoomEntry) error {
	data, err := json.MarshalIndent(rooms, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(h.roomsPath()), 0o755); err != nil {
		return err
	}
	return os.WriteFile(h.roomsPath(), data, 0o644)
}

func (h *OpenClawHandler) loadRoomMessages(roomID string) ([]ClawRoomMessage, error) {
	data, err := os.ReadFile(h.roomMessagesPath(roomID))
	if err != nil {
		if os.IsNotExist(err) {
			return []ClawRoomMessage{}, nil
		}
		return nil, err
	}
	var messages []ClawRoomMessage
	if err := json.Unmarshal(data, &messages); err != nil {
		return nil, err
	}
	return messages, nil
}

func (h *OpenClawHandler) saveRoomMessages(roomID string, messages []ClawRoomMessage) error {
	path := h.roomMessagesPath(roomID)
	data, err := json.MarshalIndent(messages, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
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
	h.mu.Lock()
	messages, err := h.loadRoomMessages(roomID)
	if err != nil {
		h.mu.Unlock()
		return err
	}
	messages = append(messages, message)
	if err := h.saveRoomMessages(roomID, messages); err != nil {
		h.mu.Unlock()
		return err
	}
	h.mu.Unlock()

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

func buildRoomTranscript(messages []ClawRoomMessage, limit int) string {
	if limit <= 0 {
		limit = 16
	}
	start := 0
	if len(messages) > limit {
		start = len(messages) - limit
	}
	lines := make([]string, 0, len(messages)-start)
	for i := start; i < len(messages); i++ {
		line := fmt.Sprintf("[%s] %s", strings.TrimSpace(messages[i].SenderName), strings.TrimSpace(messages[i].Content))
		lines = append(lines, line)
	}
	return strings.Join(lines, "\n")
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

type openAIChatRequest struct {
	Model    string              `json:"model"`
	Messages []openAIChatMessage `json:"messages"`
	Stream   bool                `json:"stream"`
	User     string              `json:"user,omitempty"`
}

type openAIChatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

const openClawPrimaryAgentModel = "openclaw:main"

var workerChatEndpointCandidates = []string{
	"/v1/chat/completions",
	"/api/openai/v1/chat/completions",
	"/api/router/v1/chat/completions",
}

func nestedObject(parent map[string]any, key string) map[string]any {
	if existing, ok := parent[key].(map[string]any); ok {
		return existing
	}
	created := map[string]any{}
	parent[key] = created
	return created
}

func enableGatewayEndpoint(config map[string]any, endpointKey string) bool {
	gateway := nestedObject(config, "gateway")
	httpCfg := nestedObject(gateway, "http")
	endpoints := nestedObject(httpCfg, "endpoints")
	endpoint := nestedObject(endpoints, endpointKey)
	if enabled, ok := endpoint["enabled"].(bool); ok && enabled {
		return false
	}
	endpoint["enabled"] = true
	return true
}

func (h *OpenClawHandler) workerConfigPath(worker ContainerEntry) string {
	if dataDir := strings.TrimSpace(worker.DataDir); dataDir != "" {
		return filepath.Join(dataDir, "openclaw.json")
	}
	return filepath.Join(h.containerDataDir(sanitizeContainerName(worker.Name)), "openclaw.json")
}

func (h *OpenClawHandler) ensureWorkerChatEndpoint(worker ContainerEntry) (bool, error) {
	if !h.canRepairWorkerChatEndpoint() {
		return false, nil
	}

	configPath := h.workerConfigPath(worker)
	data, err := os.ReadFile(configPath)
	if err != nil {
		return false, fmt.Errorf("unable to read worker config %q: %w", configPath, err)
	}

	var cfg map[string]any
	if err := json.Unmarshal(data, &cfg); err != nil {
		return false, fmt.Errorf("invalid worker config %q: %w", configPath, err)
	}

	changed := false
	changed = enableGatewayEndpoint(cfg, "chatCompletions") || changed
	changed = enableGatewayEndpoint(cfg, "responses") || changed
	if changed {
		updated, err := json.MarshalIndent(cfg, "", "  ")
		if err != nil {
			return false, fmt.Errorf("failed to marshal worker config update: %w", err)
		}
		if err := os.WriteFile(configPath, updated, 0o644); err != nil {
			return false, fmt.Errorf("failed to persist worker config update: %w", err)
		}
	}

	if _, err := h.containerCombinedOutput("restart", worker.Name); err != nil {
		if changed {
			return false, fmt.Errorf("worker restart failed after endpoint update: %w", err)
		}
		return false, fmt.Errorf("worker restart failed during endpoint recovery: %w", err)
	}

	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) {
		if h.gatewayReachable(worker.Name, worker.Port) {
			return true, nil
		}
		time.Sleep(500 * time.Millisecond)
	}
	return true, nil
}

func (h *OpenClawHandler) queryWorkerChatEndpoint(
	targetBase string,
	endpoint string,
	token string,
	payload openAIChatRequest,
) (string, int, string, error) {
	raw, err := json.Marshal(payload)
	if err != nil {
		return "", 0, "", err
	}

	url := strings.TrimRight(targetBase, "/") + endpoint
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(raw))
	if err != nil {
		return "", 0, "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("X-OpenClaw-Agent-Id", "main")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
		req.Header.Set("X-OpenClaw-Token", token)
	}

	client := &http.Client{Timeout: 300 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", 0, "", err
	}
	defer func() { _ = resp.Body.Close() }()
	body, _ := io.ReadAll(resp.Body)
	trimmedBody := strings.TrimSpace(string(body))

	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		if trimmedBody == "" {
			trimmedBody = resp.Status
		}
		return "", resp.StatusCode, trimmedBody, fmt.Errorf("worker chat request failed: %s", trimmedBody)
	}

	var parsed openAIChatResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return "", resp.StatusCode, trimmedBody, fmt.Errorf("invalid worker chat response: %w", err)
	}
	if parsed.Error != nil && strings.TrimSpace(parsed.Error.Message) != "" {
		return "", resp.StatusCode, trimmedBody, fmt.Errorf("%s", parsed.Error.Message)
	}
	if len(parsed.Choices) == 0 {
		return "", resp.StatusCode, trimmedBody, fmt.Errorf("worker returned no choices")
	}
	content := strings.TrimSpace(parsed.Choices[0].Message.Content)
	if content == "" {
		return "", resp.StatusCode, trimmedBody, fmt.Errorf("worker returned empty content")
	}
	return content, resp.StatusCode, trimmedBody, nil
}

func (h *OpenClawHandler) queryWorkerChat(worker ContainerEntry, systemPrompt, userPrompt string) (string, error) {
	targetBase, ok := h.TargetBaseForContainer(worker.Name)
	if !ok {
		return "", fmt.Errorf("worker %q is not registered", worker.Name)
	}
	token := strings.TrimSpace(h.GatewayTokenForContainer(worker.Name))

	payload := openAIChatRequest{
		Model:  openClawPrimaryAgentModel,
		Stream: false,
		User:   "team-room:" + sanitizeContainerName(worker.Name),
		Messages: []openAIChatMessage{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: userPrompt},
		},
	}

	attempt := func() (string, bool, error) {
		allEndpointMissing := true
		var lastErr error
		for _, endpoint := range workerChatEndpointCandidates {
			content, statusCode, body, err := h.queryWorkerChatEndpoint(targetBase, endpoint, token, payload)
			if err == nil {
				return content, false, nil
			}
			// 404/405 both indicate the chat API endpoint is not active/available on the gateway.
			allEndpointMissing = allEndpointMissing && (statusCode == http.StatusNotFound || statusCode == http.StatusMethodNotAllowed)

			detail := strings.TrimSpace(body)
			if detail == "" {
				detail = err.Error()
			}
			lastErr = fmt.Errorf("worker chat via %s failed: %s", endpoint, detail)
		}
		if lastErr == nil {
			lastErr = fmt.Errorf("worker chat request failed for all candidate endpoints")
		}
		return "", allEndpointMissing, lastErr
	}

	content, allEndpointMissing, err := attempt()
	if err == nil {
		return content, nil
	}
	if !allEndpointMissing {
		return "", err
	}

	recovered, ensureErr := h.ensureWorkerChatEndpoint(worker)
	if ensureErr != nil {
		return "", fmt.Errorf("%w; automatic endpoint repair failed: %w", err, ensureErr)
	}
	if !recovered {
		return "", fmt.Errorf(
			"%w; worker endpoint recovery skipped (read-only mode). ensure gateway.http.endpoints.chatCompletions.enabled=true in %s",
			err,
			h.workerConfigPath(worker),
		)
	}

	content, _, retryErr := attempt()
	if retryErr != nil {
		return "", fmt.Errorf("%w; retry after endpoint repair failed: %w", err, retryErr)
	}
	return content, nil
}

func workerDisplayName(worker ContainerEntry) string {
	if name := strings.TrimSpace(worker.AgentName); name != "" {
		return name
	}
	return worker.Name
}

func (h *OpenClawHandler) roomAutomationLock(roomID string) *sync.Mutex {
	if existing, ok := h.roomAutomationMu.Load(roomID); ok {
		return existing.(*sync.Mutex)
	}
	lock := &sync.Mutex{}
	actual, _ := h.roomAutomationMu.LoadOrStore(roomID, lock)
	return actual.(*sync.Mutex)
}

func roomMessageAutomationProcessed(message ClawRoomMessage) bool {
	if message.Metadata == nil {
		return false
	}
	return strings.TrimSpace(message.Metadata[roomAutomationProcessedAtKey]) != ""
}

func (h *OpenClawHandler) markRoomMessageAutomationProcessed(roomID, messageID string) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	messages, err := h.loadRoomMessages(roomID)
	if err != nil {
		return err
	}

	changed := false
	for i := range messages {
		if messages[i].ID != messageID {
			continue
		}
		if roomMessageAutomationProcessed(messages[i]) {
			return nil
		}
		if messages[i].Metadata == nil {
			messages[i].Metadata = map[string]string{}
		}
		messages[i].Metadata[roomAutomationProcessedAtKey] = time.Now().UTC().Format(time.RFC3339Nano)
		changed = true
		break
	}

	if !changed {
		return nil
	}
	return h.saveRoomMessages(roomID, messages)
}

func (h *OpenClawHandler) processRoomUserMessage(roomID string, triggerMessageID string) {
	lock := h.roomAutomationLock(roomID)
	lock.Lock()
	defer lock.Unlock()

	const maxAutomationTurns = 24
	queue := []string{strings.TrimSpace(triggerMessageID)}
	seen := map[string]bool{}
	turns := 0

	for len(queue) > 0 && turns < maxAutomationTurns {
		currentID := strings.TrimSpace(queue[0])
		queue = queue[1:]
		if currentID == "" || seen[currentID] {
			continue
		}
		seen[currentID] = true
		turns++

		h.mu.RLock()
		rooms, roomErr := h.loadRooms()
		teams, teamErr := h.loadTeams()
		entries, entryErr := h.loadRegistry()
		messages, msgErr := h.loadRoomMessages(roomID)
		h.mu.RUnlock()
		if roomErr != nil || teamErr != nil || entryErr != nil || msgErr != nil {
			log.Printf(
				"openclaw: room automation prefetch failed room=%s roomErr=%v teamErr=%v entryErr=%v msgErr=%v",
				roomID,
				roomErr,
				teamErr,
				entryErr,
				msgErr,
			)
			return
		}

		room := findRoomByID(rooms, roomID)
		if room == nil {
			return
		}
		team := findTeamByID(teams, room.TeamID)
		if team == nil {
			return
		}

		triggerIndex := -1
		for i := range messages {
			if messages[i].ID == currentID {
				triggerIndex = i
				break
			}
		}
		if triggerIndex < 0 {
			continue
		}

		trigger := messages[triggerIndex]
		senderType := normalizeRoomSenderType(trigger.SenderType)
		if senderType == "system" {
			continue
		}
		if roomMessageAutomationProcessed(trigger) {
			continue
		}
		if err := h.markRoomMessageAutomationProcessed(roomID, trigger.ID); err != nil {
			log.Printf("openclaw: failed to mark room message as processed room=%s message=%s err=%v", roomID, trigger.ID, err)
		}
		if senderType == "worker" {
			// Hard policy: worker mentions are non-routable.
			continue
		}

		workers := teamWorkers(entries, team.ID)
		targets := resolveMentionTargetsWithFallback(trigger.Mentions, *team, workers, senderType == "user")
		if senderType != "user" {
			leader := resolveTeamLeader(*team, workers)
			filteredTargets := make([]ContainerEntry, 0, len(targets))
			for _, target := range targets {
				isLeaderTarget := normalizeRoleKind(target.RoleKind) == "leader"
				if leader != nil && target.Name == leader.Name {
					isLeaderTarget = true
				}
				if isLeaderTarget {
					// Hard policy: only user mention may target leader.
					continue
				}
				filteredTargets = append(filteredTargets, target)
			}
			targets = filteredTargets
		}
		if len(targets) == 0 {
			continue
		}

		triggerSenderID := sanitizeContainerName(trigger.SenderID)
		type targetReplyResult struct {
			target ContainerEntry
			reply  ClawRoomMessage
			err    error
		}

		results := make(chan targetReplyResult, len(targets))
		expected := 0
		snapshotMessages := append([]ClawRoomMessage(nil), messages...)
		for _, target := range targets {
			if triggerSenderID != "" && target.Name == triggerSenderID {
				continue
			}
			expected++

			var delegatedBy *ClawRoomMessage
			if senderType == "leader" || senderType == "worker" {
				triggerCopy := trigger
				delegatedBy = &triggerCopy
			}
			targetCopy := target
			delegatedByCopy := delegatedBy

			// Create placeholder message for streaming
			placeholderID := generateRoomEntityID("room-msg")
			_ = normalizeRoleKind(targetCopy.RoleKind) // validate role kind

			go func() {
				// Stream callback to push chunks to WebSocket clients
				var contentBuilder strings.Builder
				onChunk := func(chunk string, done bool) {
					if chunk != "" {
						contentBuilder.WriteString(chunk)
					}
					// Broadcast chunk to WebSocket clients
					h.publishRoomWSEvent(roomID, WSOutboundMessage{
						Type:      "message_chunk",
						MessageID: placeholderID,
						Status:    "streaming",
					})
				}

				// Use streaming version
				reply, err := h.runWorkerReplyStream(*room, *team, workers, targetCopy, snapshotMessages, trigger, delegatedByCopy, onChunk)
				if err == nil {
					// Override message ID to match placeholder
					reply.ID = placeholderID
				}
				results <- targetReplyResult{
					target: targetCopy,
					reply:  reply,
					err:    err,
				}
			}()
		}

		for i := 0; i < expected; i++ {
			result := <-results
			target := result.target
			reply := result.reply
			err := result.err
			if err != nil {
				errMsg := newRoomMessage(
					*room,
					"system",
					"clawos-system",
					"ClawOS",
					fmt.Sprintf("@%s is unavailable: %v", target.Name, err),
					map[string]string{"worker": target.Name, "phase": "reply"},
				)
				if appendErr := h.appendRoomMessage(room.ID, errMsg); appendErr != nil {
					log.Printf("openclaw: failed to append room system message: %v", appendErr)
				}
				messages = append(messages, errMsg)
				continue
			}
			if err := h.appendRoomMessage(room.ID, reply); err != nil {
				log.Printf("openclaw: failed to append room reply: %v", err)
				continue
			}
			messages = append(messages, reply)
			if len(reply.Mentions) > 0 {
				queue = append(queue, reply.ID)
			}
		}
	}

	if turns >= maxAutomationTurns && len(queue) > 0 {
		h.mu.RLock()
		rooms, _ := h.loadRooms()
		h.mu.RUnlock()
		if room := findRoomByID(rooms, roomID); room != nil {
			warn := newRoomMessage(
				*room,
				"system",
				"clawos-system",
				"ClawOS",
				"Room automation reached max turns and was paused to avoid loops.",
				map[string]string{"phase": "safety", "reason": "max-turns"},
			)
			_ = h.appendRoomMessage(roomID, warn)
		}
	}
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
