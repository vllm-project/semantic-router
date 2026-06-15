package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

func newRoomPayloadServer(
	t *testing.T,
	responseContent string,
	onPayload func(openAIChatRequest),
) *httptest.Server {
	t.Helper()

	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload openAIChatRequest
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("failed to decode payload: %v", err)
		}
		if onPayload != nil {
			onPayload(payload)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"content": responseContent}}},
		})
	}))
}

func ensureRoomWithTeamMembers(
	t *testing.T,
	h *OpenClawHandler,
	team TeamEntry,
	entries []ContainerEntry,
) ClawRoomEntry {
	t.Helper()

	if err := h.saveTeams([]TeamEntry{team}); err != nil {
		t.Fatalf("failed to seed team: %v", err)
	}
	if err := h.saveRegistry(entries); err != nil {
		t.Fatalf("failed to seed team members: %v", err)
	}

	h.mu.Lock()
	room, err := h.ensureDefaultRoomLocked(team)
	h.mu.Unlock()
	if err != nil {
		t.Fatalf("failed to ensure room: %v", err)
	}
	return room
}

func appendRoomMessagesForTest(
	t *testing.T,
	h *OpenClawHandler,
	roomID string,
	messages ...ClawRoomMessage,
) {
	t.Helper()

	for _, message := range messages {
		if err := h.appendRoomMessage(roomID, message); err != nil {
			t.Fatalf("failed to append room message: %v", err)
		}
	}
}

func assertStructuredRoomMessages(t *testing.T, messages []openAIChatMessage) {
	t.Helper()

	for _, message := range messages {
		if strings.Contains(message.Content, "Recent messages:") || strings.Contains(message.Content, "Latest message from") {
			t.Fatalf("expected structured message array, got %+v", messages)
		}
	}
}

func TestProcessRoomUserMessage_StripsLeadingMentionsFromPrompt(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, false)

	var (
		promptMu sync.Mutex
		payloads = map[string][]openAIChatRequest{}
	)
	recordPayload := func(workerID string) func(openAIChatRequest) {
		return func(payload openAIChatRequest) {
			promptMu.Lock()
			payloads[workerID] = append(payloads[workerID], payload)
			promptMu.Unlock()
		}
	}

	workerASrv := newRoomPayloadServer(t, "worker-a ready.", recordPayload("worker-a"))
	defer workerASrv.Close()
	workerBSrv := newRoomPayloadServer(t, "worker-b ready.", recordPayload("worker-b"))
	defer workerBSrv.Close()

	team := TeamEntry{ID: "team-prompt", Name: "Prompt Team", CreatedAt: time.Now().UTC().Format(time.RFC3339), UpdatedAt: time.Now().UTC().Format(time.RFC3339)}
	room := ensureRoomWithTeamMembers(t, h, team, []ContainerEntry{
		{Name: "worker-a", Port: mustServerPort(t, workerASrv.URL), Image: "ghcr.io/openclaw/openclaw:latest", Token: "token-a", DataDir: tempDir, TeamID: team.ID, TeamName: team.Name, RoleKind: "worker"},
		{Name: "worker-b", Port: mustServerPort(t, workerBSrv.URL), Image: "ghcr.io/openclaw/openclaw:latest", Token: "token-b", DataDir: tempDir, TeamID: team.ID, TeamName: team.Name, RoleKind: "worker"},
	})

	userMessage := newRoomMessage(room, "user", "user-1", "You", "@worker-a @worker-b introduce yourselves.", nil)
	appendRoomMessagesForTest(t, h, room.ID, userMessage)
	h.processRoomUserMessage(room.ID, userMessage.ID)

	promptMu.Lock()
	defer promptMu.Unlock()
	for _, workerID := range []string{"worker-a", "worker-b"} {
		firstPayload := payloads[workerID][0]
		if got := firstPayload.User; got != roomScopedSessionUser(room, ContainerEntry{Name: workerID}) {
			t.Fatalf("expected room-scoped session user for %s, got %q", workerID, got)
		}
		if len(firstPayload.Messages) != 2 {
			t.Fatalf("expected only system + latest user message for %s, got %+v", workerID, firstPayload.Messages)
		}
		if firstPayload.Messages[len(firstPayload.Messages)-1].Content != "[You] introduce yourselves." {
			t.Fatalf("expected stripped latest message for %s, got %+v", workerID, firstPayload.Messages)
		}
		assertStructuredRoomMessages(t, firstPayload.Messages)
	}
}

func TestProcessRoomUserMessage_WorkerPromptIncludesLeaderRouting(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, false)

	var (
		promptMu      sync.Mutex
		workerPayload openAIChatRequest
	)

	workerSrv := newRoomPayloadServer(t, "Understood, I will report back to @leader.", func(payload openAIChatRequest) {
		promptMu.Lock()
		workerPayload = payload
		promptMu.Unlock()
	})
	defer workerSrv.Close()

	team := TeamEntry{ID: "team-routing", Name: "Routing Team", LeaderID: "leader-1", CreatedAt: time.Now().UTC().Format(time.RFC3339), UpdatedAt: time.Now().UTC().Format(time.RFC3339)}
	room := ensureRoomWithTeamMembers(t, h, team, []ContainerEntry{
		{Name: "leader-1", Port: 18790, Image: "ghcr.io/openclaw/openclaw:latest", Token: "token-leader", DataDir: tempDir, TeamID: team.ID, TeamName: team.Name, RoleKind: "leader", AgentName: "Echo", AgentRole: "Team Leader"},
		{Name: "worker-a", Port: mustServerPort(t, workerSrv.URL), Image: "ghcr.io/openclaw/openclaw:latest", Token: "token-worker", DataDir: tempDir, TeamID: team.ID, TeamName: team.Name, RoleKind: "worker", AgentName: "Mira"},
	})

	userMessage := newRoomMessage(room, "user", "user-1", "You", "@worker-a investigate first, then report the result to leader.", nil)
	appendRoomMessagesForTest(t, h, room.ID, userMessage)
	h.processRoomUserMessage(room.ID, userMessage.ID)

	promptMu.Lock()
	defer promptMu.Unlock()
	systemPrompt := workerPayload.Messages[0].Content
	userPrompt := workerPayload.Messages[len(workerPayload.Messages)-1].Content
	if !strings.Contains(systemPrompt, "Workers cannot use @mentions") || !strings.Contains(systemPrompt, "Leader aliases: @leader and @leader-1 = Echo") {
		t.Fatalf("expected worker system prompt to include leader routing rules, got:\n%s", systemPrompt)
	}
	if !strings.Contains(systemPrompt, "Hard rule: workers cannot use @mentions") {
		t.Fatalf("expected worker system prompt to include no-mention hard rule, got:\n%s", systemPrompt)
	}
	if workerPayload.User != roomScopedSessionUser(room, ContainerEntry{Name: "worker-a"}) {
		t.Fatalf("expected room-scoped worker session user, got %q", workerPayload.User)
	}
	if userPrompt != "[You] investigate first, then report the result to leader." {
		t.Fatalf("expected latest directed message only, got %q", userPrompt)
	}
	assertStructuredRoomMessages(t, workerPayload.Messages)
}

func TestProcessRoomUserMessage_UsesRoomScopedSessionAndDirectedHistory(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, false)

	var (
		payloadMu sync.Mutex
		payloads  []openAIChatRequest
	)

	workerSrv := newRoomPayloadServer(t, "Worker follow-up complete.", func(payload openAIChatRequest) {
		payloadMu.Lock()
		payloads = append(payloads, payload)
		payloadMu.Unlock()
	})
	defer workerSrv.Close()
	otherSrv := newRoomPayloadServer(t, "Worker B reply.", nil)
	defer otherSrv.Close()

	team := TeamEntry{ID: "team-directed-history", Name: "Directed History Team", CreatedAt: time.Now().UTC().Format(time.RFC3339), UpdatedAt: time.Now().UTC().Format(time.RFC3339)}
	room := ensureRoomWithTeamMembers(t, h, team, []ContainerEntry{
		{Name: "worker-a", Port: mustServerPort(t, workerSrv.URL), Image: "ghcr.io/openclaw/openclaw:latest", Token: "token-a", DataDir: tempDir, TeamID: team.ID, TeamName: team.Name, RoleKind: "worker", AgentName: "Mira"},
		{Name: "worker-b", Port: mustServerPort(t, otherSrv.URL), Image: "ghcr.io/openclaw/openclaw:latest", Token: "token-b", DataDir: tempDir, TeamID: team.ID, TeamName: team.Name, RoleKind: "worker", AgentName: "Noa"},
	})

	trigger := newRoomMessage(room, "user", "user-1", "You", "@worker-a second task", nil)
	appendRoomMessagesForTest(t, h, room.ID,
		newRoomMessage(room, "user", "user-1", "You", "@worker-a first investigation request", nil),
		newRoomMessage(room, "worker", "worker-a", "Mira", "Investigation started.", nil),
		newRoomMessage(room, "user", "user-1", "You", "@worker-b unrelated task", nil),
		newRoomMessage(room, "user", "user-1", "You", "general room update without mention", nil),
		trigger,
	)

	h.processRoomUserMessage(room.ID, trigger.ID)

	payloadMu.Lock()
	defer payloadMu.Unlock()
	payload := payloads[0]
	if payload.User != roomScopedSessionUser(room, ContainerEntry{Name: "worker-a"}) {
		t.Fatalf("expected room-scoped session user, got %q", payload.User)
	}
	if got := len(payload.Messages); got != 4 {
		t.Fatalf("expected system + 3 directed messages, got %+v", payload.Messages)
	}
	expected := []openAIChatMessage{
		{Role: "user", Content: "[You] first investigation request"},
		{Role: "assistant", Content: "Investigation started."},
		{Role: "user", Content: "[You] second task"},
	}
	for i, want := range expected {
		if payload.Messages[i+1] != want {
			t.Fatalf("message %d = %+v, want %+v", i+1, payload.Messages[i+1], want)
		}
	}
	assertStructuredRoomMessages(t, payload.Messages)
}

func TestProcessRoomUserMessage_SameWorkerGetsDifferentSessionUsersPerRoom(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, false)

	var (
		payloadMu sync.Mutex
		users     []string
	)

	workerSrv := newRoomPayloadServer(t, "done", func(payload openAIChatRequest) {
		payloadMu.Lock()
		users = append(users, payload.User)
		payloadMu.Unlock()
	})
	defer workerSrv.Close()

	team := TeamEntry{ID: "team-room-sessions", Name: "Room Sessions Team", CreatedAt: time.Now().UTC().Format(time.RFC3339), UpdatedAt: time.Now().UTC().Format(time.RFC3339)}
	roomA := ensureRoomWithTeamMembers(t, h, team, []ContainerEntry{{
		Name:     "worker-a",
		Port:     mustServerPort(t, workerSrv.URL),
		Image:    "ghcr.io/openclaw/openclaw:latest",
		Token:    "token-a",
		DataDir:  tempDir,
		TeamID:   team.ID,
		RoleKind: "worker",
	}})
	roomB := ClawRoomEntry{ID: "team-room-sessions-b", TeamID: team.ID, Name: "Secondary Room", CreatedAt: time.Now().UTC().Format(time.RFC3339), UpdatedAt: time.Now().UTC().Format(time.RFC3339)}
	if err := h.saveRooms([]ClawRoomEntry{roomA, roomB}); err != nil {
		t.Fatalf("failed to save rooms: %v", err)
	}

	for _, room := range []ClawRoomEntry{roomA, roomB} {
		trigger := newRoomMessage(room, "user", "user-1", "You", "@worker-a please handle this room", nil)
		appendRoomMessagesForTest(t, h, room.ID, trigger)
		h.processRoomUserMessage(room.ID, trigger.ID)
	}

	payloadMu.Lock()
	defer payloadMu.Unlock()
	if len(users) != 2 || users[0] == users[1] {
		t.Fatalf("expected distinct session users per room, got %v", users)
	}
	if users[0] != roomScopedSessionUser(roomA, ContainerEntry{Name: "worker-a"}) || users[1] != roomScopedSessionUser(roomB, ContainerEntry{Name: "worker-a"}) {
		t.Fatalf("unexpected session users: %v", users)
	}
}

func TestProcessRoomUserMessage_LeaderAndWorkerUseDifferentSessionUsers(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, false)

	var (
		leaderPayload openAIChatRequest
		workerPayload openAIChatRequest
		payloadMu     sync.Mutex
	)

	leaderSrv := newRoomPayloadServer(t, "Leader acknowledged.", func(payload openAIChatRequest) {
		payloadMu.Lock()
		leaderPayload = payload
		payloadMu.Unlock()
	})
	defer leaderSrv.Close()
	workerSrv := newRoomPayloadServer(t, "Worker acknowledged.", func(payload openAIChatRequest) {
		payloadMu.Lock()
		workerPayload = payload
		payloadMu.Unlock()
	})
	defer workerSrv.Close()

	team := TeamEntry{ID: "team-role-sessions", Name: "Role Sessions Team", LeaderID: "leader-1", CreatedAt: time.Now().UTC().Format(time.RFC3339), UpdatedAt: time.Now().UTC().Format(time.RFC3339)}
	room := ensureRoomWithTeamMembers(t, h, team, []ContainerEntry{
		{Name: "leader-1", Port: mustServerPort(t, leaderSrv.URL), Image: "ghcr.io/openclaw/openclaw:latest", Token: "leader-token", DataDir: tempDir, TeamID: team.ID, RoleKind: "leader"},
		{Name: "worker-a", Port: mustServerPort(t, workerSrv.URL), Image: "ghcr.io/openclaw/openclaw:latest", Token: "worker-token", DataDir: tempDir, TeamID: team.ID, RoleKind: "worker"},
	})

	leaderTrigger := newRoomMessage(room, "user", "user-1", "You", "@leader status update?", nil)
	workerTrigger := newRoomMessage(room, "user", "user-1", "You", "@worker-a investigate this issue", nil)
	appendRoomMessagesForTest(t, h, room.ID, leaderTrigger, workerTrigger)
	h.processRoomUserMessage(room.ID, leaderTrigger.ID)
	h.processRoomUserMessage(room.ID, workerTrigger.ID)

	payloadMu.Lock()
	defer payloadMu.Unlock()
	if leaderPayload.User != roomScopedSessionUser(room, ContainerEntry{Name: "leader-1"}) || workerPayload.User != roomScopedSessionUser(room, ContainerEntry{Name: "worker-a"}) {
		t.Fatalf("unexpected role session users: leader=%q worker=%q", leaderPayload.User, workerPayload.User)
	}
	if leaderPayload.User == workerPayload.User {
		t.Fatalf("leader and worker should not share the same session user: %q", leaderPayload.User)
	}
}

func TestProcessRoomUserMessage_NoMentionDoesNotTriggerAutomation(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, false)

	var (
		callMu sync.Mutex
		calls  int
	)

	workerSrv := newRoomPayloadServer(t, "should not happen", func(openAIChatRequest) {
		callMu.Lock()
		calls++
		callMu.Unlock()
	})
	defer workerSrv.Close()

	team := TeamEntry{ID: "team-no-mention", Name: "No Mention Team", LeaderID: "leader-1", CreatedAt: time.Now().UTC().Format(time.RFC3339), UpdatedAt: time.Now().UTC().Format(time.RFC3339)}
	room := ensureRoomWithTeamMembers(t, h, team, []ContainerEntry{
		{Name: "leader-1", Port: mustServerPort(t, workerSrv.URL), Image: "ghcr.io/openclaw/openclaw:latest", Token: "leader-token", DataDir: tempDir, TeamID: team.ID, RoleKind: "leader"},
		{Name: "worker-a", Port: mustServerPort(t, workerSrv.URL), Image: "ghcr.io/openclaw/openclaw:latest", Token: "worker-token", DataDir: tempDir, TeamID: team.ID, RoleKind: "worker"},
	})

	trigger := newRoomMessage(room, "user", "user-1", "You", "plain room note without any mention", nil)
	appendRoomMessagesForTest(t, h, room.ID, trigger)
	h.processRoomUserMessage(room.ID, trigger.ID)

	callMu.Lock()
	gotCalls := calls
	callMu.Unlock()
	if gotCalls != 0 {
		t.Fatalf("expected no automation calls for no-mention trigger, got %d", gotCalls)
	}
	messages, err := h.loadRoomMessages(room.ID)
	if err != nil {
		t.Fatalf("failed to load room messages: %v", err)
	}
	if len(messages) != 1 {
		t.Fatalf("expected no additional room replies for no-mention trigger, got %+v", messages)
	}
}
