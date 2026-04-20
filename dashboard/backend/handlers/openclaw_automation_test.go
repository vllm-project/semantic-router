package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestProcessRoomUserMessage_LeaderDelegatesToWorker(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, false)

	leaderSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assertChatCompletionPath(t, r)
		if got := r.Header.Get("X-OpenClaw-Agent-Id"); got != openClawPrimaryAgentID {
			t.Fatalf("expected X-OpenClaw-Agent-Id=%s, got %q", openClawPrimaryAgentID, got)
		}
		var payload openAIChatRequest
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("failed to decode leader payload: %v", err)
		}
		if payload.Model != openClawPrimaryAgentModel {
			t.Fatalf("unexpected leader model: %s", payload.Model)
		}
		encodeOpenAIResponse(w, "@worker-a please prepare the implementation checklist.")
	}))
	defer leaderSrv.Close()

	workerSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assertChatCompletionPath(t, r)
		var payload openAIChatRequest
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("failed to decode worker payload: %v", err)
		}
		if payload.Model != openClawPrimaryAgentModel {
			t.Fatalf("unexpected worker model: %s", payload.Model)
		}
		encodeOpenAIResponse(w, "Checklist prepared and ready for review.")
	}))
	defer workerSrv.Close()

	team := newTestTeam("team-alpha", "Alpha", "leader-1")
	room := seedTeamAndRoom(t, h, team, []ContainerEntry{
		newTestWorker("leader-1", mustServerPort(t, leaderSrv.URL), tempDir, team.ID, "leader"),
		newTestWorker("worker-a", mustServerPort(t, workerSrv.URL), tempDir, team.ID, "worker"),
	})

	userMessage := newRoomMessage(room, "user", "user-1", "You", "Please @leader break this down and delegate.", nil)
	if err := h.appendRoomMessage(room.ID, userMessage); err != nil {
		t.Fatalf("failed to append user message: %v", err)
	}
	h.processRoomUserMessage(room.ID, userMessage.ID)

	messages, err := h.loadRoomMessages(room.ID)
	if err != nil {
		t.Fatalf("failed to load room messages: %v", err)
	}
	if len(messages) < 3 {
		t.Fatalf("expected at least 3 room messages (user + leader + worker), got %d", len(messages))
	}
	assertMessageFromSender(t, messages, "leader-1", "@worker-a")
	assertMessageFromSender(t, messages, "worker-a", "checklist prepared")
}

func TestProcessRoomUserMessage_SimultaneousMentionsContinueChain(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, false)

	var mu sync.Mutex
	leaderCalls, workerCalls := 0, 0

	leaderSrv := newMultiCallServer(t, &leaderCalls, &mu, func(call int) string {
		if call == 1 {
			return "@worker-a please draft the implementation plan."
		}
		return "Leader reviewed worker output."
	})
	defer leaderSrv.Close()

	workerSrv := newMultiCallServer(t, &workerCalls, &mu, func(call int) string {
		if call == 1 {
			return "@leader initial draft is complete."
		}
		return "Worker final updates done."
	})
	defer workerSrv.Close()

	team := newTestTeam("team-sync", "Sync Team", "leader-1")
	room := seedTeamAndRoom(t, h, team, []ContainerEntry{
		newTestWorker("leader-1", mustServerPort(t, leaderSrv.URL), tempDir, team.ID, "leader"),
		newTestWorker("worker-a", mustServerPort(t, workerSrv.URL), tempDir, team.ID, "worker"),
	})

	userMessage := newRoomMessage(room, "user", "user-1", "You", "Please @leader and @worker-a collaborate.", nil)
	if err := h.appendRoomMessage(room.ID, userMessage); err != nil {
		t.Fatalf("failed to append user message: %v", err)
	}
	h.processRoomUserMessage(room.ID, userMessage.ID)

	mu.Lock()
	gotLeaderCalls, gotWorkerCalls := leaderCalls, workerCalls
	mu.Unlock()
	if gotLeaderCalls != 1 {
		t.Fatalf("expected leader to be called exactly once (worker @leader should be ignored), got %d", gotLeaderCalls)
	}
	if gotWorkerCalls < 2 {
		t.Fatalf("expected worker to be called at least twice, got %d", gotWorkerCalls)
	}

	messages, err := h.loadRoomMessages(room.ID)
	if err != nil {
		t.Fatalf("failed to load room messages: %v", err)
	}
	if len(messages) < 4 {
		t.Fatalf("expected at least 4 room messages, got %d", len(messages))
	}
	assertMessageFromSender(t, messages, "leader-1", "@worker-a")
	assertMessageFromSender(t, messages, "worker-a", "@leader")
}

func TestProcessRoomUserMessage_MentionAllTargetsEntireTeam(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, false)

	var mu sync.Mutex
	leaderCalls, workerACalls, workerBCalls := 0, 0, 0

	leaderSrv := newAckServer(t, &leaderCalls, &mu, "Leader acknowledged.")
	defer leaderSrv.Close()
	workerASrv := newAckServer(t, &workerACalls, &mu, "Worker A acknowledged.")
	defer workerASrv.Close()
	workerBSrv := newAckServer(t, &workerBCalls, &mu, "Worker B acknowledged.")
	defer workerBSrv.Close()

	team := newTestTeam("team-all", "All Team", "leader-1")
	room := seedTeamAndRoom(t, h, team, []ContainerEntry{
		newTestWorker("leader-1", mustServerPort(t, leaderSrv.URL), tempDir, team.ID, "leader"),
		newTestWorker("worker-a", mustServerPort(t, workerASrv.URL), tempDir, team.ID, "worker"),
		newTestWorker("worker-b", mustServerPort(t, workerBSrv.URL), tempDir, team.ID, "worker"),
	})

	userMessage := newRoomMessage(room, "user", "user-1", "You", "@all please share your current status.", nil)
	if err := h.appendRoomMessage(room.ID, userMessage); err != nil {
		t.Fatalf("failed to append user message: %v", err)
	}
	h.processRoomUserMessage(room.ID, userMessage.ID)

	mu.Lock()
	gotLeader, gotWorkerA, gotWorkerB := leaderCalls, workerACalls, workerBCalls
	mu.Unlock()
	if gotLeader != 1 {
		t.Fatalf("expected leader to be called once for @all, got %d", gotLeader)
	}
	if gotWorkerA != 1 {
		t.Fatalf("expected worker-a to be called once for @all, got %d", gotWorkerA)
	}
	if gotWorkerB != 1 {
		t.Fatalf("expected worker-b to be called once for @all, got %d", gotWorkerB)
	}
}

func TestProcessRoomUserMessage_DuplicateTriggerDoesNotReprocess(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, false)

	var mu sync.Mutex
	workerCall := 0
	workerSrv := newAckServer(t, &workerCall, &mu, "Done.")
	defer workerSrv.Close()

	team := newTestTeam("team-dedup", "Dedup Team", "")
	room := seedTeamAndRoom(t, h, team, []ContainerEntry{
		newTestWorker("worker-a", mustServerPort(t, workerSrv.URL), tempDir, team.ID, "worker"),
	})

	trigger := newRoomMessage(room, "user", "user-1", "You", "@worker-a run once", nil)
	if err := h.appendRoomMessage(room.ID, trigger); err != nil {
		t.Fatalf("failed to append trigger message: %v", err)
	}

	h.processRoomUserMessage(room.ID, trigger.ID)
	h.processRoomUserMessage(room.ID, trigger.ID)

	mu.Lock()
	gotCalls := workerCall
	mu.Unlock()
	if gotCalls != 1 {
		t.Fatalf("expected worker to be called once for duplicate trigger id, got %d", gotCalls)
	}

	messages, err := h.loadRoomMessages(room.ID)
	if err != nil {
		t.Fatalf("failed to load room messages: %v", err)
	}
	var triggerStored *ClawRoomMessage
	for i := range messages {
		if messages[i].ID == trigger.ID {
			triggerStored = &messages[i]
			break
		}
	}
	if triggerStored == nil {
		t.Fatalf("trigger message not found in room history")
	}
	if triggerStored.Metadata == nil || strings.TrimSpace(triggerStored.Metadata[roomAutomationProcessedAtKey]) == "" {
		t.Fatalf("trigger message should be marked as processed, got metadata: %+v", triggerStored.Metadata)
	}
}

func TestProcessRoomUserMessage_MultiMentionsDispatchInParallel(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, false)

	var mu sync.Mutex
	var startA, startB time.Time

	workerASrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assertChatCompletionPath(t, r)
		mu.Lock()
		startA = time.Now()
		mu.Unlock()
		time.Sleep(300 * time.Millisecond)
		encodeOpenAIResponse(w, "worker-a done")
	}))
	defer workerASrv.Close()

	workerBSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assertChatCompletionPath(t, r)
		mu.Lock()
		startB = time.Now()
		mu.Unlock()
		time.Sleep(300 * time.Millisecond)
		encodeOpenAIResponse(w, "worker-b done")
	}))
	defer workerBSrv.Close()

	team := newTestTeam("team-parallel", "Parallel Team", "")
	room := seedTeamAndRoom(t, h, team, []ContainerEntry{
		newTestWorker("worker-a", mustServerPort(t, workerASrv.URL), tempDir, team.ID, "worker"),
		newTestWorker("worker-b", mustServerPort(t, workerBSrv.URL), tempDir, team.ID, "worker"),
	})

	userMessage := newRoomMessage(room, "user", "user-1", "You", "@worker-a @worker-b please run in parallel", nil)
	if err := h.appendRoomMessage(room.ID, userMessage); err != nil {
		t.Fatalf("failed to append user message: %v", err)
	}
	h.processRoomUserMessage(room.ID, userMessage.ID)

	mu.Lock()
	defer mu.Unlock()
	if startA.IsZero() || startB.IsZero() {
		t.Fatalf("expected both worker requests to be started, got startA=%v startB=%v", startA, startB)
	}
	diff := startA.Sub(startB)
	if diff < 0 {
		diff = -diff
	}
	if diff > 220*time.Millisecond {
		t.Fatalf("expected worker requests to start nearly together for parallel dispatch, diff=%s", diff)
	}
}

func TestQueryWorkerChat_FallsBackToAlternativeEndpoint(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, false)

	primaryAttempts := 0
	fallbackAttempts := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/chat/completions":
			primaryAttempts++
			http.NotFound(w, r)
		case "/api/openai/v1/chat/completions":
			fallbackAttempts++
			_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"fallback endpoint reply"}}]}`))
		default:
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
	}))
	defer srv.Close()

	worker := ContainerEntry{
		Name:     "mira",
		Port:     mustServerPort(t, srv.URL),
		Image:    "ghcr.io/openclaw/openclaw:latest",
		Token:    "test-token",
		DataDir:  tempDir,
		RoleKind: "worker",
	}
	if err := h.saveRegistry([]ContainerEntry{worker}); err != nil {
		t.Fatalf("failed to seed registry: %v", err)
	}

	content, err := h.queryWorkerChat(worker, "system", "user")
	if err != nil {
		t.Fatalf("queryWorkerChat should succeed via fallback endpoint: %v", err)
	}
	if !strings.Contains(content, "fallback endpoint reply") {
		t.Fatalf("unexpected content: %q", content)
	}
	if primaryAttempts == 0 {
		t.Fatalf("expected primary endpoint to be attempted at least once")
	}
	if fallbackAttempts == 0 {
		t.Fatalf("expected fallback endpoint to be attempted at least once")
	}
}

func mustServerPort(t *testing.T, rawURL string) int {
	t.Helper()
	parsed, err := url.Parse(rawURL)
	if err != nil {
		t.Fatalf("failed to parse test server URL: %v", err)
	}
	port, err := strconv.Atoi(parsed.Port())
	if err != nil {
		t.Fatalf("failed to parse test server port: %v", err)
	}
	return port
}

// newAckServer returns a test server that counts calls and responds with a fixed message.
func newAckServer(t *testing.T, callCounter *int, mu *sync.Mutex, message string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assertChatCompletionPath(t, r)
		mu.Lock()
		*callCounter++
		mu.Unlock()
		encodeOpenAIResponse(w, message)
	}))
}

// newMultiCallServer returns a test server where each call can return a different response.
func newMultiCallServer(t *testing.T, callCounter *int, mu *sync.Mutex, responses func(call int) string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assertChatCompletionPath(t, r)
		var payload openAIChatRequest
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("failed to decode payload: %v", err)
		}
		mu.Lock()
		*callCounter++
		call := *callCounter
		mu.Unlock()
		encodeOpenAIResponse(w, responses(call))
	}))
}

func assertChatCompletionPath(t *testing.T, r *http.Request) {
	t.Helper()
	if r.URL.Path != "/v1/chat/completions" {
		t.Fatalf("unexpected path: %s", r.URL.Path)
	}
}

func assertMessageFromSender(t *testing.T, messages []ClawRoomMessage, senderID, contentSubstring string) {
	t.Helper()
	for _, msg := range messages {
		if msg.SenderID == senderID && strings.Contains(strings.ToLower(msg.Content), strings.ToLower(contentSubstring)) {
			return
		}
	}
	t.Fatalf("expected message from %s containing %q, got: %+v", senderID, contentSubstring, messages)
}
