package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

func seedRoomTeamWithLeaderAndWorker(
	t *testing.T,
	h *OpenClawHandler,
	tempDir, teamID, teamName, workerURL string,
) ClawRoomEntry {
	t.Helper()

	team := TeamEntry{
		ID:        teamID,
		Name:      teamName,
		LeaderID:  "leader-1",
		CreatedAt: time.Now().UTC().Format(time.RFC3339),
		UpdatedAt: time.Now().UTC().Format(time.RFC3339),
	}
	if err := h.saveTeams([]TeamEntry{team}); err != nil {
		t.Fatalf("failed to seed team: %v", err)
	}
	if err := h.saveRegistry([]ContainerEntry{
		{
			Name:     "leader-1",
			Port:     mustServerPort(t, workerURL),
			Image:    "ghcr.io/openclaw/openclaw:latest",
			Token:    "leader-token",
			DataDir:  tempDir,
			TeamID:   team.ID,
			RoleKind: "leader",
		},
		{
			Name:     "worker-a",
			Port:     mustServerPort(t, workerURL),
			Image:    "ghcr.io/openclaw/openclaw:latest",
			Token:    "worker-token",
			DataDir:  tempDir,
			TeamID:   team.ID,
			RoleKind: "worker",
		},
	}); err != nil {
		t.Fatalf("failed to seed workers: %v", err)
	}

	h.mu.Lock()
	room, err := h.ensureDefaultRoomLocked(team)
	h.mu.Unlock()
	if err != nil {
		t.Fatalf("failed to ensure room: %v", err)
	}
	return room
}

func postRoomMessage(t *testing.T, h *OpenClawHandler, roomID, body string) ClawRoomMessage {
	t.Helper()

	req := httptest.NewRequest(
		http.MethodPost,
		fmt.Sprintf("/api/openclaw/rooms/%s/messages", roomID),
		strings.NewReader(body),
	)
	resp := httptest.NewRecorder()
	h.RoomByIDHandler().ServeHTTP(resp, req)
	if resp.Code != http.StatusCreated {
		t.Fatalf("expected 201 for room post, got %d: %s", resp.Code, resp.Body.String())
	}

	var created ClawRoomMessage
	if err := json.Unmarshal(resp.Body.Bytes(), &created); err != nil {
		t.Fatalf("failed to parse created room message: %v", err)
	}
	return created
}

func waitForRoomMessage(
	t *testing.T,
	h *OpenClawHandler,
	roomID, description string,
	match func(ClawRoomMessage) bool,
) {
	t.Helper()

	deadline := time.Now().Add(2 * time.Second)
	for {
		messages, err := h.loadRoomMessages(roomID)
		if err != nil {
			t.Fatalf("failed to load room messages: %v", err)
		}
		for _, msg := range messages {
			if match(msg) {
				return
			}
		}
		if time.Now().After(deadline) {
			t.Fatalf("expected %s, got messages: %+v", description, messages)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func waitForWSOutboundMessage(
	t *testing.T,
	conn *websocket.Conn,
	description string,
	match func(WSOutboundMessage) bool,
) WSOutboundMessage {
	t.Helper()

	for {
		var outbound WSOutboundMessage
		if readErr := conn.ReadJSON(&outbound); readErr != nil {
			t.Fatalf("failed to read %s: %v", description, readErr)
		}
		if match(outbound) {
			return outbound
		}
	}
}

func TestRoomMessagesPost_LeaderSenderTypeTriggersAutomation(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, false)

	workerSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected worker path: %s", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"content": "Done. Worker execution completed."}}},
		})
	}))
	defer workerSrv.Close()

	room := seedRoomTeamWithLeaderAndWorker(t, h, tempDir, "team-room-post", "Room Post Team", workerSrv.URL)
	created := postRoomMessage(t, h, room.ID, `{
		"senderType":"leader",
		"senderId":"leader-1",
		"senderName":"Leader One",
		"content":"@worker-a please execute this task."
	}`)

	if created.SenderType != "leader" {
		t.Fatalf("expected senderType leader, got %q", created.SenderType)
	}
	if created.SenderID != "leader-1" {
		t.Fatalf("expected senderID leader-1, got %q", created.SenderID)
	}

	waitForRoomMessage(t, h, room.ID, "worker-a reply after leader message", func(msg ClawRoomMessage) bool {
		return msg.SenderID == "worker-a" && msg.SenderType == "worker"
	})
}

func TestRoomMessagesPost_ReadOnlyAllowsChatAndAutomation(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, true)

	workerSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected worker path: %s", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"content": "Readonly reply completed."}}},
		})
	}))
	defer workerSrv.Close()

	room := seedRoomTeamWithLeaderAndWorker(t, h, tempDir, "team-room-post-readonly", "Room Post Readonly Team", workerSrv.URL)
	postRoomMessage(t, h, room.ID, `{
		"senderType":"leader",
		"senderId":"leader-1",
		"senderName":"Leader One",
		"content":"@worker-a please execute this task."
	}`)

	waitForRoomMessage(t, h, room.ID, "worker-a readonly reply", func(msg ClawRoomMessage) bool {
		return msg.SenderID == "worker-a" &&
			msg.SenderType == "worker" &&
			strings.Contains(msg.Content, "Readonly reply completed.")
	})
}

func TestRoomMessagesWebSocket_ReadOnlyAllowsSend(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, true)
	room := ClawRoomEntry{
		ID:        "room-readonly-ws",
		TeamID:    "team-a",
		Name:      "Readonly WS",
		CreatedAt: time.Now().UTC().Format(time.RFC3339),
		UpdatedAt: time.Now().UTC().Format(time.RFC3339),
	}
	if err := h.saveRooms([]ClawRoomEntry{room}); err != nil {
		t.Fatalf("failed to seed room: %v", err)
	}

	server := httptest.NewServer(h.RoomByIDHandler())
	defer server.Close()

	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/api/openclaw/rooms/" + room.ID + "/ws"
	conn, resp, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		t.Fatalf("failed to connect websocket: %v", err)
	}
	defer func() { _ = conn.Close() }()
	if resp != nil && resp.Body != nil {
		defer func() { _ = resp.Body.Close() }()
	}

	_ = conn.SetReadDeadline(time.Now().Add(2 * time.Second))
	waitForWSOutboundMessage(t, conn, "connected message", func(outbound WSOutboundMessage) bool {
		return outbound.Type == WSTypeConnected
	})

	if writeErr := conn.WriteJSON(WSInboundMessage{
		Type:       WSTypeSendMessage,
		Content:    "readonly websocket message",
		SenderType: "system",
		SenderName: "System",
	}); writeErr != nil {
		t.Fatalf("failed to write websocket message: %v", writeErr)
	}

	outbound := waitForWSOutboundMessage(t, conn, "websocket outbound message", func(message WSOutboundMessage) bool {
		return message.Type == WSTypeNewMessage
	})
	if outbound.Message == nil || outbound.Message.Content != "readonly websocket message" {
		t.Fatalf("unexpected outbound message: %+v", outbound)
	}

	messages, err := h.loadRoomMessages(room.ID)
	if err != nil {
		t.Fatalf("failed to load saved room messages: %v", err)
	}
	if len(messages) != 1 || messages[0].Content != "readonly websocket message" {
		t.Fatalf("expected readonly websocket message to be persisted, got %+v", messages)
	}
}

func TestQueryWorkerChat_ReadOnlySkipsEndpointRepair(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, true)

	workerSrv := httptest.NewServer(http.NotFoundHandler())
	defer workerSrv.Close()

	workerDir := filepath.Join(tempDir, "worker-a")
	configPath := filepath.Join(workerDir, "openclaw.json")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatalf("failed to create worker config dir: %v", err)
	}

	originalConfig := `{"gateway":{"http":{"endpoints":{"chatCompletions":{"enabled":false}}}}}`
	if err := os.WriteFile(configPath, []byte(originalConfig), 0o644); err != nil {
		t.Fatalf("failed to write worker config: %v", err)
	}

	worker := ContainerEntry{
		Name:    "worker-a",
		Port:    mustServerPort(t, workerSrv.URL),
		Image:   "ghcr.io/openclaw/openclaw:latest",
		Token:   "worker-token",
		DataDir: workerDir,
	}
	if err := h.saveRegistry([]ContainerEntry{worker}); err != nil {
		t.Fatalf("failed to seed registry: %v", err)
	}

	_, err := h.queryWorkerChat(worker, "system prompt", "user prompt")
	if err == nil {
		t.Fatalf("expected readonly endpoint repair to be skipped")
	}
	if !strings.Contains(err.Error(), "worker endpoint recovery skipped (read-only mode)") {
		t.Fatalf("unexpected error: %v", err)
	}

	updatedConfig, readErr := os.ReadFile(configPath)
	if readErr != nil {
		t.Fatalf("failed to read worker config after query: %v", readErr)
	}
	if string(updatedConfig) != originalConfig {
		t.Fatalf("expected worker config to remain unchanged, got %s", string(updatedConfig))
	}
}
