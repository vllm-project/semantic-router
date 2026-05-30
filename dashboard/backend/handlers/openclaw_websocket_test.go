package handlers

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

func dialRoomWebSocket(t *testing.T, serverURL, roomID string) *websocket.Conn {
	t.Helper()

	wsURL := "ws" + strings.TrimPrefix(serverURL, "http") + "/api/openclaw/rooms/" + roomID + "/ws"
	conn, resp, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		t.Fatalf("failed to connect websocket: %v", err)
	}
	if resp != nil && resp.Body != nil {
		t.Cleanup(func() { _ = resp.Body.Close() })
	}
	t.Cleanup(func() { _ = conn.Close() })
	return conn
}

func readWSConnected(t *testing.T, conn *websocket.Conn) {
	t.Helper()

	_ = conn.SetReadDeadline(time.Now().Add(2 * time.Second))
	waitForWSOutboundMessage(t, conn, "connected message", func(outbound WSOutboundMessage) bool {
		return outbound.Type == WSTypeConnected
	})
	_ = conn.SetReadDeadline(time.Time{})
}

func collectWSOutboundUntil(
	t *testing.T,
	conn *websocket.Conn,
	timeout time.Duration,
	stop func(WSOutboundMessage) bool,
) []WSOutboundMessage {
	t.Helper()

	events := make(chan WSOutboundMessage, 64)
	errs := make(chan error, 1)
	go func() {
		for {
			var outbound WSOutboundMessage
			if err := conn.ReadJSON(&outbound); err != nil {
				errs <- err
				return
			}
			events <- outbound
		}
	}()

	collected := make([]WSOutboundMessage, 0)
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		select {
		case outbound := <-events:
			collected = append(collected, outbound)
			if stop(outbound) {
				return collected
			}
		case err := <-errs:
			t.Fatalf("websocket read failed: %v", err)
		case <-time.After(50 * time.Millisecond):
		}
	}
	return collected
}

func TestWorkerStreamChunkCollaborationEvent_WSOutboundShape(t *testing.T) {
	room := ClawRoomEntry{ID: "room-1", TeamID: "team-1"}
	worker := ContainerEntry{Name: "worker-a", RoleKind: "worker"}

	streaming := collaborationEventToWSOutbound(
		workerStreamChunkCollaborationEvent(room, worker, "room-msg-1", "He", false),
	)
	if streaming.Type != WSTypeMessageChunk {
		t.Fatalf("expected message_chunk, got %q", streaming.Type)
	}
	if streaming.Chunk != "He" {
		t.Fatalf("expected chunk payload, got %q", streaming.Chunk)
	}
	if streaming.Status != "streaming" {
		t.Fatalf("expected streaming status, got %q", streaming.Status)
	}

	complete := collaborationEventToWSOutbound(
		workerStreamChunkCollaborationEvent(room, worker, "room-msg-1", "llo", true),
	)
	if complete.Chunk != "llo" {
		t.Fatalf("expected final chunk payload, got %q", complete.Chunk)
	}
	if complete.Status != "complete" {
		t.Fatalf("expected complete status, got %q", complete.Status)
	}
}

func TestRoomWebSocket_ConnectReplayLastEvent(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, false)

	workerSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assertChatCompletionPath(t, r)
		encodeOpenAIResponse(w, "unused")
	}))
	defer workerSrv.Close()

	room := seedRoomTeamWithLeaderAndWorker(t, h, tempDir, "team-ws-replay", "WS Replay Team", workerSrv.URL)
	created := postRoomMessage(t, h, room.ID, `{
		"senderType":"user",
		"senderId":"playground-user",
		"senderName":"You",
		"content":"seed message for replay"
	}`)
	if created.Content != "seed message for replay" {
		t.Fatalf("unexpected seeded message: %+v", created)
	}

	server := httptest.NewServer(h.RoomByIDHandler())
	defer server.Close()

	conn := dialRoomWebSocket(t, server.URL, room.ID)
	readWSConnected(t, conn)

	_ = conn.SetReadDeadline(time.Now().Add(2 * time.Second))
	replay := waitForWSOutboundMessage(t, conn, "replay message", func(outbound WSOutboundMessage) bool {
		return outbound.Type == WSTypeNewMessage && outbound.Message != nil
	})
	if replay.Message == nil || replay.Message.Content != "seed message for replay" {
		t.Fatalf("unexpected replay message: %+v", replay)
	}
}

func TestPublishRoomCollaborationEvent_ChunkReachConnectedClient(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, false)
	room := ClawRoomEntry{
		ID:        "room-ws-chunk",
		TeamID:    "team-a",
		Name:      "Chunk Room",
		CreatedAt: time.Now().UTC().Format(time.RFC3339),
		UpdatedAt: time.Now().UTC().Format(time.RFC3339),
	}
	if err := h.saveRooms([]ClawRoomEntry{room}); err != nil {
		t.Fatalf("failed to seed room: %v", err)
	}

	server := httptest.NewServer(h.RoomByIDHandler())
	defer server.Close()

	conn := dialRoomWebSocket(t, server.URL, room.ID)
	readWSConnected(t, conn)

	worker := ContainerEntry{Name: "worker-a", RoleKind: "worker"}
	h.publishRoomCollaborationEvent(
		room.ID,
		workerStreamChunkCollaborationEvent(room, worker, "room-msg-test", "partial", false),
	)

	_ = conn.SetReadDeadline(time.Now().Add(2 * time.Second))
	outbound := waitForWSOutboundMessage(t, conn, "chunk event", func(message WSOutboundMessage) bool {
		return message.Type == WSTypeMessageChunk && message.Chunk == "partial"
	})
	if outbound.MessageID != "room-msg-test" {
		t.Fatalf("unexpected chunk event: %+v", outbound)
	}
}

func TestPublishRoomCollaborationEvent_StreamingChunkSequence(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, false)
	room := ClawRoomEntry{
		ID:        "room-ws-stream-seq",
		TeamID:    "team-a",
		Name:      "Stream Sequence Room",
		CreatedAt: time.Now().UTC().Format(time.RFC3339),
		UpdatedAt: time.Now().UTC().Format(time.RFC3339),
	}
	if err := h.saveRooms([]ClawRoomEntry{room}); err != nil {
		t.Fatalf("failed to seed room: %v", err)
	}

	server := httptest.NewServer(h.RoomByIDHandler())
	defer server.Close()

	conn := dialRoomWebSocket(t, server.URL, room.ID)
	readWSConnected(t, conn)

	worker := ContainerEntry{Name: "worker-a", RoleKind: "worker"}
	messageID := "room-msg-stream"
	for _, chunk := range []string{"He", "llo", " world"} {
		h.publishRoomCollaborationEvent(
			room.ID,
			workerStreamChunkCollaborationEvent(room, worker, messageID, chunk, false),
		)
	}
	h.publishRoomCollaborationEvent(
		room.ID,
		workerStreamChunkCollaborationEvent(room, worker, messageID, "", true),
	)
	reply := ClawRoomMessage{
		ID:         messageID,
		RoomID:     room.ID,
		TeamID:     room.TeamID,
		SenderType: "worker",
		SenderID:   "worker-a",
		SenderName: "Worker A",
		Content:    "Hello world",
		CreatedAt:  time.Now().UTC().Format(time.RFC3339),
	}
	h.publishRoomCollaborationEvent(room.ID, messageUpdatedCollaborationEvent(reply))

	events := collectWSOutboundUntil(t, conn, 2*time.Second, func(outbound WSOutboundMessage) bool {
		return outbound.Type == WSTypeMessageUpdated
	})

	var chunks []string
	var updated *ClawRoomMessage
	for _, outbound := range events {
		switch outbound.Type {
		case WSTypeMessageChunk:
			if outbound.Chunk != "" {
				chunks = append(chunks, outbound.Chunk)
			}
		case WSTypeMessageUpdated:
			if outbound.Message != nil {
				copyMsg := *outbound.Message
				updated = &copyMsg
			}
		}
	}

	if strings.Join(chunks, "") != "Hello world" {
		t.Fatalf("expected streamed chunks %q, got %q (%v)", "Hello world", strings.Join(chunks, ""), chunks)
	}
	if updated == nil || updated.Content != "Hello world" {
		t.Fatalf("expected message_updated payload, got %+v", updated)
	}
}

func TestWSOutboundFromLastRoomEvent(t *testing.T) {
	msg := ClawRoomMessage{ID: "m1", Content: "hello"}
	event := clawRoomStreamEvent{Type: "message", Message: &msg}

	outbound, ok := wsOutboundFromLastRoomEvent("room-a", event)
	if !ok {
		t.Fatalf("expected replay conversion to succeed")
	}
	if outbound.Type != WSTypeNewMessage {
		t.Fatalf("expected new_message replay type, got %q", outbound.Type)
	}
	if outbound.Message == nil || outbound.Message.Content != "hello" {
		t.Fatalf("unexpected replay payload: %+v", outbound)
	}
	if outbound.Message == event.Message {
		t.Fatalf("expected replay message to be copied")
	}
}
