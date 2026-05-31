package handlers

import (
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestRoomCollaborationBus_WSAndSSEReceiveSameEvent(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, false)
	room := seedCollaborationTestRoom(t, h)

	server := httptest.NewServer(h.RoomByIDHandler())
	defer server.Close()

	wsConn := dialRoomWebSocket(t, server.URL, room.ID)
	readWSConnected(t, wsConn)

	sseEvents := make(chan clawRoomStreamEvent, 4)
	sseClientID := "sse-test-client"
	h.roomSSEClientMap(room.ID).Store(sseClientID, sseEvents)
	t.Cleanup(func() {
		h.roomSSEClientMap(room.ID).Delete(sseClientID)
	})

	message := ClawRoomMessage{
		ID:         "msg-bus-1",
		RoomID:     room.ID,
		TeamID:     room.TeamID,
		SenderType: "user",
		SenderID:   "playground-user",
		SenderName: "You",
		Content:    "bus fanout",
		CreatedAt:  time.Now().UTC().Format(time.RFC3339),
	}
	h.publishRoomCollaborationEvent(room.ID, newMessageCollaborationEvent(message))

	wsOutbound := waitForWSOutboundMessage(t, wsConn, "websocket collaboration event", func(outbound WSOutboundMessage) bool {
		return outbound.Type == WSTypeNewMessage && outbound.Message != nil && outbound.Message.Content == "bus fanout"
	})
	if wsOutbound.Message == nil {
		t.Fatalf("expected websocket message payload")
	}

	select {
	case sseEvent := <-sseEvents:
		if sseEvent.Type != "message" || sseEvent.Message == nil || sseEvent.Message.Content != "bus fanout" {
			t.Fatalf("unexpected sse event: %+v", sseEvent)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for sse collaboration event")
	}
}

func TestRoomCollaborationBus_EventsDoNotLeakAcrossRooms(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, false)
	roomA := seedCollaborationTestRoomWithID(t, h, "room-a", "team-a")
	roomB := seedCollaborationTestRoomWithID(t, h, "room-b", "team-b")

	server := httptest.NewServer(h.RoomByIDHandler())
	defer server.Close()

	connB := dialRoomWebSocket(t, server.URL, roomB.ID)
	readWSConnected(t, connB)

	h.publishRoomCollaborationEvent(roomA.ID, newMessageCollaborationEvent(ClawRoomMessage{
		ID:         "msg-room-a",
		RoomID:     roomA.ID,
		TeamID:     roomA.TeamID,
		SenderType: "user",
		SenderID:   "playground-user",
		SenderName: "You",
		Content:    "room a only",
		CreatedAt:  time.Now().UTC().Format(time.RFC3339),
	}))

	_ = connB.SetReadDeadline(time.Now().Add(300 * time.Millisecond))
	var leaked WSOutboundMessage
	err := connB.ReadJSON(&leaked)
	if err == nil && leaked.Type == WSTypeNewMessage {
		t.Fatalf("room A event leaked to room B websocket: %+v", leaked)
	}
}

func TestRoomCollaborationBus_WSDisconnectDoesNotBreakSSE(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, false)
	room := seedCollaborationTestRoom(t, h)

	server := httptest.NewServer(h.RoomByIDHandler())
	defer server.Close()

	wsConn := dialRoomWebSocket(t, server.URL, room.ID)
	readWSConnected(t, wsConn)
	_ = wsConn.Close()

	sseEvents := make(chan clawRoomStreamEvent, 4)
	sseClientID := "sse-after-ws-close"
	h.roomSSEClientMap(room.ID).Store(sseClientID, sseEvents)
	t.Cleanup(func() {
		h.roomSSEClientMap(room.ID).Delete(sseClientID)
	})

	h.publishRoomCollaborationEvent(room.ID, newMessageCollaborationEvent(ClawRoomMessage{
		ID:         "msg-sse-only",
		RoomID:     room.ID,
		TeamID:     room.TeamID,
		SenderType: "system",
		SenderID:   "clawos-system",
		SenderName: "ClawOS",
		Content:    "sse still works",
		CreatedAt:  time.Now().UTC().Format(time.RFC3339),
	}))

	select {
	case sseEvent := <-sseEvents:
		if sseEvent.Message == nil || sseEvent.Message.Content != "sse still works" {
			t.Fatalf("unexpected sse event after ws disconnect: %+v", sseEvent)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("expected sse client to receive event after websocket disconnect")
	}
}

func TestRoomWebSocket_MultipleClientsReceiveSameEvent(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, false)
	room := seedCollaborationTestRoom(t, h)

	server := httptest.NewServer(h.RoomByIDHandler())
	defer server.Close()

	connA := dialRoomWebSocket(t, server.URL, room.ID)
	readWSConnected(t, connA)
	connB := dialRoomWebSocket(t, server.URL, room.ID)
	readWSConnected(t, connB)

	h.publishRoomCollaborationEvent(room.ID, newMessageCollaborationEvent(ClawRoomMessage{
		ID:         "msg-multi",
		RoomID:     room.ID,
		TeamID:     room.TeamID,
		SenderType: "user",
		SenderID:   "playground-user",
		SenderName: "You",
		Content:    "multi client",
		CreatedAt:  time.Now().UTC().Format(time.RFC3339),
	}))

	outA := waitForWSOutboundMessage(t, connA, "client A event", func(outbound WSOutboundMessage) bool {
		return outbound.Type == WSTypeNewMessage && outbound.Message != nil && outbound.Message.Content == "multi client"
	})
	outB := waitForWSOutboundMessage(t, connB, "client B event", func(outbound WSOutboundMessage) bool {
		return outbound.Type == WSTypeNewMessage && outbound.Message != nil && outbound.Message.Content == "multi client"
	})
	if outA.Message == nil || outB.Message == nil {
		t.Fatalf("expected both websocket clients to receive the event")
	}
}

func TestRoomCollaborationBus_HTTPPostAndWSSendEquivalent(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, false)
	room := seedCollaborationTestRoom(t, h)

	httpCreated := postRoomMessage(t, h, room.ID, `{
		"senderType":"user",
		"senderId":"playground-user",
		"senderName":"You",
		"content":"http path"
	}`)

	server := httptest.NewServer(h.RoomByIDHandler())
	defer server.Close()

	conn := dialRoomWebSocket(t, server.URL, room.ID)
	readWSConnected(t, conn)

	if writeErr := conn.WriteJSON(WSInboundMessage{
		Type:       WSTypeSendMessage,
		Content:    "ws path",
		SenderType: "user",
		SenderName: "You",
		SenderID:   "playground-user",
	}); writeErr != nil {
		t.Fatalf("failed to send websocket message: %v", writeErr)
	}

	wsOutbound := waitForWSOutboundMessage(t, conn, "websocket send event", func(outbound WSOutboundMessage) bool {
		return outbound.Type == WSTypeNewMessage && outbound.Message != nil && outbound.Message.Content == "ws path"
	})

	if httpCreated.Content != "http path" {
		t.Fatalf("unexpected http-created message: %+v", httpCreated)
	}
	if wsOutbound.Message == nil || wsOutbound.Message.Content != "ws path" {
		t.Fatalf("unexpected websocket-created message: %+v", wsOutbound)
	}

	messages, err := h.loadRoomMessages(room.ID)
	if err != nil {
		t.Fatalf("failed to load room messages: %v", err)
	}
	if len(messages) != 2 {
		t.Fatalf("expected two persisted room messages, got %d", len(messages))
	}
}

func TestWorkerStreamChunkCollaborationEvent_IncludesParticipantIdentity(t *testing.T) {
	room := ClawRoomEntry{ID: "room-1", TeamID: "team-1"}
	worker := ContainerEntry{Name: "worker-a", RoleKind: "worker"}
	event := workerStreamChunkCollaborationEvent(room, worker, "msg-1", "He", false)
	if event.ParticipantType != "worker" || event.ParticipantID != "worker-a" {
		t.Fatalf("unexpected participant identity: %+v", event)
	}
	if event.SessionUser != "room-1:worker-a" {
		t.Fatalf("expected room-scoped session user, got %q", event.SessionUser)
	}
	if event.Chunk != "He" || event.Status != "streaming" {
		t.Fatalf("unexpected chunk payload: %+v", event)
	}
}

func seedCollaborationTestRoom(t *testing.T, h *OpenClawHandler) ClawRoomEntry {
	t.Helper()
	return seedCollaborationTestRoomWithID(t, h, "room-collab", "team-collab")
}

func seedCollaborationTestRoomWithID(t *testing.T, h *OpenClawHandler, roomID, teamID string) ClawRoomEntry {
	t.Helper()
	room := ClawRoomEntry{
		ID:        roomID,
		TeamID:    teamID,
		Name:      strings.ToUpper(roomID),
		CreatedAt: time.Now().UTC().Format(time.RFC3339),
		UpdatedAt: time.Now().UTC().Format(time.RFC3339),
	}
	if err := h.saveRooms([]ClawRoomEntry{room}); err != nil {
		t.Fatalf("failed to seed room: %v", err)
	}
	return room
}
