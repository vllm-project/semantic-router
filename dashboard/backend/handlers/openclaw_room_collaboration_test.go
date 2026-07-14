package handlers

import (
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
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

func TestRoomCollaborationBus_ConcurrentWSDisconnectAndFanOut(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, false)
	room := seedCollaborationTestRoom(t, h)

	server := httptest.NewServer(h.RoomByIDHandler())
	defer server.Close()

	victimConn := dialRoomWebSocket(t, server.URL, room.ID)
	readWSConnected(t, victimConn)
	victim := waitForOnlyRoomWSClient(t, h, room.ID)

	survivorConn := dialRoomWebSocket(t, server.URL, room.ID)
	readWSConnected(t, survivorConn)

	sseEvents := make(chan clawRoomStreamEvent, 1)
	sseClientID := "sse-during-ws-close"
	h.roomSSEClientMap(room.ID).Store(sseClientID, sseEvents)
	t.Cleanup(func() {
		h.roomSSEClientMap(room.ID).Delete(sseClientID)
	})

	event := newMessageCollaborationEvent(ClawRoomMessage{
		ID:         "msg-concurrent-close",
		RoomID:     room.ID,
		TeamID:     room.TeamID,
		SenderType: "system",
		SenderID:   "clawos-system",
		SenderName: "ClawOS",
		Content:    "survives concurrent close",
		CreatedAt:  time.Now().UTC().Format(time.RFC3339),
	})

	// Force both operations to contend on the ownership lock. Whichever wins
	// after the barrier, enqueue and channel close must complete serially.
	victim.closeMu.Lock()
	start := make(chan struct{})
	ready := sync.WaitGroup{}
	done := sync.WaitGroup{}
	ready.Add(2)
	done.Add(2)
	go func() {
		defer done.Done()
		ready.Done()
		<-start
		h.publishRoomCollaborationEvent(room.ID, event)
	}()
	go func() {
		defer done.Done()
		ready.Done()
		<-start
		victim.close()
	}()
	ready.Wait()
	close(start)
	victim.closeMu.Unlock()

	completed := make(chan struct{})
	go func() {
		done.Wait()
		close(completed)
	}()
	select {
	case <-completed:
	case <-time.After(2 * time.Second):
		t.Fatal("concurrent websocket close and fan-out did not complete")
	}

	if got := victim.enqueue(WSOutboundMessage{Type: WSTypePong}); got != wsEnqueueClosed {
		t.Fatalf("expected closed client to reject enqueue, got %v", got)
	}
	if _, exists := h.roomWSClientMap(room.ID).Load(victim.clientID); exists {
		t.Fatal("closed websocket client remained registered")
	}

	_ = survivorConn.SetReadDeadline(time.Now().Add(2 * time.Second))
	waitForWSOutboundMessage(t, survivorConn, "surviving websocket event", func(outbound WSOutboundMessage) bool {
		return outbound.Type == WSTypeNewMessage &&
			outbound.Message != nil &&
			outbound.Message.Content == "survives concurrent close"
	})
	select {
	case sseEvent := <-sseEvents:
		if sseEvent.Message == nil || sseEvent.Message.Content != "survives concurrent close" {
			t.Fatalf("unexpected SSE event during websocket close: %+v", sseEvent)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("SSE fan-out stalled during concurrent websocket close")
	}
}

func TestWSClientEnqueue_BufferFullDoesNotBlock(t *testing.T) {
	client := &WSClient{send: make(chan WSOutboundMessage, 1)}
	client.send <- WSOutboundMessage{Type: WSTypeConnected}

	result := make(chan wsEnqueueResult, 1)
	go func() {
		result <- client.enqueue(WSOutboundMessage{Type: WSTypePong})
	}()

	select {
	case got := <-result:
		if got != wsEnqueueFull {
			t.Fatalf("expected full result, got %v", got)
		}
	case <-time.After(time.Second):
		t.Fatal("enqueue blocked on a full websocket buffer")
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

func TestRoomCollaboration_WorkerMessageUpdatedCarriesSessionIdentity(t *testing.T) {
	room := ClawRoomEntry{ID: "room-1", TeamID: "team-1"}
	reply := ClawRoomMessage{
		ID:         "msg-worker-final",
		RoomID:     room.ID,
		TeamID:     room.TeamID,
		SenderType: "worker",
		SenderID:   "worker-a",
		SenderName: "Worker A",
		Content:    "done",
		CreatedAt:  time.Now().UTC().Format(time.RFC3339),
	}
	event := messageUpdatedCollaborationEvent(room, reply)
	if event.SessionUser != "room-1:worker-a" {
		t.Fatalf("expected room-scoped session user on message_updated, got %q", event.SessionUser)
	}
	if event.ParticipantType != "worker" || event.ParticipantID != "worker-a" {
		t.Fatalf("unexpected participant identity: %+v", event)
	}
}

func TestRoomCollaboration_SurfaceEventReachAllWSClients(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, false)
	room := seedCollaborationTestRoom(t, h)

	server := httptest.NewServer(h.RoomByIDHandler())
	defer server.Close()

	connA := dialRoomWebSocket(t, server.URL, room.ID)
	readWSConnected(t, connA)
	connB := dialRoomWebSocket(t, server.URL, room.ID)
	readWSConnected(t, connB)

	if writeErr := connA.WriteJSON(WSInboundMessage{
		Type:       WSTypeSurfaceEvent,
		SenderType: "worker",
		SenderID:   "worker-a",
		Payload:    map[string]any{"kind": "tool_call", "name": "search"},
	}); writeErr != nil {
		t.Fatalf("failed to send surface_event: %v", writeErr)
	}

	matchSurface := func(outbound WSOutboundMessage) bool {
		return outbound.Type == WSTypeSurfaceEvent &&
			outbound.ParticipantID == "worker-a" &&
			outbound.Payload != nil &&
			outbound.Payload["kind"] == "tool_call"
	}
	outA := waitForWSOutboundMessage(t, connA, "client A surface event", matchSurface)
	outB := waitForWSOutboundMessage(t, connB, "client B surface event", matchSurface)
	if outA.Payload == nil || outB.Payload == nil {
		t.Fatalf("expected surface payload on both clients")
	}
}

func TestRoomCollaboration_FullFlowUserWorkerSurface(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, false)
	room := seedCollaborationTestRoom(t, h)
	worker := ContainerEntry{Name: "worker-a", RoleKind: "worker"}

	server := httptest.NewServer(h.RoomByIDHandler())
	defer server.Close()

	userConn := dialRoomWebSocket(t, server.URL, room.ID)
	readWSConnected(t, userConn)
	embeddedConn := dialRoomWebSocket(t, server.URL, room.ID)
	readWSConnected(t, embeddedConn)
	observerConn := dialRoomWebSocket(t, server.URL, room.ID)
	readWSConnected(t, observerConn)

	messageID := "room-msg-flow"
	h.publishRoomCollaborationEvent(
		room.ID,
		workerStreamChunkCollaborationEvent(room, worker, messageID, "Hello", false),
	)

	for _, conn := range []*websocket.Conn{userConn, embeddedConn, observerConn} {
		outbound := waitForWSOutboundMessage(t, conn, "worker chunk", func(message WSOutboundMessage) bool {
			return message.Type == WSTypeMessageChunk &&
				message.Chunk == "Hello" &&
				message.SessionUser == "room-collab:worker-a"
		})
		if outbound.ParticipantID != "worker-a" {
			t.Fatalf("unexpected chunk participant: %+v", outbound)
		}
	}

	if writeErr := embeddedConn.WriteJSON(WSInboundMessage{
		Type:       WSTypeSurfaceEvent,
		SenderType: "worker",
		SenderID:   "worker-a",
		Payload:    map[string]any{"kind": "status", "value": "thinking"},
	}); writeErr != nil {
		t.Fatalf("failed to send embedded surface_event: %v", writeErr)
	}

	for _, conn := range []*websocket.Conn{userConn, observerConn} {
		waitForWSOutboundMessage(t, conn, "surface fanout", func(message WSOutboundMessage) bool {
			return message.Type == WSTypeSurfaceEvent && message.Payload != nil && message.Payload["value"] == "thinking"
		})
	}

	reply := ClawRoomMessage{
		ID:         messageID,
		RoomID:     room.ID,
		TeamID:     room.TeamID,
		SenderType: "worker",
		SenderID:   "worker-a",
		SenderName: "Worker A",
		Content:    "Hello",
		CreatedAt:  time.Now().UTC().Format(time.RFC3339),
	}
	h.publishRoomCollaborationEvent(room.ID, messageUpdatedCollaborationEvent(room, reply))

	for _, conn := range []*websocket.Conn{userConn, embeddedConn, observerConn} {
		updated := waitForWSOutboundMessage(t, conn, "message_updated", func(message WSOutboundMessage) bool {
			return message.Type == WSTypeMessageUpdated &&
				message.Message != nil &&
				message.Message.Content == "Hello" &&
				message.SessionUser == "room-collab:worker-a"
		})
		if updated.ParticipantID != "worker-a" {
			t.Fatalf("unexpected updated participant: %+v", updated)
		}
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

func waitForOnlyRoomWSClient(t *testing.T, h *OpenClawHandler, roomID string) *WSClient {
	t.Helper()

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		var clients []*WSClient
		h.roomWSClientMap(roomID).Range(func(_, value any) bool {
			if client, ok := value.(*WSClient); ok {
				clients = append(clients, client)
			}
			return true
		})
		if len(clients) == 1 {
			return clients[0]
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("timed out waiting for websocket client registration")
	return nil
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
