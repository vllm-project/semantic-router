package handlers

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gorilla/websocket"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
)

func newLiveAuthRoomWebSocketServer(
	t *testing.T,
	h *OpenClawHandler,
	check func(context.Context) error,
) (*httptest.Server, <-chan context.Context) {
	t.Helper()
	requests := make(chan context.Context, 1)
	roomHandler := h.RoomByIDHandler()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests <- r.Context()
		ctx := auth.WithAuthContext(r.Context(), auth.AuthContext{UserID: "ws-user"})
		ctx = auth.WithPermissionRevalidator(ctx, check)
		roomHandler.ServeHTTP(w, r.WithContext(ctx))
	}))
	t.Cleanup(server.Close)
	return server, requests
}

func waitForWebSocketRequestCancellation(t *testing.T, requests <-chan context.Context) {
	t.Helper()
	select {
	case ctx := <-requests:
		select {
		case <-ctx.Done():
		case <-time.After(2 * time.Second):
			t.Fatal("upgraded HTTP request context was not canceled")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("WebSocket request context was not captured")
	}
}

func expectWebSocketClosedWithoutEvent(t *testing.T, conn *websocket.Conn) {
	t.Helper()
	_ = conn.SetReadDeadline(time.Now().Add(2 * time.Second))
	var outbound WSOutboundMessage
	err := conn.ReadJSON(&outbound)
	if err == nil {
		t.Fatalf("revoked connection received an event: %+v", outbound)
	}
	var closeErr *websocket.CloseError
	if !errors.As(err, &closeErr) {
		t.Fatalf("expected WebSocket closure after revocation, got %v", err)
	}
}

func seedLiveAuthWebSocketRoom(t *testing.T, h *OpenClawHandler) ClawRoomEntry {
	t.Helper()
	room := ClawRoomEntry{
		ID:        "room-ws-live-auth",
		TeamID:    "team-a",
		Name:      "Live Auth Room",
		CreatedAt: time.Now().UTC().Format(time.RFC3339),
		UpdatedAt: time.Now().UTC().Format(time.RFC3339),
	}
	if err := h.saveRooms([]ClawRoomEntry{room}); err != nil {
		t.Fatalf("seed room: %v", err)
	}
	return room
}

func TestWebSocketPermissionRevalidatorRequiresLiveAuthForAuthenticatedRequest(t *testing.T) {
	r := httptest.NewRequest(http.MethodGet, "/api/openclaw/rooms/r/ws", nil)
	if revalidate := websocketPermissionRevalidator(r); revalidate != nil {
		t.Fatal("direct handler request unexpectedly required live auth")
	}
	r = r.WithContext(auth.WithAuthContext(r.Context(), auth.AuthContext{UserID: "ws-user"}))
	if err := websocketPermissionRevalidator(r)(); err == nil {
		t.Fatal("authenticated WebSocket request without a revalidator was allowed")
	}
}

func TestRoomWebSocketRevalidatesOutboundAfterHTTPRequestCancellation(t *testing.T) {
	h := newTestOpenClawHandler(t, t.TempDir(), false)
	room := seedLiveAuthWebSocketRoom(t, h)
	var revoked atomic.Bool
	var checks atomic.Int32
	server, requests := newLiveAuthRoomWebSocketServer(t, h, func(ctx context.Context) error {
		checks.Add(1)
		if err := ctx.Err(); err != nil {
			return err
		}
		if revoked.Load() {
			return errors.New("permission revoked")
		}
		return nil
	})

	conn := dialRoomWebSocket(t, server.URL, room.ID)
	readWSConnected(t, conn)
	waitForWebSocketRequestCancellation(t, requests)

	worker := ContainerEntry{Name: "worker-a", RoleKind: "worker"}
	h.publishRoomCollaborationEvent(room.ID, workerStreamChunkCollaborationEvent(room, worker, "msg-1", "allowed", false))
	_ = conn.SetReadDeadline(time.Now().Add(2 * time.Second))
	allowed := waitForWSOutboundMessage(t, conn, "authorized chunk", func(outbound WSOutboundMessage) bool {
		return outbound.Type == WSTypeMessageChunk
	})
	if allowed.Chunk != "allowed" {
		t.Fatalf("authorized chunk = %q", allowed.Chunk)
	}

	revoked.Store(true)
	h.publishRoomCollaborationEvent(room.ID, workerStreamChunkCollaborationEvent(room, worker, "msg-1", "denied", false))
	expectWebSocketClosedWithoutEvent(t, conn)
	if got := checks.Load(); got < 3 {
		t.Fatalf("live permission checks = %d; want checks for connected and both events", got)
	}
}

func TestRoomWebSocketRejectsInboundMutationsAfterPermissionRevocation(t *testing.T) {
	cases := []struct {
		name    string
		message WSInboundMessage
	}{
		{name: "send message", message: WSInboundMessage{Type: WSTypeSendMessage, Content: "must not persist"}},
		{name: "surface event", message: WSInboundMessage{Type: WSTypeSurfaceEvent, Payload: map[string]any{"event": "must not publish"}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := newTestOpenClawHandler(t, t.TempDir(), false)
			room := seedLiveAuthWebSocketRoom(t, h)
			var revoked atomic.Bool
			server, requests := newLiveAuthRoomWebSocketServer(t, h, func(ctx context.Context) error {
				if err := ctx.Err(); err != nil {
					return err
				}
				if revoked.Load() {
					return errors.New("permission revoked")
				}
				return nil
			})

			conn := dialRoomWebSocket(t, server.URL, room.ID)
			readWSConnected(t, conn)
			waitForWebSocketRequestCancellation(t, requests)
			revoked.Store(true)
			if err := conn.WriteJSON(tc.message); err != nil {
				t.Fatalf("write inbound mutation: %v", err)
			}
			expectWebSocketClosedWithoutEvent(t, conn)
			messages, err := h.loadRoomMessages(room.ID)
			if err != nil {
				t.Fatalf("load room messages: %v", err)
			}
			if len(messages) != 0 {
				t.Fatalf("revoked mutation persisted %d room messages", len(messages))
			}
		})
	}
}
