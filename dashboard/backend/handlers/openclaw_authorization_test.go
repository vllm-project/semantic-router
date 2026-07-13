package handlers

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
)

func newOpenClawAuthorizationUser(
	t *testing.T,
	role string,
) (*auth.Service, *auth.User, string) {
	t.Helper()
	store, err := auth.NewStore(filepath.Join(t.TempDir(), "auth.sqlite"))
	if err != nil {
		t.Fatalf("NewStore() error = %v", err)
	}
	svc, err := auth.NewService(store, "kZrW6u9GJ4cV7nQ2pL8sT1mX5bD0fH3a", 1)
	if err != nil {
		_ = store.Close()
		t.Fatalf("NewService() error = %v", err)
	}
	t.Cleanup(func() { _ = svc.Close() })
	hash, err := svc.HashPasswordForUser("room-user@example.com", "unique room user password")
	if err != nil {
		t.Fatalf("HashPasswordForUser() error = %v", err)
	}
	user, err := svc.BootstrapRegister(
		context.Background(),
		"room-user@example.com",
		"Room User",
		hash,
	)
	if err != nil {
		t.Fatalf("BootstrapRegister() error = %v", err)
	}
	if role != auth.RoleAdmin {
		if _, createErr := store.CreateUser(
			context.Background(),
			"room-manager@example.com",
			"Room Manager",
			hash,
			auth.RoleAdmin,
			"active",
		); createErr != nil {
			t.Fatalf("CreateUser(manager) error = %v", createErr)
		}
		user, err = store.UpdateUserRoleOrStatus(context.Background(), user.ID, role, "")
		if err != nil {
			t.Fatalf("UpdateUserRoleOrStatus() error = %v", err)
		}
	}
	token, user, err := svc.Login(context.Background(), user.Email, "unique room user password")
	if err != nil {
		t.Fatalf("Login() error = %v", err)
	}
	return svc, user, token
}

func dialAuthorizedRoomWebSocket(
	t *testing.T,
	serverURL, roomID, token string,
) *websocket.Conn {
	t.Helper()
	wsURL := "ws" + strings.TrimPrefix(serverURL, "http") + "/api/openclaw/rooms/" + roomID + "/ws"
	conn, response, err := websocket.DefaultDialer.Dial(wsURL, http.Header{
		"Authorization": []string{"Bearer " + token},
		"Origin":        []string{serverURL},
	})
	if err != nil {
		if response != nil && response.Body != nil {
			body, _ := io.ReadAll(response.Body)
			_ = response.Body.Close()
			t.Fatalf("dial authorized room WebSocket: %v; status=%d body=%s", err, response.StatusCode, body)
		}
		t.Fatalf("dial authorized room WebSocket: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	return conn
}

func TestOpenClawReadPermissionCanObserveButCannotMutateRoom(t *testing.T) {
	svc, _, token := newOpenClawAuthorizationUser(t, auth.RoleRead)
	h := newTestOpenClawHandler(t, t.TempDir(), false)
	room := seedCollaborationTestRoom(t, h)
	server := httptest.NewServer(auth.AuthenticateRequest(svc)(h.RoomByIDHandler()))
	t.Cleanup(server.Close)

	postRequest, err := http.NewRequest(
		http.MethodPost,
		server.URL+"/api/openclaw/rooms/"+room.ID+"/messages",
		strings.NewReader(`{"senderType":"system","content":"must not persist"}`),
	)
	if err != nil {
		t.Fatalf("NewRequest() error = %v", err)
	}
	postRequest.Header.Set("Authorization", "Bearer "+token)
	postRequest.Header.Set("Content-Type", "application/json")
	postResponse, err := http.DefaultClient.Do(postRequest)
	if err != nil {
		t.Fatalf("POST room message: %v", err)
	}
	defer postResponse.Body.Close()
	if postResponse.StatusCode != http.StatusForbidden {
		body, _ := io.ReadAll(postResponse.Body)
		t.Fatalf("POST status = %d, want 403; body=%s", postResponse.StatusCode, body)
	}

	conn := dialAuthorizedRoomWebSocket(t, server.URL, room.ID, token)
	readWSConnected(t, conn)
	if writeErr := conn.WriteJSON(WSInboundMessage{
		Type:       WSTypeSendMessage,
		Content:    "read user websocket mutation",
		SenderType: "system",
		SenderID:   "clawos-system",
	}); writeErr != nil {
		t.Fatalf("WriteJSON() error = %v", writeErr)
	}
	forbidden := waitForWSOutboundMessage(t, conn, "permission error", func(outbound WSOutboundMessage) bool {
		return outbound.Type == WSTypeError && strings.Contains(outbound.Error, "openclaw.use")
	})
	if forbidden.Type != WSTypeError {
		t.Fatalf("unexpected response: %+v", forbidden)
	}
	messages, err := h.loadRoomMessages(room.ID)
	if err != nil {
		t.Fatalf("loadRoomMessages() error = %v", err)
	}
	if len(messages) != 0 {
		t.Fatalf("read-only mutations persisted messages: %+v", messages)
	}
}

func TestOpenClawBrowserMutationBindsServerSideUserIdentity(t *testing.T) {
	svc, user, token := newOpenClawAuthorizationUser(t, auth.RoleWrite)
	h := newTestOpenClawHandler(t, t.TempDir(), false)
	room := seedCollaborationTestRoom(t, h)
	server := httptest.NewServer(auth.AuthenticateRequest(svc)(h.RoomByIDHandler()))
	t.Cleanup(server.Close)

	conn := dialAuthorizedRoomWebSocket(t, server.URL, room.ID, token)
	readWSConnected(t, conn)
	if writeErr := conn.WriteJSON(WSInboundMessage{
		Type:       WSTypeSendMessage,
		Content:    "identity-bound message",
		SenderType: "system",
		SenderID:   "clawos-system",
		SenderName: "Forged System",
	}); writeErr != nil {
		t.Fatalf("WriteJSON() error = %v", writeErr)
	}
	outbound := waitForWSOutboundMessage(t, conn, "identity-bound message", func(outbound WSOutboundMessage) bool {
		return outbound.Type == WSTypeNewMessage && outbound.Message != nil && outbound.Message.Content == "identity-bound message"
	})
	if outbound.Message == nil {
		t.Fatal("missing persisted message")
	}
	if outbound.Message.SenderType != "user" || outbound.Message.SenderID != user.ID || outbound.Message.SenderName != user.Name {
		t.Fatalf("browser-controlled identity was not overwritten: %+v", outbound.Message)
	}
}

func TestOpenClawWebSocketClosesImmediatelyWhenSessionRevoked(t *testing.T) {
	svc, _, token := newOpenClawAuthorizationUser(t, auth.RoleRead)
	h := newTestOpenClawHandler(t, t.TempDir(), false)
	room := seedCollaborationTestRoom(t, h)
	server := httptest.NewServer(auth.AuthenticateRequest(svc)(h.RoomByIDHandler()))
	t.Cleanup(server.Close)

	conn := dialAuthorizedRoomWebSocket(t, server.URL, room.ID, token)
	readWSConnected(t, conn)
	if revokeErr := svc.RevokeToken(context.Background(), token); revokeErr != nil {
		t.Fatalf("RevokeToken() error = %v", revokeErr)
	}
	_ = conn.SetReadDeadline(time.Now().Add(time.Second))
	var outbound WSOutboundMessage
	err := conn.ReadJSON(&outbound)
	if err == nil {
		t.Fatalf("revoked WebSocket remained readable: %+v", outbound)
	}
	var closeErr *websocket.CloseError
	if errors.As(err, &closeErr) && closeErr.Code != websocket.ClosePolicyViolation {
		t.Fatalf("close code = %d, want %d", closeErr.Code, websocket.ClosePolicyViolation)
	}
}

func TestOpenClawSSEClosesImmediatelyWhenSessionRevoked(t *testing.T) {
	svc, _, token := newOpenClawAuthorizationUser(t, auth.RoleRead)
	h := newTestOpenClawHandler(t, t.TempDir(), false)
	room := seedCollaborationTestRoom(t, h)
	server := httptest.NewServer(auth.AuthenticateRequest(svc)(h.RoomByIDHandler()))
	t.Cleanup(server.Close)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	request, err := http.NewRequestWithContext(
		ctx,
		http.MethodGet,
		server.URL+"/api/openclaw/rooms/"+room.ID+"/stream",
		nil,
	)
	if err != nil {
		t.Fatalf("NewRequestWithContext() error = %v", err)
	}
	request.Header.Set("Authorization", "Bearer "+token)
	request.Header.Set("Accept", "text/event-stream")
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatalf("open SSE stream: %v", err)
	}
	defer response.Body.Close()
	reader := bufio.NewReader(response.Body)
	if line, err := reader.ReadString('\n'); err != nil || !strings.HasPrefix(line, "event: connected") {
		t.Fatalf("first SSE line = %q, err=%v", line, err)
	}
	if revokeErr := svc.RevokeToken(context.Background(), token); revokeErr != nil {
		t.Fatalf("RevokeToken() error = %v", revokeErr)
	}
	for {
		_, readErr := reader.ReadString('\n')
		if readErr != nil {
			break
		}
		select {
		case <-ctx.Done():
			t.Fatal("revoked SSE stream did not close")
		default:
		}
	}
}

func TestOpenClawHTTPBrowserIdentityCannotSpoofSystem(t *testing.T) {
	svc, user, token := newOpenClawAuthorizationUser(t, auth.RoleWrite)
	h := newTestOpenClawHandler(t, t.TempDir(), false)
	room := seedCollaborationTestRoom(t, h)
	handler := auth.AuthenticateRequest(svc)(h.RoomByIDHandler())
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(
		http.MethodPost,
		"https://dashboard.example.com/api/openclaw/rooms/"+room.ID+"/messages",
		strings.NewReader(`{"senderType":"system","senderId":"clawos-system","senderName":"Forged","content":"bound HTTP actor"}`),
	)
	request.Header.Set("Authorization", "Bearer "+token)
	request.Header.Set("Content-Type", "application/json")
	handler.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusCreated {
		t.Fatalf("status = %d, want 201; body=%s", recorder.Code, recorder.Body.String())
	}
	var created ClawRoomMessage
	if err := json.Unmarshal(recorder.Body.Bytes(), &created); err != nil {
		t.Fatalf("decode created message: %v", err)
	}
	if created.SenderType != "user" || created.SenderID != user.ID || created.SenderName != user.Name {
		t.Fatalf("HTTP browser-controlled identity was not overwritten: %+v", created)
	}
}
