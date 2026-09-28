package handlers

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
)

func TestRoomStreamStopsBeforeSendingEventAfterPermissionRevocation(t *testing.T) {
	h := newTestOpenClawHandler(t, t.TempDir(), false)
	room := seedLiveAuthWebSocketRoom(t, h)
	var revoked atomic.Bool
	checkedAfterRevocation := make(chan struct{}, 1)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ctx = auth.WithAuthContext(ctx, auth.AuthContext{UserID: "sse-user"})
	ctx = auth.WithPermissionRevalidator(ctx, func(context.Context) error {
		if revoked.Load() {
			select {
			case checkedAfterRevocation <- struct{}{}:
			default:
			}
			return errors.New("permission revoked")
		}
		return nil
	})
	req := httptest.NewRequest(http.MethodGet, "/api/openclaw/rooms/"+room.ID+"/stream", nil).WithContext(ctx)
	recorder := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		defer close(done)
		h.handleRoomStream(recorder, req, room.ID)
	}()

	var events chan clawRoomStreamEvent
	deadline := time.NewTimer(2 * time.Second)
	defer deadline.Stop()
	for events == nil {
		h.roomSSEClientMap(room.ID).Range(func(_, value any) bool {
			events = value.(chan clawRoomStreamEvent)
			return false
		})
		if events != nil {
			break
		}
		select {
		case <-done:
			t.Fatal("room stream returned before registering SSE client")
		case <-deadline.C:
			t.Fatal("room stream did not register SSE client")
		case <-time.After(5 * time.Millisecond):
		}
	}

	revoked.Store(true)
	events <- clawRoomStreamEvent{Type: "message_chunk", RoomID: room.ID, Chunk: "secret-after-revoke"}
	select {
	case <-checkedAfterRevocation:
	case <-time.After(2 * time.Second):
		t.Fatal("SSE event was not revalidated")
	}
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("SSE stream remained open after revocation")
	}
	if body := recorder.Body.String(); strings.Contains(body, "secret-after-revoke") {
		t.Fatalf("revoked stream received an event: %s", body)
	}
}
