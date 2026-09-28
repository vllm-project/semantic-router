package router

import (
	"context"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gorilla/websocket"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
)

func TestGrafanaLiveRetainsCookieAuthAndOriginChecks(t *testing.T) {
	// A generic proxy setting must not turn another Origin into Grafana's own.
	t.Setenv("PROXY_OVERRIDE_ORIGIN", "true")
	store, err := auth.NewStore(filepath.Join(t.TempDir(), "auth.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	svc := auth.NewService(store, "grafana-live-test-secret", 1)
	const email, password = "grafana@example.com", "test-admin-password"
	if bootstrapErr := svc.EnsureBootstrapAdmin(context.Background(), email, password, "Test Admin"); bootstrapErr != nil {
		t.Fatal(bootstrapErr)
	}
	token, user, err := svc.Login(context.Background(), email, password)
	if err != nil {
		t.Fatal(err)
	}
	var calls atomic.Int64
	upgrader := websocket.Upgrader{} // Default: a browser Origin must match Host.
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if r.URL.Path != "/api/live/ws" || r.URL.RawQuery != "format=json" {
			t.Errorf("upstream path changed: %s", r.URL)
		}
		conn, upgradeErr := upgrader.Upgrade(w, r, nil)
		if upgradeErr != nil {
			return
		}
		defer conn.Close()
		_ = conn.WriteMessage(websocket.TextMessage, []byte(`{"ready":true}`))
	}))
	defer upstream.Close()
	mux := http.NewServeMux()
	registerGrafanaRoutes(mux, &config.Config{GrafanaURL: upstream.URL})
	dashboard := httptest.NewServer(wrapWithAuth(mux, svc))
	defer dashboard.Close()
	for _, tc := range []struct {
		name          string
		origin        string
		authenticated bool
		wantStatus    int
		wantUpstream  bool
	}{
		{"same origin", dashboard.URL, true, http.StatusSwitchingProtocols, true},
		{"other origin", "https://other.example", true, http.StatusForbidden, true},
		{"no session", dashboard.URL, false, http.StatusUnauthorized, false},
		{"role without logs read", dashboard.URL, true, http.StatusForbidden, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.name == "role without logs read" {
				if _, err := store.UpdateUserRoleOrStatus(context.Background(), user.ID, auth.RoleRead, ""); err != nil {
					t.Fatal(err)
				}
			}
			before := calls.Load()
			headers := http.Header{"Origin": []string{tc.origin}}
			if tc.authenticated {
				headers.Set("Cookie", (&http.Cookie{Name: "vsr_session", Value: token}).String())
			}
			dialer := websocket.Dialer{HandshakeTimeout: 5 * time.Second}
			conn, response, dialErr := dialer.Dial("ws"+strings.TrimPrefix(dashboard.URL, "http")+"/embedded/grafana/api/live/ws?format=json", headers)
			if response == nil {
				t.Fatalf("no handshake response: %v", dialErr)
			}
			defer response.Body.Close()
			if response.StatusCode != tc.wantStatus || (calls.Load() > before) != tc.wantUpstream {
				t.Fatalf("status=%d upstream=%v, want %d/%v", response.StatusCode, calls.Load() > before, tc.wantStatus, tc.wantUpstream)
			}
			if tc.wantStatus != http.StatusSwitchingProtocols {
				if dialErr == nil {
					_ = conn.Close()
					t.Fatal("rejected handshake unexpectedly connected")
				}
				return
			}
			if dialErr != nil {
				t.Fatal(dialErr)
			}
			defer conn.Close()
			_ = conn.SetReadDeadline(time.Now().Add(5 * time.Second))
			_, message, err := conn.ReadMessage()
			if err != nil || string(message) != `{"ready":true}` {
				t.Fatalf("upstream frame=%q error=%v", message, err)
			}
		})
	}
}
