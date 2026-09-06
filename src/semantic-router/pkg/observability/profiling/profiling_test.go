package profiling

import (
	"fmt"
	"io"
	"net/http"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestStartServesProfilesOnEphemeralPort(t *testing.T) {
	server := startForTest(t, config.ProfilingConfig{Enabled: true, Port: 0, Bind: "127.0.0.1"})

	for _, path := range []string{"/debug/pprof/", "/debug/pprof/heap", "/debug/pprof/goroutine", "/debug/pprof/cmdline"} {
		status, body := get(t, server.Addr(), path)
		if status != http.StatusOK {
			t.Fatalf("GET %s returned %d, want %d", path, status, http.StatusOK)
		}
		if path == "/debug/pprof/heap" && len(body) == 0 {
			t.Fatalf("GET %s returned an empty profile", path)
		}
	}
}

func TestStartRejectsEmptyBind(t *testing.T) {
	server, err := Start(config.ProfilingConfig{Enabled: true, Port: 0, Bind: ""})
	if err == nil {
		if server != nil {
			_ = server.Close()
		}
		t.Fatal("Start accepted an empty bind and would have listened on every interface")
	}
	if server != nil {
		_ = server.Close()
		t.Fatal("Start returned a live server for an empty bind")
	}
}

func TestValidatePort(t *testing.T) {
	reserved := []int{8080, 9190, 50051}

	for _, port := range []int{0, 6060, 6061} {
		if err := ValidatePort(port, reserved...); err != nil {
			t.Fatalf("ValidatePort(%d) rejected a usable port: %v", port, err)
		}
	}

	for _, port := range []int{8080, 9190, 50051, -1, 70000} {
		if err := ValidatePort(port, reserved...); err == nil {
			t.Fatalf("ValidatePort(%d) accepted an invalid port", port)
		}
	}
}

func TestValidateBind(t *testing.T) {
	for _, bind := range []string{"127.0.0.1", "0.0.0.0", "::1", "localhost"} {
		if err := ValidateBind(bind); err != nil {
			t.Fatalf("ValidateBind(%q) rejected a usable bind: %v", bind, err)
		}
	}

	for _, bind := range []string{"", "   ", "router.internal", "127.0.0.1:6060", "*"} {
		if err := ValidateBind(bind); err == nil {
			t.Fatalf("ValidateBind(%q) accepted an invalid bind", bind)
		}
	}
}

func startForTest(t *testing.T, cfg config.ProfilingConfig) *Server {
	t.Helper()
	server, err := Start(cfg)
	if err != nil {
		t.Fatalf("Start returned an error: %v", err)
	}
	if server == nil {
		t.Fatal("Start returned no server while profiling is enabled")
	}
	t.Cleanup(func() {
		_ = server.Close()
	})
	return server
}

func get(t *testing.T, addr, path string) (int, []byte) {
	t.Helper()
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Get(fmt.Sprintf("http://%s%s", addr, path))
	if err != nil {
		t.Fatalf("GET %s failed: %v", path, err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("failed to read %s response: %v", path, err)
	}
	return resp.StatusCode, body
}
