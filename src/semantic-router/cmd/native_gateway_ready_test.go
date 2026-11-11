//go:build !windows

package main

import (
	"context"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

type startupStateLog struct {
	mu     sync.Mutex
	states []startupstatus.State
}

func (w *startupStateLog) Write(state startupstatus.State) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.states = append(w.states, state)
	return nil
}

func (w *startupStateLog) ready() bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	for _, state := range w.states {
		if state.Ready {
			return true
		}
	}
	return false
}

// startReadyTestGateway serves the standalone listener of a one-model config
// whose document ends with global, and returns the server, its config, the
// startup states written and a probe of the listener's /ready.
func startReadyTestGateway(t *testing.T, global string) (*extproc.Server, *config.RouterConfig, *startupStateLog, func() int) {
	t.Helper()
	t.Setenv(configsnapshot.HistoryDirEnv, t.TempDir())
	previous := modelservice.DefaultManager()
	modelservice.SetDefault(nil)
	t.Cleanup(func() { modelservice.SetDefault(previous) })
	reservation, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := strconv.Itoa(reservation.Addr().(*net.TCPAddr).Port)
	_ = reservation.Close()
	document := strings.Replace(backendDocument("http://127.0.0.1:1", ""), "port: 8899", "port: "+port, 1) + global
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err = os.WriteFile(path, []byte(document), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := config.Parse(path)
	if err != nil {
		t.Fatal(err)
	}
	server, err := extproc.NewServer(path, 0, false, "", routerruntime.NewRegistry(cfg),
		extproc.WithConfigParts(upstreamPart(config.GatewayStandalone)))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	writer := &startupStateLog{}
	done := make(chan error, 1)
	go func() { done <- startNativeGateway(ctx, server, writer, "127.0.0.1") }()
	t.Cleanup(func() { cancel(); <-done })
	if err = server.WaitForServing(ctx); err != nil {
		t.Fatal(err)
	}
	return server, cfg, writer, func() int {
		t.Helper()
		resp, getErr := http.Get("http://127.0.0.1:" + port + "/ready")
		if getErr != nil {
			t.Fatal(getErr)
		}
		_ = resp.Body.Close()
		return resp.StatusCode
	}
}

// TestNativeListenerReadyWaitsForKubernetesActivation starts the standalone
// listeners of a Router whose configuration comes from Kubernetes. As the
// gRPC health service does in ext_proc mode, the listener's /ready must answer
// 503 until the first Kubernetes configuration is active, and 200 after.
func TestNativeListenerReadyWaitsForKubernetesActivation(t *testing.T) {
	server, cfg, writer, readyStatus := startReadyTestGateway(t, "global:\n  router:\n    config_source: kubernetes\n")
	for deadline := time.Now().Add(300 * time.Millisecond); time.Now().Before(deadline); time.Sleep(20 * time.Millisecond) {
		if status := readyStatus(); status != http.StatusServiceUnavailable {
			t.Fatalf("listener /ready = %d before any Kubernetes configuration is active (management ready written: %v), want 503",
				status, writer.ready())
		}
	}
	if err := server.ActivateKubernetesConfig(context.Background(), cfg); err != nil {
		t.Fatal(err)
	}
	if status := readyStatus(); status != http.StatusOK {
		t.Fatalf("listener /ready = %d after the Kubernetes configuration is active, want 200", status)
	}
}

// TestNativeListenerReadyTurnsUnavailableWhenShutdownBegins checks that the
// listener's /ready answers 503 once the Router begins to shut down, while the
// listener still drains, as the gRPC health service turns NOT_SERVING then.
func TestNativeListenerReadyTurnsUnavailableWhenShutdownBegins(t *testing.T) {
	server, _, _, readyStatus := startReadyTestGateway(t, "")
	deadline := time.Now().Add(5 * time.Second)
	for readyStatus() != http.StatusOK {
		if time.Now().After(deadline) {
			t.Fatal("listener /ready never answered 200")
		}
		time.Sleep(20 * time.Millisecond)
	}
	if err := server.ShutdownServing(context.Background()); err != nil {
		t.Fatal(err)
	}
	if status := readyStatus(); status != http.StatusServiceUnavailable {
		t.Fatalf("listener /ready = %d after shutdown began, want 503", status)
	}
}
