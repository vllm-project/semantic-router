//go:build !windows

package apiserver

import (
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

// The runtime publishes the new document before the lifecycle reports its
// attempt finished. A write must not report "active" in that window, or a
// client reading the activation or audit right after it misses the outcome
// (#5031).
func TestRuntimeActivationWaitsForTheMatchingAttemptToFinish(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	document := []byte("version: v0.3\nrouting: {}\n")
	if err := os.WriteFile(path, document, 0o600); err != nil {
		t.Fatal(err)
	}
	hash, err := configFileHash(path)
	if err != nil {
		t.Fatal(err)
	}
	registry := routerruntime.NewRegistry(&config.RouterConfig{DocumentHash: hash})
	server := &ClassificationAPIServer{configPath: path, runtimeRegistry: registry}
	attempt := registry.BeginConfigActivation(hash, "api")

	var finished atomic.Bool
	go func() {
		time.Sleep(150 * time.Millisecond)
		finished.Store(true)
		registry.FinishConfigActivation(attempt, "active", nil)
	}()
	if _, status := server.waitForRuntimeConfigActivation(path, document, attempt-1); status != "active" {
		t.Fatalf("status = %s, want active", status)
	}
	if !finished.Load() {
		t.Fatal("reported active while the matching attempt was still preparing")
	}
}
