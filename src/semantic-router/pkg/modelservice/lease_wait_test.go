package modelservice

import (
	"context"
	"errors"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestWaitManagedHoldsUntilTheManagedModelLoads(t *testing.T) {
	hold := filepath.Join(t.TempDir(), "loaded")
	t.Setenv(fakeRuntimeHoldEnv, hold)
	lease := managedFakeLease(t)
	waited := make(chan error, 1)
	go func() { waited <- lease.WaitManaged(context.Background()) }()

	select {
	case err := <-waited:
		t.Fatalf("WaitManaged returned while the model loads: %v %+v", err, lease.Statuses())
	case <-time.After(500 * time.Millisecond):
	}
	statuses := lease.Statuses()
	if len(statuses) != 1 || statuses[0].Ready || !statuses[0].Managed || statuses[0].Artifact != "vllm-sr/Decision-2.0-Kai-0.6B" {
		t.Fatalf("a loading managed deployment reports its artifact and is not ready: %+v", statuses)
	}
	if err := os.WriteFile(hold, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-waited:
		if err != nil {
			t.Fatalf("WaitManaged after the model loaded: %v", err)
		}
	case <-time.After(20 * time.Second):
		t.Fatalf("WaitManaged did not return once the model loaded: %+v", lease.Statuses())
	}
	if _, err := lease.Decide(context.Background(), "kai", sampleRequest("text")); err != nil {
		t.Fatalf("a deployment WaitManaged returned for answers at once: %v", err)
	}
}

func TestWaitManagedDoesNotWaitForAnAttachedEndpoint(t *testing.T) {
	runtime := fakeModels([]modelEntry{{Name: "remote", Model: "vllm-sr/Decision-2.0-Kai-0.6B"}})
	runtime.SetReady("remote", false)
	server := httptest.NewServer(runtime.Handler())
	defer server.Close()
	binary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv(fakeRuntimeEnv, "1")
	t.Setenv(RuntimeCommandEnv, binary)
	t.Setenv(RuntimeDirEnv, filepath.Join(t.TempDir(), "run"))
	manager := NewManager()
	defer func() { _ = manager.Shutdown(context.Background()) }()
	lease, err := manager.Acquire(runtimeConfig(map[string]config.ModelDeployment{
		"kai":    {Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/Decision-2.0-Kai-0.6B", Device: "cpu"},
		"remote": {Provider: config.ModelRuntimeProvider, Endpoint: server.URL},
	}))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	if err := lease.WaitManaged(ctx); err != nil {
		t.Fatalf("an attached endpoint that is not ready must not hold the wait: %v", err)
	}
	for _, status := range lease.Statuses() {
		switch status.Name {
		case "kai":
			if !status.Ready {
				t.Fatalf("the managed deployment is ready once WaitManaged returns: %+v", status)
			}
		case "remote":
			if status.Ready || status.Artifact != "" {
				t.Fatalf("the attached deployment is still loading and names no artifact: %+v", status)
			}
		}
	}
}

func TestWaitManagedFailsAtOnceForAModelThatCannotLoad(t *testing.T) {
	t.Setenv(fakeRuntimeFailAlwaysEnv, "1")
	lease := managedFakeLease(t)
	started := time.Now()
	err := lease.WaitManaged(context.Background())
	if !errors.Is(err, ErrUnavailable) || !strings.Contains(err.Error(), `model_runtime deployment "kai"`) ||
		!strings.Contains(err.Error(), "failed to load 3 times") {
		t.Fatalf("a model that never loads must fail the wait with its deployment and reason, got %v", err)
	}
	if time.Since(started) > 20*time.Second {
		t.Fatal("the failure must not wait for the ready timeout")
	}
}

func TestWaitManagedIsBoundedByTheReadyTimeout(t *testing.T) {
	t.Setenv(fakeRuntimeHoldEnv, filepath.Join(t.TempDir(), "never"))
	t.Setenv(ReadyTimeoutEnv, "300ms")
	lease := managedFakeLease(t)
	err := lease.WaitManaged(context.Background())
	if !errors.Is(err, context.DeadlineExceeded) || !errors.Is(err, ErrUnavailable) ||
		!strings.Contains(err.Error(), `model_runtime deployment "kai" did not become ready within 300ms (`+ReadyTimeoutEnv+`)`) {
		t.Fatalf("a wait without a deadline ends at the ready timeout and names it, got %v", err)
	}
}

func TestWaitManagedWithoutManagedDeployments(t *testing.T) {
	if err := (*Lease)(nil).WaitManaged(context.Background()); err != nil {
		t.Fatal(err)
	}
	manager := NewManager()
	defer func() { _ = manager.Shutdown(context.Background()) }()
	lease, err := manager.AcquireDeployments(nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := lease.WaitManaged(context.Background()); err != nil {
		t.Fatal(err)
	}
}
