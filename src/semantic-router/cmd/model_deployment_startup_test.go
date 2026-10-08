package main

import (
	"context"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

func TestModelDeploymentProgressNamesTheDeploymentsNotReady(t *testing.T) {
	state := modelDeploymentProgress([]startupstatus.ModelDeploymentStatus{
		{Name: "decider", Artifact: "vllm-sr/Decision-2.0-Kai-0.6B", Process: "rocm:0", State: "loading"},
		{Name: "guard", Artifact: "vllm-sr/Vela-2.0-0.3B", Process: "cpu", State: "ready", Ready: true},
		{Name: "pii", Artifact: "/models/pii", Process: "cpu", State: "failed", Reason: "out of memory"},
	})
	if state.Phase != startupstatus.PhaseLoadingModelDeployments || state.Ready || state.TotalModels != 3 || state.ReadyModels != 1 ||
		!slices.Equal(state.PendingModels, []string{"decider", "pii"}) || len(state.ModelDeployments) != 3 {
		t.Fatalf("progress while two deployments are not ready: %+v", state)
	}
	for _, want := range []string{"1 of 3 ready", "decider (vllm-sr/Decision-2.0-Kai-0.6B) loading", "pii (/models/pii) failed: out of memory"} {
		if !strings.Contains(state.Message, want) {
			t.Fatalf("message %q does not say %q", state.Message, want)
		}
	}
	if strings.Contains(state.Message, "guard") {
		t.Fatalf("message %q names a ready deployment", state.Message)
	}

	done := modelDeploymentProgress([]startupstatus.ModelDeploymentStatus{{Name: "decider", State: "ready", Ready: true}})
	if done.Phase != "initializing_models" || done.Ready || done.TotalModels != 1 || done.ReadyModels != 1 || len(done.PendingModels) != 0 ||
		!strings.Contains(done.Message, "are ready (1)") {
		t.Fatalf("progress once every deployment is ready: %+v", done)
	}
}

func TestManagedDeploymentStatusesKeepOnlyRouterManagedDeployments(t *testing.T) {
	got := managedDeploymentStatuses([]modelservice.DeploymentStatus{
		{Name: "attached", Managed: false, State: "loading"},
		{Name: "decider", Managed: true, Process: "decisions", Artifact: "vllm-sr/Decision-2.0-Kai-0.6B", State: "ready", Ready: true},
	})
	want := []startupstatus.ModelDeploymentStatus{{Name: "decider", Artifact: "vllm-sr/Decision-2.0-Kai-0.6B", Process: "decisions", State: "ready", Ready: true}}
	if !slices.Equal(got, want) {
		t.Fatalf("managed statuses = %+v, want %+v", got, want)
	}
}

// stateLog records every startup state written.
type stateLog struct {
	mu     sync.Mutex
	states []startupstatus.State
}

func (l *stateLog) Write(state startupstatus.State) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.states = append(l.states, state)
	return nil
}

func (l *stateLog) snapshot() []startupstatus.State {
	l.mu.Lock()
	defer l.mu.Unlock()
	return slices.Clone(l.states)
}

func managedTestConfig(t *testing.T) *config.RouterConfig {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(readinessDocument(readinessPorts{api: 18080, listener: 18899, extproc: 15051}, "http://127.0.0.1:9", "decider")))
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

func TestReportModelDeploymentProgressWritesEachChangeUntilStopped(t *testing.T) {
	hold := useFakeRuntime(t)
	manager := modelservice.NewManager()
	t.Cleanup(func() { _ = manager.Shutdown(context.Background()) })
	if err := manager.Reconcile(managedTestConfig(t)); err != nil {
		t.Fatal(err)
	}
	log := &stateLog{}
	stop := reportModelDeploymentProgress(log, manager)
	waitForState(t, log, func(state startupstatus.State) bool {
		return state.Phase == startupstatus.PhaseLoadingModelDeployments && slices.Equal(state.PendingModels, []string{"decider"})
	})
	if err := os.WriteFile(hold, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	waitForState(t, log, func(state startupstatus.State) bool {
		return state.Phase == "initializing_models" && state.ReadyModels == 1 && state.TotalModels == 1
	})
	stop()
	written := len(log.snapshot())
	time.Sleep(2 * modelDeploymentProgressInterval)
	if states := log.snapshot(); len(states) != written {
		t.Fatalf("progress written after stop: %+v", states[written:])
	}
	for _, state := range log.snapshot() {
		if state.Ready {
			t.Fatalf("progress never reports ready; markRouterReady does: %+v", state)
		}
	}
}

func TestReportModelDeploymentProgressWithoutManager(t *testing.T) {
	log := &stateLog{}
	reportModelDeploymentProgress(log, nil)()
	if states := log.snapshot(); len(states) != 0 {
		t.Fatalf("no manager, no progress: %+v", states)
	}
}

func TestMarkRouterReadyListsTheManagedDeployments(t *testing.T) {
	hold := useFakeRuntime(t)
	if err := os.WriteFile(hold, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	manager := modelservice.NewManager()
	t.Cleanup(func() { _ = manager.Shutdown(context.Background()) })
	previous := modelservice.DefaultManager()
	modelservice.SetDefault(manager)
	t.Cleanup(func() { modelservice.SetDefault(previous) })
	if err := manager.Reconcile(managedTestConfig(t)); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	if err := manager.Published().WaitManaged(ctx); err != nil {
		t.Fatal(err)
	}
	writer := &recordingStartupWriter{}
	markRouterReady(writer, nil)
	state := writer.state
	if state.Phase != "ready" || !state.Ready || state.TotalModels != 1 || state.ReadyModels != 1 || len(state.PendingModels) != 0 ||
		len(state.ModelDeployments) != 1 || state.ModelDeployments[0].Name != "decider" || !state.ModelDeployments[0].Ready {
		t.Fatalf("ready state lists the ready deployment: %+v", state)
	}
}

func waitForState(t *testing.T, log *stateLog, match func(startupstatus.State) bool) {
	t.Helper()
	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) {
		for _, state := range log.snapshot() {
			if match(state) {
				return
			}
		}
		time.Sleep(25 * time.Millisecond)
	}
	t.Fatalf("no matching startup state among %+v", log.snapshot())
}
