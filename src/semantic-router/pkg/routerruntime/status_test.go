package routerruntime

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

func requireCondition(t *testing.T, report StatusReport, kind ConditionType, status ConditionStatus, reason StatusReason) {
	t.Helper()
	for _, condition := range report.Conditions {
		if condition.Type == kind {
			if condition.Status != status || condition.Reason != reason {
				t.Fatalf("condition=%+v, want=(%s,%s)", condition, status, reason)
			}
			return
		}
	}
	t.Fatalf("missing %s condition: %+v", kind, report)
}

func TestStatusTracksLocalStartupAndConfigConvergence(t *testing.T) {
	path := filepath.Join(t.TempDir(), "router.yaml")
	cfg := &config.RouterConfig{DocumentHash: strings.Repeat("a", 64)}
	registry := NewRegistry(cfg)
	writer := registry.StartupStatusWriter(startupstatus.NewFileWriter(path))
	initial := registry.Status()
	if initial.InstanceID == "" || initial.Scope != "replica" || initial.SchemaVersion != StatusSchemaVersion {
		t.Fatalf("invalid status identity: %+v", initial)
	}
	requireCondition(t, initial, ConditionStartupComplete, ConditionUnknown, ReasonStartupUnobserved)
	requireCondition(t, initial, ConditionReady, ConditionFalse, ReasonNoActiveConfig)
	requireCondition(t, initial, ConditionConverged, ConditionUnknown, ReasonConfigIdentityUnobserved)

	if err := writer.Write(startupstatus.State{Phase: "starting"}); err != nil {
		t.Fatal(err)
	}
	requireCondition(t, registry.Status(), ConditionStartupComplete, ConditionFalse, ReasonStartupIncomplete)

	registry.PublishRouterRuntimeSnapshot(RouterRuntimeSnapshot{
		Config: cfg, ClassificationService: services.NewPlaceholderClassificationService(),
	})
	if err := writer.Write(startupstatus.State{Phase: "ready", Ready: true}); err != nil {
		t.Fatal(err)
	}
	ready := registry.Status()
	requireCondition(t, ready, ConditionStartupComplete, ConditionTrue, ReasonStartupComplete)
	requireCondition(t, ready, ConditionActiveConfig, ConditionTrue, ReasonActiveConfigAvailable)
	requireCondition(t, ready, ConditionConverged, ConditionTrue, ReasonConfigHashesMatch)
	requireCondition(t, ready, ConditionReady, ConditionUnknown, ReasonRequiredDependenciesUnobserved)

	candidateHash := strings.Repeat("b", 64)
	attempt := registry.BeginConfigActivation(candidateHash, "file")
	requireCondition(t, registry.Status(), ConditionConverged, ConditionFalse, ReasonConfigActivationPending)
	registry.SetConfigActivationStage(attempt, "dependencies")
	registry.FinishConfigActivation(attempt, "failed", errors.New("credential=do-not-serialize"))
	failed := registry.Status()
	requireCondition(t, failed, ConditionConverged, ConditionFalse, ReasonConfigActivationFailed)
	requireCondition(t, failed, ConditionActiveConfig, ConditionTrue, ReasonActiveConfigAvailable)
	requireCondition(t, failed, ConditionReady, ConditionUnknown, ReasonRequiredDependenciesUnobserved)
	if failed.Config.ActiveHash != cfg.DocumentHash || failed.Config.ObservedHash != candidateHash || failed.Config.ObservedAttempt != attempt {
		t.Fatalf("lost configuration identity: %+v", failed.Config)
	}
	if registry.CurrentConfig() != cfg {
		t.Fatal("reading status replaced the active configuration")
	}
	payload, err := json.Marshal(failed)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(payload), "do-not-serialize") {
		t.Fatalf("status contains raw activation failure details: %s", payload)
	}

	recovery := registry.BeginConfigActivation(candidateHash, "file")
	registry.PublishRouterRuntimeSnapshot(RouterRuntimeSnapshot{
		Config: &config.RouterConfig{DocumentHash: candidateHash}, ClassificationService: services.NewPlaceholderClassificationService(),
	})
	registry.FinishConfigActivation(recovery, "active", nil)
	recovered := registry.Status()
	requireCondition(t, recovered, ConditionConverged, ConditionTrue, ReasonConfigHashesMatch)
	if recovered.InstanceID != initial.InstanceID || recovered.ObservedAt.IsZero() {
		t.Fatalf("invalid recovered identity: %+v", recovered)
	}
}

func TestStatusIsReplicaLocalAndReturnsIndependentSnapshots(t *testing.T) {
	path := filepath.Join(t.TempDir(), "router.yaml")
	first, second := NewRegistry(nil), NewRegistry(nil)
	state := startupstatus.State{Phase: "ready", Ready: true, Message: "private startup detail"}
	if err := first.StartupStatusWriter(startupstatus.NewFileWriter(path)).Write(state); err != nil {
		t.Fatal(err)
	}
	if err := second.StartupStatusWriter(startupstatus.NewFileWriter(path)).Write(startupstatus.State{Phase: "starting"}); err != nil {
		t.Fatal(err)
	}
	firstReport, secondReport := first.Status(), second.Status()
	requireCondition(t, firstReport, ConditionStartupComplete, ConditionTrue, ReasonStartupComplete)
	requireCondition(t, secondReport, ConditionStartupComplete, ConditionFalse, ReasonStartupIncomplete)
	if firstReport.InstanceID == secondReport.InstanceID {
		t.Fatal("registries share an instance identity")
	}
	firstReport.Startup.Ready = false
	requireCondition(t, first.Status(), ConditionStartupComplete, ConditionTrue, ReasonStartupComplete)
	payload, err := json.Marshal(first.Status())
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(payload), state.Message) {
		t.Fatalf("status contains startup diagnostics: %s", payload)
	}
}

func TestStatusRetainsLocalFactsWhenPersistenceFails(t *testing.T) {
	path := filepath.Join(t.TempDir(), "router.yaml")
	if err := os.Mkdir(startupstatus.StatusPathFromConfigPath(path), 0o755); err != nil {
		t.Fatal(err)
	}
	registry := NewRegistry(nil)
	writer := registry.StartupStatusWriter(startupstatus.NewFileWriter(path))
	if err := writer.Write(startupstatus.State{Phase: "error"}); err == nil {
		t.Fatal("expected the status file replacement to fail")
	}
	requireCondition(t, registry.Status(), ConditionStartupComplete, ConditionFalse, ReasonStartupFailed)
}

func TestStatusDoesNotReportRetiredRuntimeAsActive(t *testing.T) {
	registry := NewRegistry(nil)
	registry.PublishRouterRuntimeSnapshot(RouterRuntimeSnapshot{
		Config:                &config.RouterConfig{DocumentHash: strings.Repeat("a", 64)},
		ClassificationService: services.NewPlaceholderClassificationService(),
		AcquireClassification: func() (func(), bool) { return nil, false },
	})
	report := registry.Status()
	requireCondition(t, report, ConditionActiveConfig, ConditionFalse, ReasonNoActiveConfig)
	requireCondition(t, report, ConditionReady, ConditionFalse, ReasonNoActiveConfig)
	if report.Config.ActiveHash != "" {
		t.Fatalf("retired configuration reported active: %+v", report.Config)
	}
}
