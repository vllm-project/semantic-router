package routerruntime

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

type rejectingRuntime struct{ err error }

func (r rejectingRuntime) Validate(context.Context, *configsnapshot.Candidate) error { return r.err }

func (r rejectingRuntime) Warm(context.Context, *configsnapshot.Candidate) (configsnapshot.Warmed, error) {
	return nil, errors.New("unreachable")
}

func TestRegistryPublishesTheLifecycleAttempts(t *testing.T) {
	startup := &config.RouterConfig{DocumentHash: strings.Repeat("a", 64)}
	registry := NewRegistry(startup)
	cause := errors.New("routing_preview.max_concurrency changed")
	manager := configsnapshot.NewManager(configsnapshot.Options{
		Runtime:  rejectingRuntime{err: configsnapshot.Reject(configsnapshot.StageValidate, configsnapshot.CodeRestartRequired, cause)},
		Reporter: registry,
	})
	snapshot, err := manager.Install(context.Background(), configsnapshot.Update{Origin: configsnapshot.Origin{Source: configsnapshot.SourceStartup}, Config: startup})
	if err != nil {
		t.Fatal(err)
	}
	registry.PublishRouterRuntimeSnapshot(RouterRuntimeSnapshot{
		Config: startup, ConfigSnapshot: snapshot, ClassificationService: services.NewPlaceholderClassificationService(),
	})
	if activation := registry.ConfigActivation(); activation.Status != "active" || activation.Version != 1 ||
		activation.Source != "startup" || activation.FinishedAt == nil {
		t.Fatalf("startup activation = %+v", activation)
	}
	if status := registry.Status(); status.Config.ActiveVersion != 1 {
		t.Fatalf("status = %+v", status.Config)
	}

	candidate := &config.RouterConfig{DocumentHash: strings.Repeat("b", 64)}
	if _, err := manager.Apply(context.Background(), configsnapshot.Update{
		Origin: configsnapshot.Origin{Source: configsnapshot.SourceFile}, Config: candidate,
	}); err == nil {
		t.Fatal("Apply() accepted a rejected candidate")
	}
	activation := registry.ConfigActivation()
	if activation.Status != "failed" || activation.Attempt != 2 || activation.FailureDetail != cause.Error() ||
		len(activation.Reasons) != 1 || activation.Reasons[0].Code != configsnapshot.CodeRestartRequired {
		t.Fatalf("rejected activation = %+v", activation)
	}
	rejection, ok := registry.LastConfigRejection()
	if !ok || rejection.DocumentHash != candidate.DocumentHash {
		t.Fatalf("last rejection = %+v, %v", rejection, ok)
	}
	if registry.ConfigSnapshot() != snapshot {
		t.Fatal("a rejection replaced the published snapshot")
	}

	stale := configsnapshot.Attempt{ID: 1, Status: configsnapshot.AttemptActive, StartedAt: time.Now()}
	registry.ReportConfigAttempt(stale, nil)
	if registry.ConfigActivation().Attempt != 2 {
		t.Fatal("an older attempt replaced the published one")
	}
}

func TestConfigAttemptListenersHearFinishedAttemptsOnly(t *testing.T) {
	registry := NewRegistry(nil)
	var heard []configsnapshot.Attempt
	registry.OnConfigAttempt(func(attempt configsnapshot.Attempt) { heard = append(heard, attempt) })
	manager := configsnapshot.NewManager(configsnapshot.Options{Reporter: registry})
	if _, err := manager.Install(context.Background(), configsnapshot.Update{
		Origin: configsnapshot.Origin{Source: configsnapshot.SourceStartup}, Config: &config.RouterConfig{DocumentHash: "a"},
	}); err != nil {
		t.Fatal(err)
	}
	if len(heard) != 1 || heard[0].Status != configsnapshot.AttemptActive || heard[0].Version != 1 {
		t.Fatalf("listeners heard %+v, want the one finished startup attempt", heard)
	}
}
