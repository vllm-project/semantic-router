package extproc

import (
	"context"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/outcomefeedback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

type outcomeProjectionRuntimeStub struct{}

func (outcomeProjectionRuntimeStub) Read(context.Context, string) (outcomefeedback.Projection, error) {
	return outcomefeedback.Projection{}, outcomefeedback.ErrNotFound
}

func TestRuntimeDependenciesEnforceStrictAccessMode(t *testing.T) {
	durableConfig := &config.RouterConfig{
		Access:             config.AccessServiceConfig{Enabled: true},
		AccessStore:        &config.AccessStoreConfig{},
		AccessRuntimeStore: &config.AccessRuntimeStoreConfig{},
	}
	if err := (RuntimeDependencies{}).validate(durableConfig); err == nil || !strings.Contains(err.Error(), "required") {
		t.Fatalf("durableConfig missing runtime error = %v", err)
	}
	if err := (RuntimeDependencies{
		InferenceAccess:      &fakeInferenceAccess{},
		DispatchCapabilities: dispatchCapabilityRuntimeStub{metered: true},
		OutcomeFeedback:      &outcomeRuntimeStub{},
		OutcomeProjection:    outcomeProjectionRuntimeStub{},
		ResponseTerminals:    backendinvoker.NewLocalResponseTerminalStore(),
		ProtocolCodecs:       protocolcodec.NewBuiltinRegistry(),
	}).validate(durableConfig); err != nil {
		t.Fatalf("durableConfig injected runtime error = %v", err)
	}
	if err := (RuntimeDependencies{
		InferenceAccess:      &fakeInferenceAccess{},
		DispatchCapabilities: dispatchCapabilityRuntimeStub{metered: true},
		OutcomeProjection:    outcomeProjectionRuntimeStub{},
		ResponseTerminals:    backendinvoker.NewLocalResponseTerminalStore(),
		ProtocolCodecs:       protocolcodec.NewBuiltinRegistry(),
	}).validate(durableConfig); err == nil || !strings.Contains(err.Error(), "outcome feedback") {
		t.Fatalf("durableConfig missing outcome feedback error = %v", err)
	}
	if err := (RuntimeDependencies{
		InferenceAccess:      &fakeInferenceAccess{},
		DispatchCapabilities: dispatchCapabilityRuntimeStub{metered: true},
		OutcomeFeedback:      &outcomeRuntimeStub{},
		ResponseTerminals:    backendinvoker.NewLocalResponseTerminalStore(),
		ProtocolCodecs:       protocolcodec.NewBuiltinRegistry(),
	}).validate(durableConfig); err == nil || !strings.Contains(err.Error(), "outcome learning projection") {
		t.Fatalf("durableConfig missing outcome projection error = %v", err)
	}

	fileConfig := &config.RouterConfig{}
	if err := (RuntimeDependencies{}).validate(fileConfig); err == nil ||
		!strings.Contains(err.Error(), "backend dispatch") {
		t.Fatalf("fileConfig missing dispatch runtime error = %v", err)
	}
	fileDependencies := RuntimeDependencies{
		DispatchCapabilities: dispatchCapabilityRuntimeStub{metered: false},
		ResponseTerminals:    backendinvoker.NewLocalResponseTerminalStore(),
		ProtocolCodecs:       protocolcodec.NewBuiltinRegistry(),
	}
	if err := fileDependencies.validate(fileConfig); err != nil {
		t.Fatalf("fileConfig dispatch dependencies error = %v", err)
	}
	fileDependencies.InferenceAccess = &fakeInferenceAccess{}
	if err := fileDependencies.validate(fileConfig); err == nil ||
		!strings.Contains(err.Error(), "disabled") {
		t.Fatalf("fileConfig injected runtime error = %v", err)
	}
	if err := (RuntimeDependencies{
		DispatchCapabilities: dispatchCapabilityRuntimeStub{metered: true},
		ResponseTerminals:    backendinvoker.NewLocalResponseTerminalStore(),
		ProtocolCodecs:       protocolcodec.NewBuiltinRegistry(),
	}).validate(fileConfig); err == nil ||
		!strings.Contains(err.Error(), "native access") {
		t.Fatalf("fileConfig metered dispatch runtime error = %v", err)
	}
}
