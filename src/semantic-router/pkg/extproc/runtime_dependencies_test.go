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
	managed := &config.RouterConfig{
		ControlPlane: config.ControlPlaneConfig{Mode: config.ControlPlaneModeManaged},
		Access:       config.AccessServiceConfig{Enabled: true},
	}
	if err := (RuntimeDependencies{}).validate(managed); err == nil || !strings.Contains(err.Error(), "required") {
		t.Fatalf("managed missing runtime error = %v", err)
	}
	if err := (RuntimeDependencies{
		InferenceAccess:      &fakeInferenceAccess{},
		DispatchCapabilities: dispatchCapabilityRuntimeStub{metered: true},
		OutcomeFeedback:      &outcomeRuntimeStub{},
		OutcomeProjection:    outcomeProjectionRuntimeStub{},
		ResponseTerminals:    backendinvoker.NewLocalResponseTerminalStore(),
		ProtocolCodecs:       protocolcodec.NewBuiltinRegistry(),
	}).validate(managed); err != nil {
		t.Fatalf("managed injected runtime error = %v", err)
	}
	if err := (RuntimeDependencies{
		InferenceAccess:      &fakeInferenceAccess{},
		DispatchCapabilities: dispatchCapabilityRuntimeStub{metered: true},
		OutcomeProjection:    outcomeProjectionRuntimeStub{},
		ResponseTerminals:    backendinvoker.NewLocalResponseTerminalStore(),
		ProtocolCodecs:       protocolcodec.NewBuiltinRegistry(),
	}).validate(managed); err == nil || !strings.Contains(err.Error(), "outcome feedback") {
		t.Fatalf("managed missing outcome feedback error = %v", err)
	}
	if err := (RuntimeDependencies{
		InferenceAccess:      &fakeInferenceAccess{},
		DispatchCapabilities: dispatchCapabilityRuntimeStub{metered: true},
		OutcomeFeedback:      &outcomeRuntimeStub{},
		ResponseTerminals:    backendinvoker.NewLocalResponseTerminalStore(),
		ProtocolCodecs:       protocolcodec.NewBuiltinRegistry(),
	}).validate(managed); err == nil || !strings.Contains(err.Error(), "outcome learning projection") {
		t.Fatalf("managed missing outcome projection error = %v", err)
	}

	standalone := &config.RouterConfig{
		ControlPlane: config.ControlPlaneConfig{Mode: config.ControlPlaneModeStandalone},
	}
	if err := (RuntimeDependencies{}).validate(standalone); err == nil ||
		!strings.Contains(err.Error(), "backend dispatch") {
		t.Fatalf("standalone missing dispatch runtime error = %v", err)
	}
	standaloneDependencies := RuntimeDependencies{
		DispatchCapabilities: dispatchCapabilityRuntimeStub{metered: false},
		ResponseTerminals:    backendinvoker.NewLocalResponseTerminalStore(),
		ProtocolCodecs:       protocolcodec.NewBuiltinRegistry(),
	}
	if err := standaloneDependencies.validate(standalone); err != nil {
		t.Fatalf("standalone dispatch dependencies error = %v", err)
	}
	standaloneDependencies.InferenceAccess = &fakeInferenceAccess{}
	if err := standaloneDependencies.validate(standalone); err == nil ||
		!strings.Contains(err.Error(), "disabled") {
		t.Fatalf("standalone injected runtime error = %v", err)
	}
	if err := (RuntimeDependencies{
		DispatchCapabilities: dispatchCapabilityRuntimeStub{metered: true},
		ResponseTerminals:    backendinvoker.NewLocalResponseTerminalStore(),
		ProtocolCodecs:       protocolcodec.NewBuiltinRegistry(),
	}).validate(standalone); err == nil ||
		!strings.Contains(err.Error(), "mode") {
		t.Fatalf("standalone metered dispatch runtime error = %v", err)
	}
}
