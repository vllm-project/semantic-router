package extproc

import (
	"errors"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// RuntimeDependencies are process-owned resources borrowed by every immutable
// router generation. Generations must never close these resources.
type RuntimeDependencies struct {
	InferenceAccess      InferenceAccessRuntime
	DispatchCapabilities DispatchCapabilityRuntime
	OutcomeFeedback      OutcomeFeedbackRuntime
	OutcomeProjection    OutcomeLearningProjectionRuntime
	ResponseTerminals    backendinvoker.ResponseTerminalReader
	ProtocolCodecs       *protocolcodec.Registry
}

func (dependencies RuntimeDependencies) validate(cfg *config.RouterConfig) error {
	if cfg == nil {
		return errors.New("router configuration is required")
	}
	if dependencies.DispatchCapabilities == nil {
		return errors.New("backend dispatch capability runtime is required")
	}
	if dependencies.ResponseTerminals == nil {
		return errors.New("semantic response terminal runtime is required")
	}
	if dependencies.ProtocolCodecs == nil {
		return errors.New("protocol codec registry is required")
	}
	if dependencies.DispatchCapabilities.Metered() != cfg.Access.Enabled {
		return errors.New("backend dispatch authority mode does not match access configuration")
	}
	if cfg.Access.Enabled {
		if cfg.ControlPlane.Mode != config.ControlPlaneModeManaged {
			return errors.New("inference access requires managed control-plane mode")
		}
		if dependencies.InferenceAccess == nil {
			return errors.New("managed inference access runtime is required")
		}
		if dependencies.OutcomeFeedback == nil {
			return errors.New("managed outcome feedback runtime is required")
		}
		if dependencies.OutcomeProjection == nil {
			return errors.New("managed outcome learning projection runtime is required")
		}
		return nil
	}
	if dependencies.InferenceAccess != nil {
		return errors.New("inference access runtime was injected while access is disabled")
	}
	if dependencies.OutcomeFeedback != nil {
		return errors.New("outcome feedback runtime was injected while access is disabled")
	}
	if dependencies.OutcomeProjection != nil {
		return errors.New("outcome learning projection runtime was injected while access is disabled")
	}
	return nil
}
