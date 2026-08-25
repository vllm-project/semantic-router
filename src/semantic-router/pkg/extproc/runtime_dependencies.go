package extproc

import (
	"errors"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/runtimecapabilities"
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
	capabilities, err := runtimecapabilities.Derive(cfg)
	if err != nil {
		return err
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
	if dependencies.DispatchCapabilities.Metered() != capabilities.NativeAccess {
		return errors.New("backend dispatch metering does not match the native access capability")
	}
	if capabilities.NativeAccess {
		if dependencies.InferenceAccess == nil {
			return errors.New("native access runtime is required")
		}
		if dependencies.OutcomeFeedback == nil {
			return errors.New("durable outcome feedback runtime is required")
		}
		if dependencies.OutcomeProjection == nil {
			return errors.New("durable outcome learning projection runtime is required")
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
