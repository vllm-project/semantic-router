package agentruntime

import (
	"context"
	"errors"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type modelStepFailureStage string

const (
	modelStepStageInferenceStream   modelStepFailureStage = "inference_stream"
	modelStepStageRouterObservation modelStepFailureStage = "router_observation"
	modelStepStageFinish            modelStepFailureStage = "finish"
	modelStepStageCommit            modelStepFailureStage = "commit"
)

type modelStepStageFailure struct {
	stage modelStepFailureStage
	err   error
}

func (failure *modelStepStageFailure) Error() string {
	return "agent model step " + string(failure.stage) + " failed"
}

func (failure *modelStepStageFailure) Unwrap() error { return failure.err }

func wrapModelStepStageFailure(stage modelStepFailureStage, err error) error {
	if err == nil {
		return nil
	}
	return &modelStepStageFailure{stage: stage, err: err}
}

type workerFailureDiagnostic struct {
	class            string
	upstreamStatus   int
	modelStepStage   modelStepFailureStage
	protocolCategory llmprotocol.ErrorCategory
	protocolCode     string
}

func safeWorkerFailureDiagnostic(err error) workerFailureDiagnostic {
	diagnostic := workerFailureDiagnostic{class: "internal"}
	var stageFailure *modelStepStageFailure
	if errors.As(err, &stageFailure) {
		diagnostic.modelStepStage = stageFailure.stage
	}
	var httpFailure *publicInferenceHTTPError
	var protocolFailure *llmprotocol.ProtocolError
	switch {
	case errors.As(err, &httpFailure):
		diagnostic.class = "public_inference_http"
		diagnostic.upstreamStatus = httpFailure.statusCode
	case errors.As(err, &protocolFailure):
		diagnostic.class = "protocol"
		diagnostic.protocolCategory = safeProtocolCategory(protocolFailure.Category)
		diagnostic.protocolCode = safeProtocolCode(protocolFailure.Code)
	case errors.Is(err, agentmanagement.ErrDenied):
		diagnostic.class = "authorization"
	case errors.Is(err, agentmanagement.ErrToolUnavailable):
		diagnostic.class = "tool_unavailable"
	case errors.Is(err, agentmanagement.ErrConflict):
		diagnostic.class = "state_conflict"
	case errors.Is(err, context.DeadlineExceeded):
		diagnostic.class = "deadline"
	}
	return diagnostic
}

func safeProtocolCategory(category llmprotocol.ErrorCategory) llmprotocol.ErrorCategory {
	switch category {
	case llmprotocol.ErrorInvalidRequest, llmprotocol.ErrorAuthentication,
		llmprotocol.ErrorPermission, llmprotocol.ErrorNotFound, llmprotocol.ErrorConflict,
		llmprotocol.ErrorUnsupportedFeature, llmprotocol.ErrorRateLimited,
		llmprotocol.ErrorUpstreamUnavailable, llmprotocol.ErrorUpstreamTimeout,
		llmprotocol.ErrorInternal:
		return category
	default:
		return ""
	}
}

func safeProtocolCode(code string) string {
	if code == "" || len(code) > 96 {
		return ""
	}
	for _, character := range code {
		if character != '_' && (character < 'a' || character > 'z') &&
			(character < '0' || character > '9') {
			return ""
		}
	}
	return code
}
