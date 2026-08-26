package agentruntime

import (
	"fmt"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestWorkerFailureDiagnosticRetainsOnlyClosedProtocolIdentity(t *testing.T) {
	protocolFailure := llmprotocol.NewError(
		llmprotocol.ErrorUpstreamUnavailable,
		"invalid_upstream_json",
		"must not enter the server log",
		fmt.Errorf("provider payload must not enter the server log"),
	)
	diagnostic := safeWorkerFailureDiagnostic(
		wrapModelStepStageFailure(modelStepStageInferenceStream, fmt.Errorf("decode stream: %w", protocolFailure)),
	)
	if diagnostic.class != "protocol" || diagnostic.modelStepStage != modelStepStageInferenceStream ||
		diagnostic.protocolCategory != llmprotocol.ErrorUpstreamUnavailable ||
		diagnostic.protocolCode != "invalid_upstream_json" {
		t.Fatalf("diagnostic = %#v", diagnostic)
	}
}

func TestWorkerFailureDiagnosticIdentifiesCollectorFinishConflict(t *testing.T) {
	diagnostic := safeWorkerFailureDiagnostic(wrapModelStepStageFailure(
		modelStepStageFinish,
		fmt.Errorf("%w: provider-controlled detail", agentmanagement.ErrConflict),
	))
	if diagnostic.class != "state_conflict" || diagnostic.modelStepStage != modelStepStageFinish ||
		diagnostic.protocolCategory != "" || diagnostic.protocolCode != "" {
		t.Fatalf("diagnostic = %#v", diagnostic)
	}
}

func TestWorkerFailureDiagnosticDropsUntrustedProtocolIdentity(t *testing.T) {
	diagnostic := safeWorkerFailureDiagnostic(llmprotocol.NewError(
		llmprotocol.ErrorCategory("future\ncategory"), "provider code", "secret", nil,
	))
	if diagnostic.class != "protocol" || diagnostic.protocolCategory != "" || diagnostic.protocolCode != "" {
		t.Fatalf("diagnostic = %#v", diagnostic)
	}
}
