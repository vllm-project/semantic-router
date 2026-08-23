package extproc

import (
	"bytes"
	"context"
	"strings"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dispatchauthority"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingcontext"
)

type capturingDispatchCapabilityRuntime struct {
	issued              dispatchauthority.MeteredChainIssueRequest
	verifiedToken       string
	verificationRequest dispatchauthority.OutcomeVerificationRequest
	outcome             backendinvoker.DispatchOutcome
	verifyErr           error
}

func (*capturingDispatchCapabilityRuntime) Metered() bool { return true }

func (runtime *capturingDispatchCapabilityRuntime) IssueMeteredPrimary(request dispatchauthority.PrimaryIssueRequest) (string, error) {
	return "final-capability", nil
}

func (runtime *capturingDispatchCapabilityRuntime) IssueMeteredChain(request dispatchauthority.MeteredChainIssueRequest) (string, error) {
	runtime.issued = request
	return "final-capability", nil
}

func (*capturingDispatchCapabilityRuntime) IssueMeteredGrant(dispatchauthority.GrantIssueRequest) (string, error) {
	return "grant", nil
}

func (*capturingDispatchCapabilityRuntime) IssueRoutingOnlyPrimary(context.Context, dispatchauthority.RoutingOnlyIssueRequest) (string, error) {
	return "final-capability", nil
}

func (*capturingDispatchCapabilityRuntime) IssueRoutingOnlyChain(context.Context, dispatchauthority.RoutingOnlyChainIssueRequest) (string, error) {
	return "final-capability", nil
}

func (*capturingDispatchCapabilityRuntime) IssueRoutingOnlyGrant(context.Context, dispatchauthority.RoutingOnlyGrantIssueRequest) (string, error) {
	return "grant", nil
}

func (*capturingDispatchCapabilityRuntime) VerifyGrant(context.Context, string, dispatchauthority.GrantVerificationRequest) (dispatchauthority.VerifiedGrant, error) {
	return dispatchauthority.VerifiedGrant{}, nil
}

func (*capturingDispatchCapabilityRuntime) IssueFromGrant(context.Context, dispatchauthority.VerifiedGrant, dispatchauthority.FinalRequest) (string, error) {
	return "final-capability", nil
}

func (runtime *capturingDispatchCapabilityRuntime) VerifyDispatchOutcome(
	_ context.Context,
	token string,
	request dispatchauthority.OutcomeVerificationRequest,
) (backendinvoker.DispatchOutcome, error) {
	runtime.verifiedToken = token
	runtime.verificationRequest = request
	return runtime.outcome, runtime.verifyErr
}

func TestManagedDispatchCapabilityBindsFinalMutatedBody(t *testing.T) {
	runtime := &capturingDispatchCapabilityRuntime{}
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			ControlPlane: config.ControlPlaneConfig{Mode: config.ControlPlaneModeManaged},
			Access:       config.AccessServiceConfig{Enabled: true},
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					"public-model": {ResourceID: "model-id", ResourceRevision: 7},
				},
			},
		},
		DispatchCapabilities: runtime,
	}
	trace, err := routingcontext.WithGeneration(context.Background(), routingcontext.Generation{
		NamespaceID: "namespace", QuotaPartition: "partition", PublicationID: "publication",
		RuntimeEpoch: 2, SnapshotRevision: 11, RoutingDigest: strings.Repeat("a", 64),
	})
	if err != nil {
		t.Fatal(err)
	}
	admissionDigest := strings.Repeat("b", 64)
	dispatchPlanDigest := strings.Repeat("c", 64)
	ctx := &RequestContext{
		RequestID:    "request",
		TraceContext: trace,
		SourceFormat: llmprotocol.OpenAIChatV1,
		InferenceAccess: &inferenceRequestAccess{
			admission: &accessruntime.Admission{
				Tenant:        accessruntime.TenantContext{AdmissionID: "admission"},
				RequestDigest: admissionDigest,
			},
		},
		ManagedDispatch: &managedRequestDispatch{
			primaryDispatchID: "dispatch", primaryCandidateCount: 1,
			dispatches: []*inferenceDispatch{{
				id: "dispatch", model: "public-model", modelID: "model-id",
				modelRevision: 7, planDigest: dispatchPlanDigest, planned: true,
				dispatchType: "primary",
			}},
		},
	}
	ctx.SemanticRequest = testNeutralRequest("public-model", "route this request")
	ctx.SemanticRequest.Tools = []llmprotocol.Tool{{
		Name:        "lookup",
		InputSchema: []byte(`{"type":"object"}`),
	}}
	ctx.SemanticRequest.Generation++
	finalBody, err := router.encodeDispatchRequest(ctx)
	if err != nil {
		t.Fatal(err)
	}
	response := router.buildBackendDispatchResponse("public-model", []byte(`{"stage":"routing"}`), ctx)

	response = router.finalizeBackendDispatchResponse("public-model", response, ctx)
	if !bytes.Equal(runtime.issued.Final.Body, finalBody) {
		t.Fatalf("capability body = %q, want final body %q", runtime.issued.Final.Body, finalBody)
	}
	if body := response.GetRequestBody().GetResponse().GetBodyMutation().GetBody(); !bytes.Equal(body, finalBody) {
		t.Fatalf("forwarded body = %q, want capability body %q", body, finalBody)
	}
	if runtime.issued.Admission.RequestDigest != admissionDigest ||
		len(runtime.issued.Candidates) != 1 ||
		runtime.issued.Candidates[0].Dispatch.DispatchPlanDigest != dispatchPlanDigest {
		t.Fatalf("immutable identities = %+v", runtime.issued)
	}
	if got := requestMutationHeader(response, backendinvoker.DispatchCapabilityHeader); got != "final-capability" {
		t.Fatalf("dispatch capability header = %q", got)
	}
}

func requestMutationHeader(response *ext_proc.ProcessingResponse, key string) string {
	for _, option := range response.GetRequestBody().GetResponse().GetHeaderMutation().GetSetHeaders() {
		if option.GetHeader() != nil && strings.EqualFold(option.GetHeader().GetKey(), key) {
			return string(option.GetHeader().GetRawValue())
		}
	}
	return ""
}
