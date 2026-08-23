package backendinvoker

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const (
	testPublicationID = "publication"
	testRuntimeEpoch  = uint64(2)
	testRequestID     = "request"
)

var testRoutingDigest = strings.Repeat("d", 64)

const testChatResponseBody = `{"id":"chatcmpl-test","object":"chat.completion","created":1,"model":"provider-model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`

func testDispatchCandidate(dispatchID, modelID string, revision int64) DispatchCandidate {
	return DispatchCandidate{
		DispatchID: dispatchID, DispatchType: "primary", Ordinal: 0,
		DispatchPlanDigest: strings.Repeat("a", 64),
		ModelID:            modelID, ModelRevision: revision,
	}
}

func completeTestCapability(capability DispatchCapability) DispatchCapability {
	capability.PublicationID = testPublicationID
	capability.RuntimeEpoch = testRuntimeEpoch
	capability.RoutingDigest = testRoutingDigest
	capability.RequestID = testRequestID
	if capability.WireFormat == "" {
		capability.WireFormat = llmprotocol.OpenAIChatV1
	}
	return capability
}

func completeTestCapabilityIssue(request CapabilityIssueRequest) CapabilityIssueRequest {
	request.PublicationID = testPublicationID
	request.RuntimeEpoch = testRuntimeEpoch
	request.RoutingDigest = testRoutingDigest
	request.RequestID = testRequestID
	if request.WireFormat == "" {
		request.WireFormat = llmprotocol.OpenAIChatV1
	}
	return request
}

func completeTestGrantIssue(request DispatchGrantIssueRequest) DispatchGrantIssueRequest {
	request.PublicationID = testPublicationID
	request.RuntimeEpoch = testRuntimeEpoch
	request.RoutingDigest = testRoutingDigest
	request.RequestID = testRequestID
	return request
}

func completeTestGrant(grant DispatchGrant) DispatchGrant {
	grant.PublicationID = testPublicationID
	grant.RuntimeEpoch = testRuntimeEpoch
	grant.RoutingDigest = testRoutingDigest
	grant.RequestID = testRequestID
	return grant
}

func completeTestPlan(plan Plan) Plan {
	plan.PublicationID = testPublicationID
	plan.RuntimeEpoch = testRuntimeEpoch
	plan.RoutingDigest = testRoutingDigest
	plan.RequestID = testRequestID
	if plan.SourceFormat == "" {
		plan.SourceFormat = llmprotocol.OpenAIChatV1
	}
	return plan
}
