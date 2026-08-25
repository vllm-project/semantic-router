package managementapi

import (
	"slices"
	"strings"
	"testing"
)

func TestAgentMutationContractsPinCASAndIdempotency(t *testing.T) {
	for _, test := range []struct {
		method      HTTPMethod
		path        string
		revision    RevisionMode
		idempotency IdempotencyMode
		scope       OperationScope
	}{
		{MethodPOST, BasePath + "/agent-profiles", RevisionReturns, IdempotencyRequired, ScopeNamespace},
		{MethodPATCH, BasePath + "/agent-profiles/{profile}", RevisionCAS, IdempotencyNone, ScopeResource},
		{MethodPATCH, BasePath + "/agent-sessions/{session}", RevisionCAS, IdempotencyNone, ScopeResource},
		{MethodPOST, BasePath + "/agent-sessions/{session}/turns", RevisionNone, IdempotencyRequired, ScopeResource},
		{MethodPOST, BasePath + "/agent-tool-credentials/{credential}:rotate", RevisionCAS, IdempotencyRequired, ScopeResource},
		{MethodPOST, BasePath + "/publication-plans/{plan}:commit", RevisionCAS, IdempotencyRequired, ScopeCompound},
	} {
		contract, found := LookupOperation(test.method, test.path)
		if !found {
			t.Fatalf("operation %s %s is absent", test.method, test.path)
		}
		if contract.Revision != test.revision || contract.Idempotency != test.idempotency || contract.Scope != test.scope {
			t.Errorf("%s %s metadata = revision %s, idempotency %s, scope %s; want %s, %s, %s",
				test.method, test.path, contract.Revision, contract.Idempotency, contract.Scope,
				test.revision, test.idempotency, test.scope)
		}
	}
}

func TestAgentCollectionsDoNotAuthorizeCreateAgainstInventedResource(t *testing.T) {
	for _, path := range []string{
		BasePath + "/agent-profiles",
		BasePath + "/agent-skills",
		BasePath + "/agent-tool-sources",
	} {
		list, _ := LookupOperation(MethodGET, path)
		create, _ := LookupOperation(MethodPOST, path)
		if list.Scope != ScopeResultSet || create.Scope != ScopeNamespace {
			t.Fatalf("%s scopes = list %s/create %s", path, list.Scope, create.Scope)
		}
	}
}

func TestAgentEventContractProvidesDurableHistoryAndSSE(t *testing.T) {
	document := GenerateOpenAPI()
	operation := document.Paths[BasePath+"/agent-sessions/{session}/events"]["get"]
	accept, found := openAPIParameter(operation.Parameters, "Accept", "header")
	if !found || !accept.Required || !slices.Equal(accept.Schema.Enum, []string{JSONMediaType, EventStreamMediaType}) {
		t.Fatalf("Agent event Accept contract = %#v", accept)
	}
	response := operation.Responses["200"]
	if response.Content[JSONMediaType].Schema.Ref != "#/components/schemas/AgentEventPage" {
		t.Fatal("Agent event JSON response is not a typed durable page")
	}
	if response.Content[EventStreamMediaType].Schema.Ref != "#/components/schemas/AgentEventStream" {
		t.Fatal("Agent event response omits SSE")
	}
	if operation.Responses["410"].Content[JSONMediaType].Schema.Ref != "#/components/schemas/AgentEventHistoryExpiredError" {
		t.Fatal("Agent event response omits typed retention recovery")
	}
	assistant := document.Components.Schemas["AgentAssistantDeltaEventPayload"]
	for _, field := range []string{"modelStepId", "chunkIndex", "delta"} {
		if !slices.Contains(assistant.Required, field) {
			t.Fatalf("durable assistant delta omits reconciliation field %s", field)
		}
	}
	summary := document.Components.Schemas["AgentModelStepSummaryEventPayload"]
	for _, field := range []string{"modelStepId", "requestId", "latencyMilliseconds"} {
		if !slices.Contains(summary.Required, field) {
			t.Fatalf("durable model-step summary omits field %s", field)
		}
	}
	if _, leaked := summary.Properties["providerOpaque"]; leaked {
		t.Fatal("durable model-step summary exposes provider-opaque data")
	}
	if _, guessed := summary.Properties["cost"]; guessed {
		t.Fatal("durable model-step summary exposes non-authoritative cost")
	}
	usage := document.Components.Schemas["AgentModelStepUsage"]
	for _, field := range []string{"inputTokens", "outputTokens", "totalTokens"} {
		if !slices.Contains(usage.Required, field) {
			t.Fatalf("durable model-step usage omits authoritative total %s", field)
		}
	}
	if len(document.Components.Schemas["AgentLiveModelStepEvent"].OneOf) != 2 {
		t.Fatal("live Agent model step does not distinguish delta and terminal frames")
	}
	event := document.Components.Schemas["AgentEvent"]
	if len(event.OneOf) != 11 {
		t.Fatalf("durable Agent event has %d variants, want 11", len(event.OneOf))
	}
	for _, branch := range event.OneOf {
		if !strings.HasPrefix(branch.Ref, "#/components/schemas/Agent") ||
			!strings.HasSuffix(branch.Ref, "Event") {
			t.Fatalf("durable Agent event branch is not a typed event: %q", branch.Ref)
		}
	}
	stream := document.Components.Schemas["AgentEventStream"]
	if !strings.Contains(stream.Description, "without an id") ||
		!strings.Contains(stream.Description, "never replayed") {
		t.Fatal("Agent SSE contract does not separate provisional and durable resume semantics")
	}
}

func TestAgentOpenAPISchemasKeepDelegationAndSecretsPrivate(t *testing.T) {
	schemas := GenerateOpenAPI().Components.Schemas
	sessionInput := schemas["AgentSessionCreateRequest"]
	if _, leaked := sessionInput.Properties["delegatedInferenceSessionId"]; leaked {
		t.Fatal("Agent session input exposes delegated inference implementation state")
	}
	if _, found := sessionInput.Properties["keyId"]; !found || !slices.Contains(sessionInput.Required, "keyId") {
		t.Fatal("Agent session input does not pin one eligible API key")
	}
	session := schemas["AgentSession"]
	if _, found := session.Properties["keyId"]; !found || !slices.Contains(session.Required, "keyId") {
		t.Fatal("Agent session output cannot restore its pinned API key")
	}
	credential := schemas["AgentToolCredential"]
	for _, field := range []string{"secret", "ciphertext", "activeVersionId"} {
		if _, leaked := credential.Properties[field]; leaked {
			t.Fatalf("Agent credential metadata exposes %s", field)
		}
	}
	tool := schemas["AgentToolDefinition"]
	for _, field := range []string{"requiredPermissions", "class", "idempotency", "timeoutMilliseconds"} {
		if _, found := tool.Properties[field]; !found {
			t.Fatalf("Agent Tool schema omits %s", field)
		}
	}
}

func TestAgentWaitingApprovalIsNotATerminalEvent(t *testing.T) {
	status := GenerateOpenAPI().Components.Schemas["AgentTerminalEventPayload"].Properties["status"]
	for _, candidate := range status.Enum {
		if candidate == "waiting_approval" {
			t.Fatal("waiting_approval is represented by approval_request, not terminal")
		}
	}
}
