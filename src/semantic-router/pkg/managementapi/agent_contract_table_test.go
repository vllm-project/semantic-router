package managementapi

import (
	"sort"
	"strings"
	"testing"
)

type agentRouteContractExpectation struct {
	method      HTTPMethod
	path        string
	permission  string
	scope       OperationScope
	revision    RevisionMode
	idempotency IdempotencyMode
	async       AsyncMode
	pagination  PaginationMode
	status      string
	media       map[string]string
}

// agentRouteContractTable is the executable transport contract for every
// Router-owned Agent endpoint. Keep this table explicit: deriving expectations
// from agentOperations would let an accidental registry change bless itself.
var agentRouteContractTable = []agentRouteContractExpectation{
	{MethodGET, BasePath + "/agent-profiles", "agent.read@target", ScopeResultSet, RevisionNone, IdempotencyNone, AsyncSynchronous, PaginationKeyset, "200", agentJSON("AgentProfilePage")},
	{MethodPOST, BasePath + "/agent-profiles", "agent.manage@target", ScopeNamespace, RevisionReturns, IdempotencyRequired, AsyncSynchronous, PaginationNone, "201", agentJSON("MutationReceipt")},
	{MethodGET, BasePath + "/agent-profiles/{profile}", "agent.read@target", ScopeResource, RevisionReturns, IdempotencyNone, AsyncSynchronous, PaginationNone, "200", agentJSON("AgentProfileDetail")},
	{MethodPATCH, BasePath + "/agent-profiles/{profile}", "agent.manage@target", ScopeResource, RevisionCAS, IdempotencyNone, AsyncSynchronous, PaginationNone, "200", agentJSON("MutationReceipt")},
	{MethodDELETE, BasePath + "/agent-profiles/{profile}", "agent.manage@target", ScopeResource, RevisionCAS, IdempotencyNone, AsyncSynchronous, PaginationNone, "204", nil},

	{MethodGET, BasePath + "/agent-skills", "agent.read@target", ScopeResultSet, RevisionNone, IdempotencyNone, AsyncSynchronous, PaginationKeyset, "200", agentJSON("AgentSkillPage")},
	{MethodPOST, BasePath + "/agent-skills", "agent.manage@target", ScopeNamespace, RevisionReturns, IdempotencyRequired, AsyncSynchronous, PaginationNone, "201", agentJSON("MutationReceipt")},
	{MethodGET, BasePath + "/agent-skills/{skill}", "agent.read@target", ScopeResource, RevisionReturns, IdempotencyNone, AsyncSynchronous, PaginationNone, "200", agentJSON("AgentSkillDetail")},
	{MethodPATCH, BasePath + "/agent-skills/{skill}", "agent.manage@target", ScopeResource, RevisionCAS, IdempotencyNone, AsyncSynchronous, PaginationNone, "200", agentJSON("MutationReceipt")},
	{MethodDELETE, BasePath + "/agent-skills/{skill}", "agent.manage@target", ScopeResource, RevisionCAS, IdempotencyNone, AsyncSynchronous, PaginationNone, "204", nil},

	{MethodGET, BasePath + "/agent-tools", "tool.read@target", ScopeResultSet, RevisionNone, IdempotencyNone, AsyncSynchronous, PaginationKeyset, "200", agentJSON("AgentToolPage")},

	{MethodGET, BasePath + "/agent-tool-credentials", "tool.read@target", ScopeResultSet, RevisionNone, IdempotencyNone, AsyncSynchronous, PaginationKeyset, "200", agentJSON("AgentToolCredentialPage")},
	{MethodPOST, BasePath + "/agent-tool-credentials", "tool.manage@request_namespace", ScopeNamespace, RevisionReturns, IdempotencyRequired, AsyncSynchronous, PaginationNone, "201", agentJSON("MutationReceipt")},
	{MethodGET, BasePath + "/agent-tool-credentials/{credential}", "tool.read@target", ScopeResource, RevisionReturns, IdempotencyNone, AsyncSynchronous, PaginationNone, "200", agentJSON("AgentToolCredentialDetail")},
	{MethodPATCH, BasePath + "/agent-tool-credentials/{credential}", "tool.manage@target", ScopeResource, RevisionCAS, IdempotencyNone, AsyncSynchronous, PaginationNone, "200", agentJSON("MutationReceipt")},
	{MethodDELETE, BasePath + "/agent-tool-credentials/{credential}", "tool.manage@target", ScopeResource, RevisionCAS, IdempotencyNone, AsyncSynchronous, PaginationNone, "204", nil},
	{MethodPOST, BasePath + "/agent-tool-credentials/{credential}:rotate", "tool.manage@target", ScopeResource, RevisionCAS, IdempotencyRequired, AsyncSynchronous, PaginationNone, "200", agentJSON("MutationReceipt")},

	{MethodGET, BasePath + "/agent-tool-sources", "tool.read@target", ScopeResultSet, RevisionNone, IdempotencyNone, AsyncSynchronous, PaginationKeyset, "200", agentJSON("AgentToolSourcePage")},
	{MethodPOST, BasePath + "/agent-tool-sources", "tool.manage@target", ScopeNamespace, RevisionReturns, IdempotencyRequired, AsyncSynchronous, PaginationNone, "201", agentJSON("MutationReceipt")},
	{MethodGET, BasePath + "/agent-tool-sources/{source}", "tool.read@target", ScopeResource, RevisionReturns, IdempotencyNone, AsyncSynchronous, PaginationNone, "200", agentJSON("AgentToolSourceDetail")},
	{MethodPATCH, BasePath + "/agent-tool-sources/{source}", "tool.manage@target", ScopeResource, RevisionCAS, IdempotencyNone, AsyncSynchronous, PaginationNone, "200", agentJSON("MutationReceipt")},
	{MethodDELETE, BasePath + "/agent-tool-sources/{source}", "tool.manage@target", ScopeResource, RevisionCAS, IdempotencyNone, AsyncSynchronous, PaginationNone, "204", nil},
	{MethodPOST, BasePath + "/agent-tool-sources/{source}:test", "(tool.read@target AND tool.invoke@target)", ScopeResource, RevisionReturns, IdempotencyRequired, AsyncSynchronous, PaginationNone, "200", agentJSON("MutationReceipt")},
	{MethodPOST, BasePath + "/agent-tool-sources/{source}:approve", "tool.manage@target", ScopeResource, RevisionCAS, IdempotencyRequired, AsyncSynchronous, PaginationNone, "200", agentJSON("MutationReceipt")},

	{MethodGET, BasePath + "/agent-sessions", "agent.read@target", ScopeResultSet, RevisionNone, IdempotencyNone, AsyncSynchronous, PaginationKeyset, "200", agentJSON("AgentSessionPage")},
	{MethodPOST, BasePath + "/agent-sessions", "(agent.use@attributed_subject AND delegation.use@attributed_subject)", ScopeCompound, RevisionReturns, IdempotencyRequired, AsyncSynchronous, PaginationNone, "201", agentJSON("MutationReceipt")},
	{MethodGET, BasePath + "/agent-sessions/{session}", "agent.read@target", ScopeResource, RevisionReturns, IdempotencyNone, AsyncSynchronous, PaginationNone, "200", agentJSON("AgentSessionDetail")},
	{MethodPATCH, BasePath + "/agent-sessions/{session}", "agent.use@target", ScopeResource, RevisionCAS, IdempotencyNone, AsyncSynchronous, PaginationNone, "200", agentJSON("MutationReceipt")},
	{MethodDELETE, BasePath + "/agent-sessions/{session}", "agent.use@target", ScopeResource, RevisionCAS, IdempotencyNone, AsyncSynchronous, PaginationNone, "204", nil},
	{MethodPOST, BasePath + "/agent-sessions/{session}/turns", "agent.use@target", ScopeResource, RevisionNone, IdempotencyRequired, AsyncSynchronous, PaginationNone, "201", agentJSON("MutationReceipt")},
	{MethodGET, BasePath + "/agent-sessions/{session}/turns", "agent.read@target", ScopeResource, RevisionNone, IdempotencyNone, AsyncSynchronous, PaginationKeyset, "200", agentJSON("AgentTurnPage")},
	{MethodGET, BasePath + "/agent-sessions/{session}/events", "agent.read@target", ScopeResource, RevisionNone, IdempotencyNone, AsyncSynchronous, PaginationKeyset, "200", map[string]string{
		JSONMediaType:        "AgentEventPage",
		EventStreamMediaType: "AgentEventStream",
	}},
	{MethodPOST, BasePath + "/agent-sessions/{session}/turns/{turn}:cancel", "agent.use@target", ScopeResource, RevisionNone, IdempotencyRequired, AsyncSynchronous, PaginationNone, "200", agentJSON("MutationReceipt")},

	{MethodGET, BasePath + "/agent-artifacts/{artifact}", "agent.read@target", ScopeResource, RevisionNone, IdempotencyNone, AsyncSynchronous, PaginationNone, "200", agentJSON("AgentArtifactDetail")},
	{MethodGET, BasePath + "/agent-artifacts/{artifact}/content", "agent.read@target", ScopeResource, RevisionNone, IdempotencyNone, AsyncSynchronous, PaginationNone, "200", agentJSON("AgentArtifactContentDetail")},

	{MethodPOST, BasePath + "/publication-plans/{plan}:commit", "(routing.publish@target AND routing.read@all_dependencies)", ScopeCompound, RevisionCAS, IdempotencyRequired, AsyncOperation, PaginationNone, "202", agentJSON("MutationReceipt")},
}

func agentJSON(schema string) map[string]string {
	return map[string]string{JSONMediaType: schema}
}

func TestAgentRouteContractTable(t *testing.T) {
	if got, want := len(agentRouteContractTable), 36; got != want {
		t.Fatalf("Agent route contract table has %d rows, want %d", got, want)
	}

	document := GenerateOpenAPI()
	seen := make(map[string]struct{}, len(agentRouteContractTable))
	for _, expectation := range agentRouteContractTable {
		key := string(expectation.method) + " " + expectation.path
		if _, duplicate := seen[key]; duplicate {
			t.Fatalf("Agent route contract table contains duplicate %s", key)
		}
		seen[key] = struct{}{}

		t.Run(key, func(t *testing.T) {
			contract, found := LookupOperation(expectation.method, expectation.path)
			if !found {
				t.Fatalf("registry omits %s", key)
			}
			if got := contract.Permission.Canonical(); got != expectation.permission {
				t.Errorf("permission = %q, want %q", got, expectation.permission)
			}
			if contract.Scope != expectation.scope || contract.Revision != expectation.revision ||
				contract.Idempotency != expectation.idempotency || contract.Async != expectation.async ||
				contract.Pagination != expectation.pagination {
				t.Errorf("metadata = scope %s, revision %s, idempotency %s, async %s, pagination %s; want %s, %s, %s, %s, %s",
					contract.Scope, contract.Revision, contract.Idempotency, contract.Async, contract.Pagination,
					expectation.scope, expectation.revision, expectation.idempotency, expectation.async, expectation.pagination)
			}

			pathItem, found := document.Paths[expectation.path]
			if !found {
				t.Fatalf("generated OpenAPI omits path")
			}
			operation, found := pathItem[strings.ToLower(string(expectation.method))]
			if !found {
				t.Fatalf("generated OpenAPI omits method")
			}
			assertAgentOpenAPIMetadata(t, operation, expectation.permission, expectation.scope,
				expectation.revision, expectation.idempotency, expectation.async, expectation.pagination)
			assertAgentOpenAPIParameters(t, expectation, operation.Parameters)
			assertAgentOpenAPIResponse(t, expectation, operation.Responses)
			assertAgentOpenAPIPageRepresentation(t, document, expectation)
		})
	}

	registryAgentRoutes := make(map[string]struct{})
	for _, contract := range Operations() {
		if strings.HasPrefix(contract.Path, BasePath+"/agent-") ||
			contract.Path == BasePath+"/publication-plans/{plan}:commit" {
			registryAgentRoutes[string(contract.Method)+" "+contract.Path] = struct{}{}
		}
	}
	if len(registryAgentRoutes) != len(seen) {
		t.Errorf("registry has %d Agent routes, contract table has %d", len(registryAgentRoutes), len(seen))
	}
	for key := range registryAgentRoutes {
		if _, covered := seen[key]; !covered {
			t.Errorf("registry Agent route %s is not pinned by the contract table", key)
		}
	}
}

func assertAgentOpenAPIPageRepresentation(
	t *testing.T,
	document OpenAPIDocument,
	expectation agentRouteContractExpectation,
) {
	t.Helper()
	if expectation.pagination != PaginationKeyset {
		return
	}
	schemaName, found := expectation.media[JSONMediaType]
	if !found {
		t.Fatal("keyset collection has no JSON page representation")
	}
	page, found := document.Components.Schemas[schemaName]
	if !found {
		t.Fatalf("page schema %s is absent", schemaName)
	}
	if got, want := page.Properties["page"].Ref, "#/components/schemas/PageInfo"; got != want {
		t.Errorf("%s page metadata = %q, want %q", schemaName, got, want)
	}
	data := page.Properties["data"]
	if data.Type != "array" || data.Items == nil || data.Items.Ref == "" {
		t.Errorf("%s data is not a typed array: %#v", schemaName, data)
	}
}

func assertAgentOpenAPIMetadata(
	t *testing.T,
	operation OpenAPIOperation,
	permission string,
	scope OperationScope,
	revision RevisionMode,
	idempotency IdempotencyMode,
	async AsyncMode,
	pagination PaginationMode,
) {
	t.Helper()
	if operation.RouterPermissionCanonical != permission || operation.RouterScope != scope ||
		operation.RouterRevision != revision || operation.RouterIdempotency != idempotency ||
		operation.RouterAsync != async || operation.RouterPagination != pagination {
		t.Errorf("OpenAPI metadata = permission %q, scope %s, revision %s, idempotency %s, async %s, pagination %s; want %q, %s, %s, %s, %s, %s",
			operation.RouterPermissionCanonical, operation.RouterScope, operation.RouterRevision,
			operation.RouterIdempotency, operation.RouterAsync, operation.RouterPagination,
			permission, scope, revision, idempotency, async, pagination)
	}
}

func assertAgentOpenAPIParameters(
	t *testing.T,
	expectation agentRouteContractExpectation,
	parameters []OpenAPIParameter,
) {
	t.Helper()
	want := []string{"header:Accept", "header:" + HeaderNamespaceID}
	for _, match := range openAPIPathParameterPattern.FindAllStringSubmatch(expectation.path, -1) {
		want = append(want, "path:"+match[1])
	}
	if expectation.pagination == PaginationKeyset {
		want = append(want, "query:cursor", "query:pageSize")
	}
	if agentSearchableCollection(OperationContract{Method: expectation.method, Path: expectation.path}) {
		want = append(want, "query:search")
	}
	if expectation.idempotency == IdempotencyRequired {
		want = append(want, "header:"+HeaderIdempotencyKey)
	}
	if expectation.revision == RevisionCAS {
		want = append(want, "header:"+HeaderIfMatch)
	}
	if expectation.method == MethodGET && expectation.path == BasePath+"/agent-sessions/{session}/events" {
		want = append(want, "query:afterSequence", "header:Last-Event-ID")
	}
	sort.Strings(want)

	got := make([]string, 0, len(parameters))
	for _, parameter := range parameters {
		got = append(got, parameter.In+":"+parameter.Name)
	}
	sort.Strings(got)
	if strings.Join(got, "\n") != strings.Join(want, "\n") {
		t.Errorf("OpenAPI parameters = %v, want %v", got, want)
	}
}

func assertAgentOpenAPIResponse(
	t *testing.T,
	expectation agentRouteContractExpectation,
	responses map[string]OpenAPIResponse,
) {
	t.Helper()
	var successStatuses []string
	for status := range responses {
		if strings.HasPrefix(status, "2") {
			successStatuses = append(successStatuses, status)
		}
	}
	sort.Strings(successStatuses)
	if len(successStatuses) != 1 || successStatuses[0] != expectation.status {
		t.Fatalf("success statuses = %v, want [%s]", successStatuses, expectation.status)
	}

	response := responses[expectation.status]
	if len(response.Content) != len(expectation.media) {
		t.Errorf("success representations = %v, want %v", agentMediaNames(response.Content), agentExpectedMediaNames(expectation.media))
	}
	for mediaType, schema := range expectation.media {
		media, found := response.Content[mediaType]
		if !found {
			t.Errorf("success response omits %s", mediaType)
			continue
		}
		if got, want := media.Schema.Ref, "#/components/schemas/"+schema; got != want {
			t.Errorf("%s response schema = %q, want %q", mediaType, got, want)
		}
	}

	_, hasETag := response.Headers[HeaderETag]
	if want := expectation.revision != RevisionNone; hasETag != want {
		t.Errorf("success ETag present = %t, want %t", hasETag, want)
	}
	_, hasReplay := response.Headers[HeaderIdempotencyReplayed]
	if want := expectation.idempotency == IdempotencyRequired; hasReplay != want {
		t.Errorf("idempotency replay header present = %t, want %t", hasReplay, want)
	}
}

func agentMediaNames(content map[string]OpenAPIMedia) []string {
	result := make([]string, 0, len(content))
	for mediaType := range content {
		result = append(result, mediaType)
	}
	sort.Strings(result)
	return result
}

func agentExpectedMediaNames(content map[string]string) []string {
	result := make([]string, 0, len(content))
	for mediaType := range content {
		result = append(result, mediaType)
	}
	sort.Strings(result)
	return result
}
